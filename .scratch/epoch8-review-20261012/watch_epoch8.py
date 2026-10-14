"""本次运行专用：第八轮完整 checkpoint 保存后退出训练进程。"""

from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import time
from zoneinfo import ZoneInfo

import torch


ROOT = Path('/home/tml104/Mortal')
WORK = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'artifacts/mortal_precomputed_optimize/latest.pth'
LOG = WORK / 'training.log'
PID = 11531
START_TIME = '140925'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def process_state():
    """核对进程启动时间，避免向被重新使用的 PID 发送信号。"""
    try:
        fields = Path(f'/proc/{PID}/stat').read_text().split(') ', 1)[1].split()
    except FileNotFoundError:
        return None
    if fields[19] != START_TIME:
        raise RuntimeError('目标训练 PID 已被重新使用')
    return fields[0]


def report(phase, **fields):
    state = {'phase': phase, 'time': now(), 'watch_pid': os.getpid(), 'training_pid': PID, **fields}
    target = WORK / 'status.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(state, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(state, ensure_ascii=False), flush=True)


def read_checkpoint():
    """安全读取已原子提交的训练状态，返回简洁的恢复进度。"""
    state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
    summary = {key: state[key] for key in (
        'completed_epochs', 'current_epoch', 'epoch_samples_consumed',
        'samples_seen', 'batches_completed', 'last_validation',
    )}
    return summary


def stop_after_epoch8():
    summary = read_checkpoint()
    if summary['completed_epochs'] < 8:
        raise RuntimeError('尚未保存第八轮完整 checkpoint')
    if summary['last_validation']['kind'] != 'epoch':
        raise RuntimeError('检测点不是完整 epoch 验证保存点')
    report('epoch8_saved', checkpoint=summary)
    if process_state() not in (None, 'Z'):
        os.kill(PID, signal.SIGINT)
        report('stopping', checkpoint=summary)
        started = time.monotonic()
        while process_state() not in (None, 'Z'):
            if time.monotonic() - started > 60:
                raise RuntimeError('发送 SIGINT 后 60 秒仍未退出')
            time.sleep(0.2)
    saved = read_checkpoint()
    if saved != summary:
        raise RuntimeError('停止期间 checkpoint 发生意外变化')
    (WORK / 'paused-checkpoint.json').write_text(json.dumps(saved, ensure_ascii=False, indent=2) + '\n')
    report('paused', checkpoint=saved)


def main():
    os.chdir(ROOT)
    (WORK / 'watch.pid').write_text(str(os.getpid()) + '\n')
    summary = read_checkpoint()
    if summary['completed_epochs'] >= 8:
        stop_after_epoch8()
        return
    latest = None
    last_report = 0
    with LOG.open() as stream:
        stream.seek(0, 2)
        while True:
            line = stream.readline()
            if line:
                fields = dict(re.findall(r'(\w+)=([^\s]+)', line))
                if fields.get('event') in {'train', 'validation', 'validation_start', 'validation_end', 'checkpoint'}:
                    latest = fields
                if fields.get('event') == 'epoch_complete' and int(fields['completed_epochs']) >= 8:
                    stop_after_epoch8()
                    return
            if process_state() in (None, 'Z'):
                raise RuntimeError('目标训练进程在第八轮保存前退出')
            current = time.monotonic()
            if current - last_report >= 30:
                report('waiting_epoch8', latest_event=latest)
                last_report = current
            if not line:
                time.sleep(0.1)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        report('failed', error=str(error))
        raise
