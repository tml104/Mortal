"""一次性操作：紧跟 checkpoint 保存重启训练，并确认恢复后再次保存。"""

from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


ROOT = Path('/home/tml104/Mortal')
CONFIG = ROOT / 'mortal/precompute_pipeline/config.toml'
WORK = Path(__file__).resolve().parent
OLD_PID = 257095
SCRIPT = 'mortal/precompute_pipeline/train/train.py'
PYTHON = '/home/tml104/miniconda3/envs/mortal/bin/python'
sys.path.insert(0, str(ROOT / 'mortal'))
from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.train.train import _load_checkpoint, _training_contract


def now():
    """记录本机使用的北京时间。"""
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def report(phase, **fields):
    """操作记录写入独立目录，不修改训练 checkpoint。"""
    value = {'time': now(), 'phase': phase, **fields}
    temporary = WORK / 'status.json.tmp'
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(WORK / 'status.json')
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value


def alive(pid):
    """已退出但尚未被父进程回收的僵尸进程也视为退出。"""
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def identify_old():
    """发信号前核对原训练入口、配置和工作目录。"""
    arguments = Path(f'/proc/{OLD_PID}/cmdline').read_bytes().split(b'\0')
    arguments = [arg.decode() for arg in arguments if arg]
    if SCRIPT not in arguments or '--config' not in arguments:
        raise RuntimeError('原 PID 已不是指定训练入口')
    if arguments[arguments.index('--config') + 1] != str(CONFIG):
        raise RuntimeError('原 PID 的配置路径不匹配')
    if Path(os.readlink(f'/proc/{OLD_PID}/cwd')) != ROOT:
        raise RuntimeError('原 PID 的工作目录不匹配')


def version(path):
    """训练器通过原子替换提交 checkpoint，文件身份变化即表示提交完成。"""
    metadata = path.stat()
    return metadata.st_ino, metadata.st_mtime_ns


def snapshot(path, contract):
    """用正式恢复校验读取 checkpoint，只保留轻量进度字段。"""
    state = _load_checkpoint(path, contract)
    names = ('main_seed', 'completed_epochs', 'current_epoch',
             'epoch_samples_consumed', 'samples_seen', 'batches_completed')
    result = {name: state[name] for name in names}
    result['last_validation'] = state.get('last_validation')
    result['optimizer_present'] = bool(state['optimizer']['state'])
    result['amp_present'] = bool(state['scaler'])
    result['cuda_rng_present'] = 'cuda_rng_state_all' in state
    return result


def main():
    """仅重启用户批准的进程，沿用原 checkpoint 和训练输出目录。"""
    os.chdir(ROOT)
    settings = load_settings(CONFIG).training
    if settings.num_epochs != 30:
        raise RuntimeError('当前配置的 num_epochs 不是 30')
    dataset = load_dataset_state(settings.dataset_progress)
    contract = _training_contract(settings, dataset['generation_id'])
    latest = settings.output_dir / 'latest.pth'
    identify_old()
    previous_version = version(latest)
    before = snapshot(latest, contract)
    report('waiting_for_next_checkpoint', old_pid=OLD_PID, before=before)
    started = time.monotonic()
    while version(latest) == previous_version:
        if not alive(OLD_PID):
            raise RuntimeError('等待新 checkpoint 时原训练进程意外退出')
        if time.monotonic() - started > 600:
            raise RuntimeError('10 分钟未发现新 checkpoint，保留原进程供检查')
        time.sleep(0.05)

    identify_old()
    os.kill(OLD_PID, signal.SIGINT)
    report('stopping_old_process', old_pid=OLD_PID)
    deadline = time.monotonic() + 60
    while alive(OLD_PID):
        if time.monotonic() >= deadline:
            raise RuntimeError('原进程中断后 60 秒仍未退出，尚未启动第二个训练进程')
        time.sleep(0.1)
    restored = snapshot(latest, contract)
    if restored['main_seed'] != before['main_seed'] or restored['batches_completed'] < before['batches_completed']:
        raise RuntimeError('停止后 checkpoint 的种子或进度与原训练不一致')
    current_settings = load_settings(CONFIG).training
    if current_settings.num_epochs != 30 or _training_contract(current_settings, dataset['generation_id']) != contract:
        raise RuntimeError('切换期间训练配置发生变化')
    restored_version = version(latest)
    started_wall = time.time()
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    with (WORK / 'train.log').open('ab', buffering=0) as output:
        process = subprocess.Popen(
            [PYTHON, SCRIPT, '--config', str(CONFIG)], cwd=ROOT,
            stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
            env=environment, start_new_session=True,
        )
    (WORK / 'train.pid').write_text(str(process.pid) + '\n')
    report('training_restarted', old_pid=OLD_PID, training_pid=process.pid,
           target_epochs=30, before=before, restored=restored)

    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    accumulator = EventAccumulator(str(settings.output_dir / 'tensorboard'), size_guidance={'scalars': 0})
    saved = None
    observed_step = restored['batches_completed']
    observed_metrics = {}
    observation_started = time.monotonic()
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'恢复后的训练提前退出，exit={process.returncode}，见 train.log')
        accumulator.Reload()
        tags = accumulator.Tags().get('scalars', [])
        recent = [event for event in accumulator.Scalars('train/loss')
                  if event.wall_time >= started_wall and event.step > restored['batches_completed']] if 'train/loss' in tags else []
        if recent:
            observed_step = recent[-1].step
            observed_metrics = {}
            for tag in tags:
                if tag.startswith('train/'):
                    events = [event for event in accumulator.Scalars(tag)
                              if event.step == observed_step and event.wall_time >= started_wall]
                    if events:
                        observed_metrics[tag] = events[-1].value
            if not all(math.isfinite(value) for value in observed_metrics.values()):
                raise RuntimeError('恢复后的训练指标出现 NaN/Inf')
        if version(latest) != restored_version:
            saved = snapshot(latest, contract)
            if saved['main_seed'] != restored['main_seed'] or saved['batches_completed'] <= restored['batches_completed']:
                raise RuntimeError('恢复后的 checkpoint 未沿用原种子或未向前推进')
            restored_version = version(latest)
        elapsed = time.monotonic() - observation_started
        confirmed = saved is not None and observed_step >= restored['batches_completed'] + 200 and len(observed_metrics) >= 7 and elapsed >= 120
        result = report(
            'resume_confirmed' if confirmed else 'observing_resumed_training',
            old_pid=OLD_PID, training_pid=process.pid, target_epochs=30,
            restored=restored, saved=saved, observed_step=observed_step,
            metrics=observed_metrics, observation_seconds=round(elapsed, 1),
            checkpoint=str(latest), train_log=str(WORK / 'train.log'),
        )
        if confirmed:
            (WORK / 'result.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
            return
        if elapsed > 900:
            raise RuntimeError('观察恢复训练 15 分钟仍未满足确认条件，保留训练进程供检查')
        time.sleep(15)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        report('failed', error=str(error), traceback=traceback.format_exc())
        raise
