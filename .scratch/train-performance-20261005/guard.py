"""等待正式提交后暂停训练，执行临时实验，结束或空闲超时后恢复原训练。"""
import json
import os
from pathlib import Path
import signal
import subprocess
import time
import traceback
from datetime import datetime

ROOT = Path('/home/tml104/Mortal')
WORK = Path(__file__).resolve().parent
PID = 381125
CHECKPOINT = ROOT / 'artifacts/mortal_precomputed_v2/latest.pth'
PYTHON = '/home/tml104/miniconda3/envs/mortal/bin/python'


def report(phase, **fields):
    value = {'phase': phase, 'time': datetime.now().isoformat(), 'guard_pid': os.getpid(), **fields}
    temporary = WORK / 'status.tmp'
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(WORK / 'status.json')
    print(json.dumps(value), flush=True)


def alive(pid):
    try:
        return Path(f'/proc/{pid}/stat').read_text().split(') ', 1)[1].split()[0] != 'Z'
    except FileNotFoundError:
        return False


def main():
    args = [x.decode() for x in Path(f'/proc/{PID}/cmdline').read_bytes().split(b'\0') if x]
    if not any(x.endswith('precompute_pipeline/train/train.py') for x in args):
        raise RuntimeError('目标 PID 不是预期训练入口')
    environment = os.environ.copy()
    environment.update({item.split(b'=', 1)[0].decode(): item.split(b'=', 1)[1].decode()
                        for item in Path(f'/proc/{PID}/environ').read_bytes().split(b'\0') if b'=' in item})
    stat = CHECKPOINT.stat()
    old = (stat.st_ino, stat.st_mtime_ns)
    report('waiting_checkpoint', training_pid=PID)
    deadline = time.monotonic() + 240
    while True:
        if not alive(PID):
            raise RuntimeError('正式训练在暂停前已经退出')
        current = CHECKPOINT.stat()
        if (current.st_ino, current.st_mtime_ns) != old:
            break
        if time.monotonic() > deadline:
            raise RuntimeError('未等到新的 checkpoint，保留正式训练')
        time.sleep(.2)
    stopped = False
    child = None
    try:
        os.kill(PID, signal.SIGTERM)
        stopped = True
        for _ in range(100):
            if not alive(PID): break
            time.sleep(.1)
        if alive(PID):
            raise RuntimeError('正式进程未退出；不运行性能实验')
        import torch
        state = torch.load(CHECKPOINT, weights_only=True, map_location='cpu')
        snapshot = {'batches_completed': state['batches_completed'],
                    'epoch_samples_consumed': state['epoch_samples_consumed'],
                    'current_epoch': state['current_epoch'], 'main_seed': state['main_seed']}
        (WORK / 'paused-checkpoint.json').write_text(json.dumps(snapshot, indent=2) + '\n')
        del state
        report('paused', snapshot=snapshot)
        completed = set()
        last_work = time.monotonic()
        while not (WORK / 'finish').exists():
            jobs = sorted((WORK / 'jobs').glob('*.json'))
            pending = [path for path in jobs if path.name not in completed]
            if not pending:
                if time.monotonic() - last_work > 180:
                    break
                time.sleep(2)
                continue
            path = pending[0]
            job = json.loads(path.read_text())
            command = [PYTHON, '-u', str(WORK / 'bench.py'), '--name', job['name'], *job.get('args', [])]
            case_env = environment.copy()
            case_env['TORCHINDUCTOR_COMPILE_THREADS'] = '2'
            report('benchmark', job=job, completed=len(completed), snapshot=snapshot)
            with (WORK / f'{job["name"]}.log').open('ab', buffering=0) as output:
                child = subprocess.Popen(command, cwd=ROOT, env=case_env, stdin=subprocess.DEVNULL,
                                         stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                try:
                    code = child.wait(timeout=job.get('timeout', 360))
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGTERM)
                    try:
                        child.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(child.pid, signal.SIGKILL)
                        child.wait()
                    code = 'timeout'
                child = None
            completed.add(path.name)
            last_work = time.monotonic()
            report('benchmark_finished', name=job['name'], exit_code=code, completed=len(completed), snapshot=snapshot)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            child.wait(timeout=10)
        if stopped and not alive(PID):
            with (WORK / 'resumed-train.log').open('ab', buffering=0) as output:
                resumed = subprocess.Popen(args, cwd=ROOT, env=environment, stdin=subprocess.DEVNULL,
                                           stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
            (WORK / 'resumed-train.pid').write_text(str(resumed.pid) + '\n')
            report('training_resumed', training_pid=resumed.pid)


if __name__ == '__main__':
    try:
        main()
    except BaseException:
        print(traceback.format_exc(), flush=True)
        raise
