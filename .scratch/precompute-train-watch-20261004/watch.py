"""一次性接力监视：预计算完成 → 核对文件 → 后台训练 → 有限健康观察。"""

from collections import defaultdict
from datetime import datetime
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from zoneinfo import ZoneInfo


REPOSITORY = Path('/home/tml104/Mortal')
CONFIG = REPOSITORY / 'mortal/precompute_pipeline/config.toml'
PROGRESS = REPOSITORY / 'data/precomputed_chunks/progress.json'
WORK = Path(__file__).resolve().parent
PYTHON = '/home/tml104/miniconda3/envs/mortal/bin/python'
PRECOMPUTE_SCRIPT = 'mortal/precompute_pipeline/precompute/precompute.py'
TRAIN_SCRIPT = 'mortal/precompute_pipeline/train/train.py'
METRICS = ('loss', 'dqn_loss', 'cql_loss', 'rank_loss', 'rank_accuracy', 'q_mean', 'q_target_mean')
sys.path.insert(0, str(REPOSITORY / 'mortal'))
from precompute_pipeline.common.dataset import iter_chunk_records, load_dataset_state
from precompute_pipeline.common.settings import load_settings


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def report(phase, **fields):
    """监视状态独立保存，不修改预计算恢复 JSON。"""
    value = {'phase': phase, 'time': now(), 'watch_pid': os.getpid(), **fields}
    target = WORK / 'status.json'
    temporary = target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(target)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def matching_processes(script):
    """只识别本仓库入口和同一配置，避免启动第二个训练进程。"""
    matches = []
    for directory in Path('/proc').iterdir():
        if not directory.name.isdigit() or int(directory.name) == os.getpid():
            continue
        try:
            arguments = [part.decode() for part in (directory / 'cmdline').read_bytes().split(b'\0') if part]
            if not any(arg.endswith(script) for arg in arguments) or '--config' not in arguments:
                continue
            cwd = Path(os.readlink(directory / 'cwd'))
            config = Path(arguments[arguments.index('--config') + 1])
            if not config.is_absolute():
                config = cwd / config
            if config.resolve() != CONFIG.resolve():
                continue
            state = (directory / 'stat').read_text().split(') ', 1)[1].split()[0]
            if state != 'Z':
                matches.append(int(directory.name))
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            continue
    return matches


def verify_chunk_files(progress_file):
    """仅扫描文件名和元数据，核对所有登记块；不读取或解压内容。"""
    state = load_dataset_state(progress_file)
    groups = defaultdict(set)
    totals = {}
    physical_bytes = {'c': 0, 'd': 0}
    directory_volume = {}
    for split in ('train', 'val'):
        info = state['splits'][split]
        if not info['complete'] or info['chunk_count'] <= 0:
            raise RuntimeError(f'{split} 尚未完成或没有 chunks')
        records = iter_chunk_records(progress_file, split)
        if sum(record.sample_count for record in records) != info['sample_count']:
            raise RuntimeError(f'{split} 样本计数与 chunk 数不一致')
        for record in records:
            groups[record.path.parent].add(record.path.name)
            directory_volume[record.path.parent] = record.volume
        totals[split] = {'chunks': len(records), 'samples': info['sample_count'], 'source_files': info['source_file_count']}
    extras = {}
    for directory, expected in groups.items():
        found = set()
        extra = []
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name in expected:
                    if not entry.is_file() or entry.stat().st_size <= 0:
                        raise RuntimeError(f'登记 chunk 不是非空文件: {entry.path}')
                    found.add(entry.name)
                    physical_bytes[directory_volume[directory]] += entry.stat().st_size
                elif entry.name.startswith('chunk_'):
                    extra.append(entry.name)
        missing = expected - found
        if missing:
            raise RuntimeError(f'{directory} 缺少 {len(missing)} 块: {sorted(missing)[:8]}')
        if extra:
            extras[str(directory)] = {'count': len(extra), 'examples': sorted(extra)[:8]}
    if physical_bytes != state['volume_bytes']:
        raise RuntimeError(f'登记块总字节数不匹配: {physical_bytes} != {state["volume_bytes"]}')
    return {'generation_id': state['generation_id'], 'chunk_size': state['chunk_size'], 'splits': totals, 'volume_bytes': physical_bytes, 'extra_files': extras, 'content_checked': False}


def observe_training(process, settings, verification, initial_step, started_by_watch):
    """用实际 TensorBoard 训练步数与有限指标确认跨多块训练正常。"""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    event_dir = settings.output_dir / 'tensorboard'
    accumulator = None
    started = time.monotonic()
    latest_step = initial_step
    latest_metrics = {}
    while True:
        if process.poll() is not None:
            raise RuntimeError(f'训练进程提前退出，退出码 {process.returncode}；见 train.log')
        if event_dir.exists():
            if accumulator is None:
                accumulator = EventAccumulator(str(event_dir), size_guidance={'scalars': 0})
            accumulator.Reload()
            tags = accumulator.Tags().get('scalars', [])
            if 'train/loss' in tags:
                candidates = [event for event in accumulator.Scalars('train/loss') if event.step > initial_step]
                if candidates:
                    latest_step = max(event.step for event in candidates)
                    current = {}
                    for name in METRICS:
                        tag = 'train/' + name
                        if tag not in tags:
                            continue
                        values = [event for event in accumulator.Scalars(tag) if event.step == latest_step]
                        if values:
                            current[name] = values[-1].value
                    if any(not math.isfinite(value) for value in current.values()):
                        if started_by_watch:
                            process.terminate()
                        raise RuntimeError(f'训练指标出现非有限值: step={latest_step}, values={current}')
                    if len(current) == len(METRICS):
                        latest_metrics = current
        duration = time.monotonic() - started
        observed_batches = latest_step - initial_step
        new_samples = observed_batches * settings.batch_size
        confirmed = observed_batches >= 200 and len(latest_metrics) == len(METRICS) and duration >= 120
        fields = {
            'training_pid': process.pid, 'initial_batch_step': initial_step,
            'observed_batch_step': latest_step, 'new_batches_observed': observed_batches,
            'new_samples_lower_bound': new_samples,
            'chunks_consumed_lower_bound': math.ceil(new_samples / verification['chunk_size']),
            'metrics': latest_metrics, 'observation_seconds': round(duration, 1),
            'checkpoint_exists': (settings.output_dir / 'latest.pth').exists(),
            'training_output': str(settings.output_dir), 'train_log': str(WORK / 'train.log'),
            'verification': verification,
        }
        report('training_confirmed' if confirmed else 'observing_training', **fields)
        if confirmed:
            (WORK / 'result.json').write_text(json.dumps({'completed_at': now(), **fields}, ensure_ascii=False, indent=2) + '\n')
            return
        if duration > 900 and latest_step == initial_step:
            raise RuntimeError('观察训练 15 分钟仍未看到新增 TensorBoard 训练步；保留进程供调查')
        time.sleep(15)


def main():
    os.chdir(REPOSITORY)
    (WORK / 'watch.pid').write_text(str(os.getpid()) + '\n')
    initial = load_dataset_state(PROGRESS, require_complete=False)
    generation_id = initial['generation_id']
    while True:
        state = load_dataset_state(PROGRESS, require_complete=False)
        if state['generation_id'] != generation_id:
            raise RuntimeError('预计算 generation_id 发生变化，停止本次接力')
        processes = matching_processes(PRECOMPUTE_SCRIPT)
        if state['status'] == 'complete' and not processes:
            break
        report('waiting_precompute', precompute_pids=processes,
               split=state['current_split'], cursor=state['source_cursor'],
               splits=state['splits'], volume_bytes=state['volume_bytes'])
        if not processes and state['status'] != 'complete':
            raise RuntimeError('预计算进程已退出但数据未完成；未启动训练')
        time.sleep(30)
    report('verifying_chunk_files')
    verification = verify_chunk_files(PROGRESS)
    (WORK / 'verification.json').write_text(json.dumps(verification, ensure_ascii=False, indent=2) + '\n')
    report('chunk_files_verified', verification=verification)
    settings = load_settings(CONFIG).training
    if settings is None:
        raise RuntimeError('没有训练配置')
    existing = matching_processes(TRAIN_SCRIPT)
    if existing:
        raise RuntimeError(f'发现已存在训练进程 {existing}，未重复启动；需接续观察已有进程')
    initial_step = 0
    latest = settings.output_dir / 'latest.pth'
    if latest.exists():
        import torch
        checkpoint = torch.load(latest, weights_only=True, map_location='cpu')
        initial_step = int(checkpoint['batches_completed'])
        del checkpoint
    environment = os.environ.copy()
    environment['PYTHONUNBUFFERED'] = '1'
    with (WORK / 'train.log').open('ab', buffering=0) as output:
        process = subprocess.Popen(
            [PYTHON, TRAIN_SCRIPT, '--config', str(CONFIG)], cwd=REPOSITORY,
            env=environment, stdin=subprocess.DEVNULL, stdout=output,
            stderr=subprocess.STDOUT, start_new_session=True,
        )
    (WORK / 'train.pid').write_text(str(process.pid) + '\n')
    report('training_started', training_pid=process.pid, verification=verification)
    observe_training(process, settings, verification, initial_step, True)


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        report('failed', error=str(error), traceback=traceback.format_exc())
        raise
