"""扫描并隔离不可用于训练的 mjai 牌谱。"""

import argparse
from concurrent.futures import (
    FIRST_COMPLETED,
    ProcessPoolExecutor,
    ThreadPoolExecutor,
    wait,
)
from dataclasses import dataclass
from functools import partial
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
import time
import zlib


MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.files import atomic_write_json, iter_dataset_files
from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.quarantine_logs.log_validation import (
    InvalidLogData,
    validate_protocol_text,
    validate_tile_and_score_rules,
)


@dataclass(frozen=True)
class ValidationResult:
    reason: str | None
    gzip_seconds: float
    python_seconds: float
    libriichi_seconds: float


@dataclass(frozen=True)
class QuarantineSummary:
    checked: int
    quarantined: int


@dataclass
class _RunStats:
    checked: int = 0
    quarantined: int = 0
    gzip_seconds: float = 0.0
    python_seconds: float = 0.0
    libriichi_seconds: float = 0.0


_WORKER_VALIDATOR = None


def _read_raw_log(filename):
    try:
        with gzip.open(filename, 'rt', encoding='utf-8') as f:
            return f.read()
    except (gzip.BadGzipFile, EOFError, UnicodeError, zlib.error) as exc:
        raise InvalidLogData(f'gzip 或 UTF-8 无法读取: {exc}') from exc


def _validate_file(filename, sample_validator):
    gzip_seconds = 0.0
    python_seconds = 0.0
    libriichi_seconds = 0.0
    try:
        started_at = time.perf_counter()
        try:
            raw_log = _read_raw_log(filename)
        finally:
            gzip_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        try:
            events = validate_protocol_text(raw_log)
            validate_tile_and_score_rules(events)
        finally:
            python_seconds = time.perf_counter() - started_at

        started_at = time.perf_counter()
        try:
            sample_validator.validate_raw_log(raw_log)
        finally:
            libriichi_seconds = time.perf_counter() - started_at
    except InvalidLogData as exc:
        return ValidationResult(
            reason=str(exc),
            gzip_seconds=gzip_seconds,
            python_seconds=python_seconds,
            libriichi_seconds=libriichi_seconds,
        )
    return ValidationResult(
        reason=None,
        gzip_seconds=gzip_seconds,
        python_seconds=python_seconds,
        libriichi_seconds=libriichi_seconds,
    )


def _initialize_worker(rayon_threads):
    """在导入 libriichi 前设置线程数，并为进程创建可复用校验器。"""
    global _WORKER_VALIDATOR
    os.environ['RAYON_NUM_THREADS'] = str(rayon_threads)
    from precompute_pipeline.quarantine_logs.sample_applicability import (
        SampleApplicabilityValidator,
    )

    _WORKER_VALIDATOR = SampleApplicabilityValidator(max_seq_len=60)


def _validate_file_worker(filename):
    if _WORKER_VALIDATOR is None:
        raise RuntimeError('筛选 worker 尚未初始化')
    return _validate_file(filename, _WORKER_VALIDATOR)


def _create_executor(settings):
    return ProcessPoolExecutor(
        max_workers=settings.worker_processes,
        mp_context=multiprocessing.get_context('spawn'),
        initializer=_initialize_worker,
        initargs=(settings.rayon_threads_per_worker,),
    )


def _append_report(filename, record):
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def _quarantine_file(source, destination):
    if destination.exists():
        raise FileExistsError(f'隔离目标已存在，拒绝覆盖: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)


def _load_progress(filename):
    if not filename.exists():
        return None
    with open(filename, encoding='utf-8') as f:
        progress = json.load(f)
    if progress.get('schema_version') != 1:
        raise ValueError(f'不支持的筛选进度 schema: {progress.get("schema_version")}')
    if progress.get('status') not in {'incomplete', 'complete'}:
        raise ValueError(f'筛选进度状态无效: {progress.get("status")}')
    return progress


def _save_progress(settings, *, status, split, last_relative_path):
    atomic_write_json(settings.progress_file, {
        'schema_version': 1,
        'status': status,
        'split': split,
        'last_relative_path': last_relative_path,
    })


def _format_eta(seconds):
    if seconds is None:
        return 'unknown'
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'


def _display_position(settings, split, root, source_files, cursor, rescan):
    if rescan:
        return len(source_files), 0
    names = {
        source.relative_to(root).as_posix()
        for source in source_files
    }
    quarantine_root = settings.quarantine_dir / split
    names.update(
        source.relative_to(quarantine_root).as_posix()
        for source in iter_dataset_files(quarantine_root)
    )
    processed = 0 if cursor is None else sum(name <= cursor for name in names)
    return len(names), processed


def _print_progress(
    *,
    output,
    split,
    processed,
    total,
    quarantined,
    cursor,
    stats,
    run_started_at,
    clock,
):
    elapsed = max(clock() - run_started_at, 1e-9)
    rate = stats.checked / elapsed
    remaining = max(total - processed, 0)
    eta = remaining / rate if rate > 0 else None
    percentage = 100.0 if total == 0 else processed * 100.0 / total
    divisor = max(stats.checked, 1)
    output(
        '[quarantine] '
        f'split={split} '
        f'processed={processed}/{total} ({percentage:.2f}%) '
        f'quarantined={quarantined} '
        f'rate={rate:.2f} files/s '
        f'eta={_format_eta(eta)} '
        f'cursor={cursor or "-"} '
        'avg_ms['
        f'gzip={stats.gzip_seconds * 1000.0 / divisor:.2f},'
        f'python={stats.python_seconds * 1000.0 / divisor:.2f},'
        f'libriichi={stats.libriichi_seconds * 1000.0 / divisor:.2f}'
        ']'
    )


def _commit_result(settings, split, source, relative, result, stats):
    if result.reason is not None:
        destination = settings.quarantine_dir / split / relative
        _quarantine_file(source, destination)
        _append_report(settings.report_file, {
            'source': f'{split}/{relative.as_posix()}',
            'target': destination.relative_to(settings.quarantine_dir).as_posix(),
            'reason': result.reason,
        })
        stats.quarantined += 1
    stats.checked += 1
    stats.gzip_seconds += result.gzip_seconds
    stats.python_seconds += result.python_seconds
    stats.libriichi_seconds += result.libriichi_seconds


def _process_split(
    settings,
    *,
    executor,
    validate_file,
    split,
    root,
    cursor,
    rescan,
    stats,
    output,
    clock,
    run_started_at,
):
    """有界收集乱序结果，且只让连续完成的文件前缀产生提交副作用。"""
    source_files = iter_dataset_files(root)
    total, initial_processed = _display_position(
        settings, split, root, source_files, cursor, rescan,
    )
    remaining = [
        source
        for source in source_files
        if cursor is None or source.relative_to(root).as_posix() > cursor
    ]
    future_to_item = {}
    completed = {}
    next_submit = 0
    next_commit = 0
    since_commit = 0
    last_relative_name = cursor
    last_reported_checked = stats.checked

    def fill_window():
        """运行中任务与等待前序的完成结果共同占用在途窗口。"""
        nonlocal next_submit
        while (
            next_submit < len(remaining)
            and len(future_to_item) + len(completed) < settings.max_in_flight
        ):
            source = remaining[next_submit]
            relative = source.relative_to(root)
            future = executor.submit(validate_file, source)
            future_to_item[future] = (next_submit, source, relative)
            next_submit += 1

    fill_window()
    try:
        while future_to_item:
            done, _ = wait(tuple(future_to_item), return_when=FIRST_COMPLETED)
            for future in done:
                index, source, relative = future_to_item.pop(future)
                completed[index] = (source, relative, future.result())

            while next_commit in completed:
                source, relative, result = completed.pop(next_commit)
                _commit_result(
                    settings, split, source, relative, result, stats,
                )
                next_commit += 1
                since_commit += 1
                last_relative_name = relative.as_posix()
                if since_commit >= settings.commit_every:
                    _save_progress(
                        settings,
                        status='incomplete',
                        split=split,
                        last_relative_path=last_relative_name,
                    )
                    _print_progress(
                        output=output,
                        split=split,
                        processed=initial_processed + next_commit,
                        total=total,
                        quarantined=stats.quarantined,
                        cursor=last_relative_name,
                        stats=stats,
                        run_started_at=run_started_at,
                        clock=clock,
                    )
                    last_reported_checked = stats.checked
                    since_commit = 0
            fill_window()
    except BaseException:
        for future in future_to_item:
            future.cancel()
        raise

    if since_commit:
        _save_progress(
            settings,
            status='incomplete',
            split=split,
            last_relative_path=last_relative_name,
        )
    if stats.checked != last_reported_checked or not remaining:
        _print_progress(
            output=output,
            split=split,
            processed=initial_processed + next_commit,
            total=total,
            quarantined=stats.quarantined,
            cursor=last_relative_name,
            stats=stats,
            run_started_at=run_started_at,
            clock=clock,
        )


def _shutdown_executor(executor, *, cancel):
    try:
        executor.shutdown(wait=True, cancel_futures=cancel)
    except TypeError:
        executor.shutdown(wait=True)


def run_quarantine(
    settings,
    *,
    sample_validator=None,
    rescan=False,
    executor_factory=None,
    validate_file=None,
    output=print,
    clock=time.monotonic,
):
    """并行校验 train/val，并由主进程按文件顺序提交隔离副作用。"""
    progress = None if rescan else _load_progress(settings.progress_file)
    if progress is not None and progress['status'] == 'complete':
        return QuarantineSummary(checked=0, quarantined=0)
    progress_split = progress.get('split') if progress is not None else 'train'
    cursor = progress.get('last_relative_path') if progress is not None else None
    split_order = ('train', 'val')
    if progress_split not in split_order:
        raise ValueError(f'筛选进度 split 无效: {progress_split}')

    if executor_factory is None and sample_validator is None:
        executor = _create_executor(settings)
    elif executor_factory is None:
        executor = ThreadPoolExecutor(max_workers=settings.worker_processes)
    else:
        executor = executor_factory(settings)
    if validate_file is not None:
        task = validate_file
    elif sample_validator is not None:
        task = partial(_validate_file, sample_validator=sample_validator)
    else:
        task = _validate_file_worker
    stats = _RunStats()
    run_started_at = clock()
    try:
        for split, root in (('train', settings.train_dir), ('val', settings.val_dir)):
            if split_order.index(split) < split_order.index(progress_split):
                continue
            _process_split(
                settings,
                executor=executor,
                validate_file=task,
                split=split,
                root=root,
                cursor=cursor if split == progress_split else None,
                rescan=rescan,
                stats=stats,
                output=output,
                clock=clock,
                run_started_at=run_started_at,
            )
            if split == 'train':
                _save_progress(
                    settings,
                    status='incomplete',
                    split='val',
                    last_relative_path=None,
                )
                progress_split = 'val'
                cursor = None
    except BaseException:
        _shutdown_executor(executor, cancel=True)
        raise
    _shutdown_executor(executor, cancel=False)
    _save_progress(
        settings,
        status='complete',
        split='val',
        last_relative_path=None,
    )
    return QuarantineSummary(
        checked=stats.checked,
        quarantined=stats.quarantined,
    )


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='筛选并隔离不可用的 mjai 牌谱')
    parser.add_argument('--config', type=Path, required=True, help='新预计算链路 TOML 配置')
    parser.add_argument('--rescan', action='store_true', help='忽略完成状态并重新扫描源目录')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    settings = load_settings(args.config)
    summary = run_quarantine(settings.quarantine, rescan=args.rescan)
    print(json.dumps({
        'checked': summary.checked,
        'quarantined': summary.quarantined,
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
