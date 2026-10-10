"""预计算的终端反馈；所有统计均为本次运行的内存状态。"""

from contextlib import contextmanager
import json
import time


class StageTimings:
    """按本次调用累计五段时间，Python 整理排除嵌套的 Rust/GRP 开销。"""

    def __init__(self, clock=time.monotonic):
        self.clock = clock
        self.seconds = dict.fromkeys(('rust', 'grp', 'python', 'encode', 'commit'), 0.0)

    @contextmanager
    def measure(self, stage):
        """即使调用失败也记录已执行时间，成功与否仍由上层控制输出。"""
        started = self.clock()
        try:
            yield
        finally:
            self.seconds[stage] += self.clock() - started

    @contextmanager
    def generation(self):
        """一批生成总时间扣除 Rust 和 GRP 的增量，避免重复计时。"""
        started = self.clock()
        backend_before = self.seconds['rust'] + self.seconds['grp']
        try:
            yield
        finally:
            backend = self.seconds['rust'] + self.seconds['grp'] - backend_before
            self.seconds['python'] += max(0.0, self.clock() - started - backend)


class PrecomputeProgress:
    """仅在调用者确认提交成功后展示对应的数据成果。"""

    def __init__(self, settings, *, clock=time.monotonic):
        self.settings = settings
        self.clock = clock
        self.started_at = clock()
        self.timings = StageTimings(clock)
        self.initial_samples = None
        self.since_log = {'train': 0, 'val': 0}
        self.file_totals = {}
        self.files = {}
        self.initial_files = {}

    def begin_split(self, split, total, completed):
        """记录有序文件列表的计数，不把未提交 buffer 中的文件计入成果。"""
        self.file_totals[split] = total
        self.files[split] = completed
        self.initial_files[split] = completed

    def resolve_resume_file(self, split, *, complete):
        """重读游标牌谱后，校正旧游标恰好位于谱末的基线。"""
        if complete:
            self.files[split] += 1
            self.initial_files[split] += 1

    def event(self, event, state=None, split=None):
        """块/样本属于标示的 split，run 速度只统计本次运行增量。"""
        elapsed = max(0.0, self.clock() - self.started_at)
        fields = [f'event={event}', f'elapsed_s={elapsed:.2f}']
        if state is not None:
            split = split or state['current_split']
            info = state['splits'][split]
            samples = sum(part['sample_count'] for part in state['splits'].values())
            if self.initial_samples is None:
                self.initial_samples = samples
            added_files = sum(self.files[key] - self.initial_files[key] for key in self.files)
            file_rate = added_files / elapsed if elapsed > 0 else 0.0
            sample_rate = (samples - self.initial_samples) / elapsed if elapsed > 0 else 0.0
            cursor = state['source_cursor']
            path = cursor['relative_path'] or '-'
            if cursor.get('eof') or info['complete']:
                path = '<end>'
            used = state['volume_bytes']
            fields.extend((
                f'split={split}',
                f'chunks={info["chunk_count"]}',
                f'samples={info["sample_count"]}',
                f'run_files_s={file_rate:.2f}',
                f'run_samples_s={sample_rate:.2f}',
                f'c_gib={used["c"] / 2 ** 30:.3f}/{self.settings.c_budget_bytes / 2 ** 30:.3f}',
                f'd_gib={used["d"] / 2 ** 30:.3f}/{self.settings.d_budget_bytes / 2 ** 30:.3f}',
                f'cursor={json.dumps(path, ensure_ascii=False)}',
                f'offset={cursor["sample_offset"]}',
            ))
            if split in self.files or info['complete']:
                total = self.file_totals.get(split, info['source_file_count'])
                completed = self.files.get(split, total)
                remaining = total - completed
                percent = 100.0 * completed / total if total else 100.0
                if remaining == 0:
                    eta = '0.0'
                elif file_rate > 0:
                    eta = f'{remaining / file_rate:.1f}'
                else:
                    eta = 'unknown'
                fields.extend((
                    f'files={completed}/{total}',
                    f'percent={percent:.2f}',
                    f'eta_s={eta}',
                ))
        elif split is not None:
            fields.append(f'split={split}')
        fields.extend(f'{stage}_s={seconds:.6f}' for stage, seconds in self.timings.seconds.items())
        print('[precompute] ' + ' '.join(fields), flush=True)

    def committed(self, state, split, completed_files):
        """成功保存之后才增加文件计数，并按本次已提交块触发日志。"""
        self.files[split] = completed_files
        self.since_log[split] += 1
        if self.since_log[split] >= self.settings.log_every_chunks:
            self.event('chunk_committed', state, split)
            self.since_log[split] = 0

    def finished_split(self, state, split):
        """结束摘要同时展示最后余量，以及已经确认结束的空样本文件。"""
        self.files[split] = self.file_totals[split]
        self.event('split_complete', state, split)
        self.since_log[split] = 0
