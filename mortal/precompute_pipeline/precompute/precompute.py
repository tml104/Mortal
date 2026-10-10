"""按文件顺序生成固定样本数的跨盘无损 chunk。"""

import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch


MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from precompute_pipeline.common.codec import (
    ACTION_SPACE,
    OBS_PLANES,
    OBS_TILES,
    SampleBatch,
    encode_chunk,
)
from precompute_pipeline.common.files import iter_dataset_files
from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.precompute.progress import PrecomputeProgress
from precompute_pipeline.precompute.storage import (
    DatasetStorage,
    initialize_new_dataset,
    prepare_resume,
)


class SampleBuffer:
    """预分配定长数组，避免满块时复制整份 dense observation。"""

    def __init__(self, capacity):
        self.capacity = capacity
        self.size = 0
        self.obs = np.empty((capacity, OBS_PLANES, OBS_TILES), dtype=np.float32)
        self.actions = np.empty(capacity, dtype=np.uint8)
        self.masks = np.empty((capacity, ACTION_SPACE), dtype=np.bool_)
        self.steps = np.empty(capacity, dtype=np.uint16)
        self.rewards = np.empty(capacity, dtype=np.float64)
        self.ranks = np.empty(capacity, dtype=np.uint8)

    @property
    def remaining(self):
        return self.capacity - self.size

    def append(self, batch, start, count):
        target = slice(self.size, self.size + count)
        source = slice(start, start + count)
        self.obs[target] = batch.obs[source]
        self.actions[target] = batch.actions[source]
        self.masks[target] = batch.masks[source]
        self.steps[target] = batch.steps_to_done[source]
        self.rewards[target] = batch.rewards[source]
        self.ranks[target] = batch.ranks[source]
        self.size += count

    def batch(self):
        view = slice(0, self.size)
        return SampleBatch(
            obs=self.obs[view],
            actions=self.actions[view],
            masks=self.masks[view],
            steps_to_done=self.steps[view],
            rewards=self.rewards[view],
            ranks=self.ranks[view],
        )

    def clear(self):
        self.size = 0


def _build_generator(settings):
    os.environ['RAYON_NUM_THREADS'] = str(settings.rayon_threads)
    torch.set_num_threads(settings.torch_threads)
    from precompute_pipeline.precompute.samples import SampleGenerator
    reward = settings.reward
    return SampleGenerator(
        state_file=reward.state_file,
        d_model=reward.d_model,
        nhead=reward.nhead,
        num_layers=reward.num_layers,
        max_seq_len=reward.max_seq_len,
        pts=reward.pts,
    )


def _generate_split(settings, state, storage, generator, split, root, cursor=None, *, progress):
    """顺序生成一个 split，日志始终跟在持久化提交之后。"""
    progress.event('scanning', state, split)
    files = iter_dataset_files(root)
    buffer = SampleBuffer(settings.chunk_size)
    cursor = cursor or {'relative_path': None, 'sample_offset': 0}
    if cursor.get('eof'):
        progress.begin_split(split, len(files), len(files))
        with progress.timings.measure('commit'):
            storage.finish_split(split, len(files))
        progress.finished_split(state, split)
        return
    relative_names = [filename.relative_to(root).as_posix() for filename in files]
    cursor_name = cursor.get('relative_path')
    start_index = 0
    if cursor_name is not None:
        try:
            start_index = relative_names.index(cursor_name)
        except ValueError as exc:
            raise RuntimeError(f'恢复游标对应的源牌谱不存在: {cursor_name}') from exc
    progress.begin_split(split, len(files), start_index)
    first_offset = int(cursor.get('sample_offset', 0))
    if first_offset < 0:
        raise ValueError('恢复样本 offset 不能为负数')

    for batch_start in range(start_index, len(files), settings.file_batch_size):
        selected = files[batch_start:batch_start + settings.file_batch_size]
        with progress.timings.generation():
            generated = generator.load_files(selected)
        if len(generated) != len(selected):
            raise RuntimeError('样本生成器返回的文件数与输入不一致')
        for file_index, (filename, samples) in enumerate(
            zip(selected, generated, strict=True), start=batch_start,
        ):
            relative_name = filename.relative_to(root).as_posix()
            sample_offset = first_offset if relative_name == cursor_name else 0
            if sample_offset > len(samples.actions):
                raise RuntimeError(
                    f'恢复样本 offset 超出牌谱样本数: '
                    f'{relative_name} {sample_offset}>{len(samples.actions)}'
                )
            if relative_name == cursor_name:
                progress.resolve_resume_file(
                    split, complete=sample_offset == len(samples.actions),
                )
            while sample_offset < len(samples.actions):
                count = min(buffer.remaining, len(samples.actions) - sample_offset)
                with progress.timings.measure('python'):
                    buffer.append(samples, sample_offset, count)
                sample_offset += count
                if buffer.size == buffer.capacity:
                    with progress.timings.measure('encode'):
                        encoded = encode_chunk(
                            buffer.batch(),
                            compression_level=settings.compression_level,
                        )
                    with progress.timings.measure('commit'):
                        storage.commit_chunk(
                            split,
                            encoded,
                            buffer.size,
                            {
                                'relative_path': relative_name,
                                'sample_offset': sample_offset,
                            },
                        )
                    completed_files = file_index + int(sample_offset == len(samples.actions))
                    progress.committed(state, split, completed_files)
                    buffer.clear()
    if buffer.size:
        with progress.timings.measure('encode'):
            encoded = encode_chunk(
                buffer.batch(),
                compression_level=settings.compression_level,
            )
        with progress.timings.measure('commit'):
            storage.commit_chunk(
                split,
                encoded,
                buffer.size,
                {'relative_path': None, 'sample_offset': 0, 'eof': True},
                is_tail=True,
            )
        progress.committed(state, split, len(files))
    with progress.timings.measure('commit'):
        storage.finish_split(split, len(files))
    progress.finished_split(state, split)


def run_precompute(settings, *, generator=None, clock=time.monotonic):
    """从头生成或从最后已提交 chunk 恢复 train/val 数据集。"""
    progress = PrecomputeProgress(settings, clock=clock)
    if settings.progress_file.exists():
        from precompute_pipeline.common.dataset import load_dataset_state
        state = load_dataset_state(settings.progress_file, require_complete=False)
        if state['status'] == 'complete':
            progress.event('already_complete', state)
            return state
        progress.event('resume', state)
        prepare_resume(settings, state)
    else:
        progress.event('start')
        state = initialize_new_dataset(settings)
        from precompute_pipeline.common.files import atomic_write_json
        with progress.timings.measure('commit'):
            atomic_write_json(settings.progress_file, state)
    storage = DatasetStorage(settings, state)
    if generator is None:
        generator = _build_generator(settings)
    # 注入本次计时器，恢复或重复调用时不沿用生成器旧的累计值。
    generator.timings = progress.timings
    if state['current_split'] == 'train':
        _generate_split(
            settings,
            state,
            storage,
            generator,
            'train',
            settings.train_dir,
            cursor=state['source_cursor'],
            progress=progress,
        )
    if state['status'] != 'complete' and state['current_split'] == 'val':
        _generate_split(
            settings,
            state,
            storage,
            generator,
            'val',
            settings.val_dir,
            cursor=state['source_cursor'],
            progress=progress,
        )
    progress.event('complete', state)
    return state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='生成 Mortal 无损预计算 chunks')
    parser.add_argument('--config', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pipeline_settings = load_settings(args.config)
    if pipeline_settings.precompute is None:
        raise ValueError('配置缺少 [precompute] 和 [reward]')
    state = run_precompute(pipeline_settings.precompute)
    print(json.dumps({
        'status': state['status'],
        'generation_id': state['generation_id'],
        'train_samples': state['splits']['train']['sample_count'],
        'val_samples': state['splits']['val']['sample_count'],
    }, ensure_ascii=False))


if __name__ == '__main__':
    main()
