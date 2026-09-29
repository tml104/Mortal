"""预计算私有的跨盘路由、chunk 发布与进度提交。"""

from pathlib import Path
import uuid

from precompute_pipeline.common.dataset import CHUNK_FILENAME, DATASET_SCHEMA_VERSION
from precompute_pipeline.common.files import atomic_write_bytes, atomic_write_json


def _clean_managed_chunks(directory):
    directory.mkdir(parents=True, exist_ok=True)
    for pattern in ('chunk_*.mpc.zst', 'chunk_*.mpc.zst.tmp-*'):
        for filename in directory.glob(pattern):
            filename.unlink()


def _ensure_link(link, target):
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() != target.resolve():
            raise RuntimeError(f'数据入口链接指向其他位置: {link}')
        return
    if link.exists():
        raise RuntimeError(f'数据入口已存在且不是链接: {link}')
    link.symlink_to(target.resolve(), target_is_directory=True)


def _ensure_dataset_links(settings):
    locations = {
        ('train', 'c'): settings.c_root / 'train',
        ('train', 'd'): settings.d_root / 'train',
        ('val', 'c'): settings.c_root / 'val',
    }
    for (split, volume), target in locations.items():
        _ensure_link(settings.logical_root / split / volume, target)


def _generation_contract(settings):
    reward = settings.reward
    return {
        'train_dir': str(settings.train_dir.resolve()),
        'val_dir': str(settings.val_dir.resolve()),
        'logical_root': str(settings.logical_root.resolve()),
        'c_root': str(settings.c_root.resolve()),
        'd_root': str(settings.d_root.resolve()),
        'chunk_size': settings.chunk_size,
        'compression_level': settings.compression_level,
        'observation_version': 4,
        'reward': {
            'state_file': str(reward.state_file.resolve()),
            'd_model': reward.d_model,
            'nhead': reward.nhead,
            'num_layers': reward.num_layers,
            'max_seq_len': reward.max_seq_len,
            'pts': list(reward.pts),
        },
    }


def initialize_new_dataset(settings):
    """只清理本链路命名的旧 chunk，并建立统一逻辑入口。"""
    directories = {
        settings.c_root / 'train',
        settings.d_root / 'train',
        settings.c_root / 'val',
    }
    for directory in directories:
        _clean_managed_chunks(directory)
    _ensure_dataset_links(settings)
    return {
        'schema_version': DATASET_SCHEMA_VERSION,
        'status': 'incomplete',
        'generation_id': uuid.uuid4().hex,
        'chunk_size': settings.chunk_size,
        'compression_level': settings.compression_level,
        'contract': _generation_contract(settings),
        'current_split': 'train',
        'source_cursor': {'relative_path': None, 'sample_offset': 0},
        'volume_bytes': {'c': 0, 'd': 0},
        'splits': {
            split: {
                'complete': False,
                'source_file_count': 0,
                'chunk_count': 0,
                'sample_count': 0,
                'tail_sample_count': 0,
                'ranges': [],
            }
            for split in ('train', 'val')
        },
    }


def prepare_resume(settings, state):
    """核对不可变输入，并清理 JSON 未提交的下一个 chunk ID。"""
    if state.get('contract') != _generation_contract(settings):
        raise RuntimeError('预计算输入或生成语义已变化，拒绝恢复')
    if state['volume_bytes']['c'] > settings.c_budget_bytes:
        raise RuntimeError('C 盘预算低于已经提交的用量')
    if state['volume_bytes']['d'] > settings.d_budget_bytes:
        raise RuntimeError('D 盘预算低于已经提交的用量')
    _ensure_dataset_links(settings)
    split = state['current_split']
    chunk_id = state['splits'][split]['chunk_count']
    filename = CHUNK_FILENAME.format(chunk_id=chunk_id)
    for root in (settings.c_root, settings.d_root):
        candidate = root / split / filename
        candidate.unlink(missing_ok=True)
        for temporary in candidate.parent.glob(f'{filename}.tmp-*'):
            temporary.unlink()


class DatasetStorage:
    def __init__(self, settings, state):
        self.settings = settings
        self.state = state

    def _choose_volume(self, split, size):
        used = self.state['volume_bytes']
        if split == 'train' and used['d'] + size <= self.settings.d_budget_bytes:
            return 'd'
        if used['c'] + size <= self.settings.c_budget_bytes:
            return 'c'
        raise RuntimeError(
            f'跨盘缓存预算不足: split={split}, chunk_bytes={size}, '
            f'c={used["c"]}/{self.settings.c_budget_bytes}, '
            f'd={used["d"]}/{self.settings.d_budget_bytes}'
        )

    def _physical_path(self, split, volume, chunk_id):
        root = self.settings.c_root if volume == 'c' else self.settings.d_root
        return root / split / CHUNK_FILENAME.format(chunk_id=chunk_id)

    def _append_range(self, split, chunk_id, volume):
        ranges = self.state['splits'][split]['ranges']
        if ranges and ranges[-1]['volume'] == volume and ranges[-1]['end'] == chunk_id:
            ranges[-1]['end'] += 1
        else:
            ranges.append({'start': chunk_id, 'end': chunk_id + 1, 'volume': volume})

    def commit_chunk(self, split, encoded, sample_count, source_cursor, *, is_tail=False):
        split_state = self.state['splits'][split]
        chunk_id = split_state['chunk_count']
        volume = self._choose_volume(split, len(encoded))
        filename = self._physical_path(split, volume, chunk_id)
        atomic_write_bytes(filename, encoded)
        self._append_range(split, chunk_id, volume)
        split_state['chunk_count'] += 1
        split_state['sample_count'] += sample_count
        if is_tail:
            split_state['tail_sample_count'] = sample_count
        self.state['volume_bytes'][volume] += len(encoded)
        self.state['current_split'] = split
        self.state['source_cursor'] = source_cursor
        atomic_write_json(self.settings.progress_file, self.state)
        return filename

    def finish_split(self, split, source_file_count):
        split_state = self.state['splits'][split]
        split_state['complete'] = True
        split_state['source_file_count'] = source_file_count
        if split == 'train':
            self.state['current_split'] = 'val'
            self.state['source_cursor'] = {'relative_path': None, 'sample_offset': 0}
        else:
            self.state['status'] = 'complete'
            self.state['source_cursor'] = {'relative_path': None, 'sample_offset': 0}
        atomic_write_json(self.settings.progress_file, self.state)
