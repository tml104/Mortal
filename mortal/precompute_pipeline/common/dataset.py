"""预计算生产端与训练消费端共享的数据集描述。"""

from dataclasses import dataclass
import json
from pathlib import Path


DATASET_SCHEMA_VERSION = 1
CHUNK_FILENAME = 'chunk_{chunk_id:08d}.mpc.zst'


@dataclass(frozen=True)
class ChunkRecord:
    split: str
    chunk_id: int
    volume: str
    sample_count: int
    path: Path


def load_dataset_state(progress_file, *, require_complete=True):
    progress_file = Path(progress_file).resolve()
    with open(progress_file, encoding='utf-8') as f:
        state = json.load(f)
    if state.get('schema_version') != DATASET_SCHEMA_VERSION:
        raise ValueError(f'不支持的数据集 schema: {state.get("schema_version")}')
    if state.get('status') not in {'incomplete', 'complete'}:
        raise ValueError(f'数据集状态无效: {state.get("status")}')
    if require_complete and state['status'] != 'complete':
        raise RuntimeError('预计算数据集尚未完成')
    return state


def iter_chunk_records(progress_file, split):
    """仅根据已提交区间构造稳定 chunk 列表。"""
    progress_file = Path(progress_file).resolve()
    state = load_dataset_state(progress_file)
    if split not in {'train', 'val'}:
        raise ValueError(f'未知 split: {split}')
    split_state = state['splits'][split]
    chunk_count = split_state['chunk_count']
    chunk_size = state['chunk_size']
    tail = split_state['tail_sample_count']
    records = []
    expected_id = 0
    for item in split_state['ranges']:
        start = item['start']
        end = item['end']
        volume = item['volume']
        if start != expected_id or not start < end or volume not in {'c', 'd'}:
            raise ValueError(f'{split} 的 chunk 分配区间不连续或无效: {item}')
        for chunk_id in range(start, end):
            sample_count = (
                tail
                if tail > 0 and chunk_id == chunk_count - 1
                else chunk_size
            )
            records.append(ChunkRecord(
                split=split,
                chunk_id=chunk_id,
                volume=volume,
                sample_count=sample_count,
                path=(
                    progress_file.parent
                    / split
                    / volume
                    / CHUNK_FILENAME.format(chunk_id=chunk_id)
                ),
            ))
        expected_id = end
    if expected_id != chunk_count or len(records) != chunk_count:
        raise ValueError(f'{split} 的 chunk 区间未覆盖全部已提交 ID')
    return records
