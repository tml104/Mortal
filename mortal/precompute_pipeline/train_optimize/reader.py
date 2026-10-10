"""按旧样本排列契约独立读取 chunks，并按样本游标重新拼 batch。"""

from concurrent.futures import ThreadPoolExecutor
import hashlib
import random

import numpy as np
import torch

from precompute_pipeline.common.codec import decode_chunk_torch, decode_packed_chunk
from precompute_pipeline.common.dataset import iter_chunk_records


def _derived_seed(*parts):
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode('utf-8'))
        digest.update(b'\0')
    return int.from_bytes(digest.digest(), 'little')


def _read_chunk(record):
    packed = decode_packed_chunk(record.path.read_bytes())
    if packed.sample_count != record.sample_count:
        raise ValueError(f'chunk 样本数量与完成记录不一致: {record.path}')
    return packed


class DatasetBatchReader:
    """只保留当前块和一块预取；batch 大小不影响逻辑乱序。"""

    def __init__(self, progress_file, *, split, batch_size, device, seed, epoch, prefetch=True):
        if type(batch_size) is not int or batch_size <= 0:
            raise ValueError('batch_size 必须是正整数')
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed, self.epoch, self.split = int(seed), int(epoch), split
        self.prefetch = prefetch
        self.records = list(iter_chunk_records(progress_file, split))
        random.Random(_derived_seed(self.seed, self.epoch, split, 'chunks')).shuffle(self.records)
        self.total_samples = sum(record.sample_count for record in self.records)

    def __iter__(self):
        return self.iter_from(0)

    def iter_from(self, consumed_samples):
        """直接跳过完整旧块，从当前块的乱序 offset 恢复。"""
        if type(consumed_samples) is not int or not 0 <= consumed_samples <= self.total_samples:
            raise ValueError('样本游标超出数据集范围')
        offset = consumed_samples
        for index, record in enumerate(self.records):
            if offset < record.sample_count:
                return self._iterate(index, offset)
            offset -= record.sample_count
        return iter(())

    def _iterate(self, start_index, offset):
        executor = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        carry = None
        future = executor.submit(_read_chunk, self.records[start_index]) if executor else None
        try:
            for index in range(start_index, len(self.records)):
                record = self.records[index]
                packed = future.result() if executor else _read_chunk(record)
                if executor and index + 1 < len(self.records):
                    future = executor.submit(_read_chunk, self.records[index + 1])
                order = np.random.default_rng(_derived_seed(
                    self.seed, self.epoch, self.split, record.chunk_id, 'rows',
                )).permutation(record.sample_count)
                if index == start_index:
                    order = order[offset:]
                position = 0
                while position < len(order):
                    needed = self.batch_size - (0 if carry is None else len(carry[0]))
                    count = min(needed, len(order) - position)
                    decoded = decode_chunk_torch(packed, self.device, indices=order[position:position + count])
                    carry = decoded if carry is None else tuple(
                        torch.cat((left, right), dim=0) for left, right in zip(carry, decoded, strict=True)
                    )
                    position += count
                    if len(carry[0]) == self.batch_size:
                        yield carry
                        carry = None
            if carry is not None:
                yield carry
        finally:
            if executor:
                executor.shutdown(wait=True, cancel_futures=True)
