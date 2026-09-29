"""确定性读取预计算 chunks，并跨块形成设备 batch。"""

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
        raise ValueError(
            f'chunk 样本数与完成记录不一致: '
            f'{record.path} expected={record.sample_count}, actual={packed.sample_count}'
        )
    return packed


class DatasetBatchReader:
    """对完整 split 执行两级打乱、单块预取和设备端按批解码。"""

    def __init__(
        self,
        progress_file,
        *,
        split,
        batch_size,
        device,
        seed,
        epoch,
        prefetch=True,
    ):
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size 必须是正整数')
        self.progress_file = progress_file
        self.split = split
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.seed = int(seed)
        self.epoch = int(epoch)
        self.prefetch = bool(prefetch)
        self.records = list(iter_chunk_records(progress_file, split))
        rng = random.Random(_derived_seed(self.seed, self.epoch, split, 'chunks'))
        rng.shuffle(self.records)
        self.total_samples = sum(record.sample_count for record in self.records)

    def __iter__(self):
        return self.iter_from(0)

    def _locate(self, consumed_samples):
        if not isinstance(consumed_samples, int) or not 0 <= consumed_samples <= self.total_samples:
            raise ValueError(
                f'consumed_samples 必须位于 0..{self.total_samples}，实际为 {consumed_samples}'
            )
        remaining = consumed_samples
        for record_index, record in enumerate(self.records):
            if remaining < record.sample_count:
                return record_index, remaining
            remaining -= record.sample_count
        return len(self.records), 0

    def _row_order(self, record):
        rng = np.random.default_rng(_derived_seed(
            self.seed,
            self.epoch,
            self.split,
            record.chunk_id,
            'rows',
        ))
        return rng.permutation(record.sample_count)

    def iter_from(self, consumed_samples):
        """从逻辑样本位置直接定位，并重新组成后续 batch。"""
        start_index, first_offset = self._locate(consumed_samples)
        if start_index == len(self.records):
            return iter(())
        return self._iterate(start_index, first_offset)

    def _iterate(self, start_index, first_offset):
        executor = ThreadPoolExecutor(max_workers=1) if self.prefetch else None
        future = None
        carry = None
        try:
            if executor is not None:
                future = executor.submit(_read_chunk, self.records[start_index])
            for record_index in range(start_index, len(self.records)):
                record = self.records[record_index]
                if executor is not None:
                    packed = future.result()
                    if record_index + 1 < len(self.records):
                        future = executor.submit(_read_chunk, self.records[record_index + 1])
                else:
                    packed = _read_chunk(record)
                order = self._row_order(record)
                if record_index == start_index and first_offset:
                    order = order[first_offset:]
                position = 0

                if carry is not None:
                    needed = self.batch_size - len(carry[0])
                    count = min(needed, len(order))
                    decoded = decode_chunk_torch(
                        packed,
                        self.device,
                        indices=order[:count],
                    )
                    carry = tuple(
                        torch.cat((left, right), dim=0)
                        for left, right in zip(carry, decoded, strict=True)
                    )
                    position = count
                    if len(carry[0]) == self.batch_size:
                        yield carry
                        carry = None

                while position + self.batch_size <= len(order):
                    yield decode_chunk_torch(
                        packed,
                        self.device,
                        indices=order[position:position + self.batch_size],
                    )
                    position += self.batch_size
                if position < len(order):
                    remainder = decode_chunk_torch(
                        packed,
                        self.device,
                        indices=order[position:],
                    )
                    if carry is None:
                        carry = remainder
                    else:
                        carry = tuple(
                            torch.cat((left, right), dim=0)
                            for left, right in zip(carry, remainder, strict=True)
                        )
            if carry is not None:
                yield carry
        finally:
            if executor is not None:
                executor.shutdown(wait=True, cancel_futures=True)
