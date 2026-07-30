import hashlib
import io
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import zstandard as zstd


CACHE_SCHEMA_VERSION = 1
OBS_PLANES = 1012
OBS_TILES = 34
ACTION_SPACE = 46


@dataclass(frozen=True)
class EncodedChunk:
    """保存一个 plane-aware 缓存 chunk 的数组和解释元数据。"""

    metadata: dict
    arrays: dict[str, np.ndarray]


def sha256_file(filename):
    """流式计算缓存文件摘要。"""
    digest = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pack_bits(values):
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f'bit-pack 输入必须是二维数组，实际为 {values.shape}')
    return np.packbits(values, axis=1, bitorder='little')


def _validate_sample_arrays(obs, actions, masks, steps_to_done, rewards, ranks, source):
    sample_count = obs.shape[0]
    if obs.shape != (sample_count, OBS_PLANES, OBS_TILES):
        raise ValueError(f'observation shape 不兼容: {obs.shape}')
    expected_shapes = {
        'actions': (sample_count,),
        'masks': (sample_count, ACTION_SPACE),
        'steps_to_done': (sample_count,),
        'rewards': (sample_count,),
        'ranks': (sample_count,),
        'file_ids': (sample_count,),
        'player_ids': (sample_count,),
        'sample_indices': (sample_count,),
    }
    actual = {
        'actions': actions.shape,
        'masks': masks.shape,
        'steps_to_done': steps_to_done.shape,
        'rewards': rewards.shape,
        'ranks': ranks.shape,
        'file_ids': source['file_ids'].shape,
        'player_ids': source['player_ids'].shape,
        'sample_indices': source['sample_indices'].shape,
    }
    for name, expected in expected_shapes.items():
        if actual[name] != expected:
            raise ValueError(f'{name} shape 不兼容: expected={expected}, actual={actual[name]}')
    if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
        raise ValueError('observation 或 reward 含 NaN/Inf')
    if np.abs(obs).max(initial=0) > np.finfo(np.float16).max:
        raise ValueError('observation 超出 float16 可表示范围')
    if np.any(actions < 0) or np.any(actions >= ACTION_SPACE):
        raise ValueError('action 超出动作空间')
    rows = np.arange(sample_count)
    if not masks[rows, actions].all():
        raise ValueError('待编码样本包含 mask 不允许的 action')
    if np.any(steps_to_done < 0) or np.any(steps_to_done > np.iinfo(np.uint16).max):
        raise ValueError('steps_to_done 超出 uint16 范围')
    if np.any(ranks < 0) or np.any(ranks > 3):
        raise ValueError('rank 超出 0..3')
    if np.any(source['file_ids'] < 0) or np.any(source['file_ids'] > np.iinfo(np.uint32).max):
        raise ValueError('file_id 超出 uint32 范围')
    if np.any(source['player_ids'] < 0) or np.any(source['player_ids'] > 3):
        raise ValueError('player_id 超出 0..3')
    if np.any(source['sample_indices'] < 0) or np.any(
        source['sample_indices'] > np.iinfo(np.uint16).max
    ):
        raise ValueError('sample_idx 超出 uint16 范围')


def encode_chunk(obs, actions, masks, steps_to_done, rewards, ranks, source):
    """按平面值域编码 observation，并返回可压缩数组。"""
    obs = np.asarray(obs, dtype=np.float32, order='C')
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.bool_)
    steps_to_done = np.asarray(steps_to_done, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    source = {name: np.asarray(value, dtype=np.int64) for name, value in source.items()}
    _validate_sample_arrays(obs, actions, masks, steps_to_done, rewards, ranks, source)

    sample_count = obs.shape[0]
    all_zero = np.all(obs == 0, axis=(0, 2))
    all_one = np.all(obs == 1, axis=(0, 2))
    remaining = ~(all_zero | all_one)
    constant_per_sample = np.all(obs == obs[:, :, :1], axis=(0, 2))
    scalar_values = obs[:, :, 0]
    scalar_binary = np.all((scalar_values == 0) | (scalar_values == 1), axis=0)
    constant_binary = remaining & constant_per_sample & scalar_binary
    constant_float = remaining & constant_per_sample & ~scalar_binary
    remaining &= ~constant_per_sample
    binary = remaining & np.all((obs == 0) | (obs == 1), axis=(0, 2))
    dense = remaining & ~binary

    categories = {
        'zero_planes': np.flatnonzero(all_zero).astype(np.uint16),
        'one_planes': np.flatnonzero(all_one).astype(np.uint16),
        'constant_binary_planes': np.flatnonzero(constant_binary).astype(np.uint16),
        'constant_float_planes': np.flatnonzero(constant_float).astype(np.uint16),
        'binary_planes': np.flatnonzero(binary).astype(np.uint16),
        'dense_planes': np.flatnonzero(dense).astype(np.uint16),
    }
    partition = np.concatenate(tuple(categories.values()))
    if len(partition) != OBS_PLANES or len(np.unique(partition)) != OBS_PLANES:
        raise RuntimeError('observation plane 分类没有形成完整互斥分区')

    constant_binary_values = scalar_values[:, categories['constant_binary_planes']]
    binary_values = obs[:, categories['binary_planes'], :].reshape(sample_count, -1)
    constant_float_values = scalar_values[:, categories['constant_float_planes']]
    dense_values = obs[:, categories['dense_planes'], :]

    quantized_sources = (constant_float_values, dense_values)
    quantized_count = sum(value.size for value in quantized_sources)
    quantized_abs_sum = 0.0
    quantized_abs_max = 0.0
    for value in quantized_sources:
        if value.size == 0:
            continue
        error = np.abs(value - value.astype(np.float16).astype(np.float32))
        quantized_abs_sum += float(error.sum(dtype=np.float64))
        quantized_abs_max = max(quantized_abs_max, float(error.max(initial=0)))

    arrays = {
        **categories,
        'constant_binary_bits': _pack_bits(constant_binary_values),
        'constant_float_values': constant_float_values.astype(np.float16),
        'binary_bits': _pack_bits(binary_values),
        'dense_values': dense_values.astype(np.float16),
        'actions': actions.astype(np.uint8),
        'mask_bits': _pack_bits(masks),
        'steps_to_done': steps_to_done.astype(np.uint16),
        'rewards': rewards.astype(np.float64),
        'ranks': ranks.astype(np.uint8),
        'file_ids': source['file_ids'].astype(np.uint32),
        'player_ids': source['player_ids'].astype(np.uint8),
        'sample_indices': source['sample_indices'].astype(np.uint16),
    }
    packed_array_bytes = sum(value.nbytes for value in arrays.values())
    metadata = {
        'schema_version': CACHE_SCHEMA_VERSION,
        'sample_count': sample_count,
        'obs_shape': [OBS_PLANES, OBS_TILES],
        'continuous_dtype': 'float16',
        'reward_dtype': 'float64',
        'plane_counts': {name: len(value) for name, value in categories.items()},
        'dense_float32_bytes': int(obs.nbytes),
        'packed_array_bytes': packed_array_bytes,
        'quantized_value_count': quantized_count,
        'quantization_abs_sum': quantized_abs_sum,
        'quantization_abs_max': quantized_abs_max,
    }
    return EncodedChunk(metadata=metadata, arrays=arrays)


def _unpack_numpy(packed, count):
    if count == 0:
        return np.empty((packed.shape[0], 0), dtype=np.uint8)
    return np.unpackbits(packed, axis=1, count=count, bitorder='little')


def decode_chunk_numpy(chunk, indices=None):
    """在 CPU 解码选定样本，主要用于正确性验证。"""
    metadata, arrays = chunk.metadata, chunk.arrays
    total = metadata['sample_count']
    rows = np.arange(total) if indices is None else np.asarray(indices, dtype=np.int64)
    obs = np.zeros((len(rows), OBS_PLANES, OBS_TILES), dtype=np.float32)

    one_planes = arrays['one_planes'].astype(np.int64)
    obs[:, one_planes, :] = 1
    planes = arrays['constant_binary_planes'].astype(np.int64)
    values = _unpack_numpy(arrays['constant_binary_bits'][rows], len(planes))
    obs[:, planes, :] = values[:, :, None]
    planes = arrays['constant_float_planes'].astype(np.int64)
    obs[:, planes, :] = arrays['constant_float_values'][rows].astype(np.float32)[:, :, None]
    planes = arrays['binary_planes'].astype(np.int64)
    values = _unpack_numpy(arrays['binary_bits'][rows], len(planes) * OBS_TILES)
    obs[:, planes, :] = values.reshape(len(rows), len(planes), OBS_TILES)
    planes = arrays['dense_planes'].astype(np.int64)
    obs[:, planes, :] = arrays['dense_values'][rows].astype(np.float32)

    masks = _unpack_numpy(arrays['mask_bits'][rows], ACTION_SPACE).astype(np.bool_)
    return (
        obs,
        arrays['actions'][rows].astype(np.int64),
        masks,
        arrays['steps_to_done'][rows].astype(np.int64),
        arrays['rewards'][rows].astype(np.float64),
        arrays['ranks'][rows].astype(np.int64),
    )


def _torch_rows(array, rows, device, dtype=None, pin_memory=False):
    selected = np.ascontiguousarray(array[rows])
    tensor = torch.from_numpy(selected)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if pin_memory and device.type == 'cuda':
        tensor = tensor.pin_memory()
    return tensor.to(device=device, non_blocking=pin_memory and device.type == 'cuda')


def _unpack_torch(packed, count):
    if count == 0:
        return torch.empty((packed.shape[0], 0), dtype=torch.uint8, device=packed.device)
    positions = torch.arange(count, device=packed.device)
    return (packed[:, positions // 8] >> (positions % 8)) & 1


def decode_chunk_torch(chunk, device, indices=None, obs_dtype=torch.float32, pin_memory=True):
    """把 packed chunk 传到目标设备并并行展开成训练 batch。"""
    device = torch.device(device)
    metadata, arrays = chunk.metadata, chunk.arrays
    total = metadata['sample_count']
    rows = np.arange(total) if indices is None else np.asarray(indices, dtype=np.int64)
    obs = torch.zeros((len(rows), OBS_PLANES, OBS_TILES), dtype=obs_dtype, device=device)

    one_planes = torch.as_tensor(arrays['one_planes'].astype(np.int64), device=device)
    obs[:, one_planes, :] = 1
    planes = torch.as_tensor(arrays['constant_binary_planes'].astype(np.int64), device=device)
    packed = _torch_rows(arrays['constant_binary_bits'], rows, device, pin_memory=pin_memory)
    values = _unpack_torch(packed, len(planes)).to(obs_dtype)
    obs[:, planes, :] = values[:, :, None]
    planes = torch.as_tensor(arrays['constant_float_planes'].astype(np.int64), device=device)
    values = _torch_rows(
        arrays['constant_float_values'], rows, device, dtype=obs_dtype, pin_memory=pin_memory
    )
    obs[:, planes, :] = values[:, :, None]
    planes = torch.as_tensor(arrays['binary_planes'].astype(np.int64), device=device)
    packed = _torch_rows(arrays['binary_bits'], rows, device, pin_memory=pin_memory)
    values = _unpack_torch(packed, len(planes) * OBS_TILES).to(obs_dtype)
    obs[:, planes, :] = values.reshape(len(rows), len(planes), OBS_TILES)
    planes = torch.as_tensor(arrays['dense_planes'].astype(np.int64), device=device)
    values = _torch_rows(arrays['dense_values'], rows, device, dtype=obs_dtype, pin_memory=pin_memory)
    obs[:, planes, :] = values

    mask_packed = _torch_rows(arrays['mask_bits'], rows, device, pin_memory=pin_memory)
    masks = _unpack_torch(mask_packed, ACTION_SPACE).to(torch.bool)
    return (
        obs,
        _torch_rows(arrays['actions'], rows, device, torch.int64, pin_memory),
        masks,
        _torch_rows(arrays['steps_to_done'], rows, device, torch.int64, pin_memory),
        _torch_rows(arrays['rewards'], rows, device, torch.float64, pin_memory),
        _torch_rows(arrays['ranks'], rows, device, torch.int64, pin_memory),
    )


def write_chunk(chunk, filename, compression_level=1):
    """将 chunk 序列化、Zstd 压缩并原子落盘。"""
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    metadata_bytes = json.dumps(
        chunk.metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    payload = io.BytesIO()
    np.savez(
        payload,
        __metadata__=np.frombuffer(metadata_bytes, dtype=np.uint8),
        **chunk.arrays,
    )
    raw = payload.getvalue()
    compressed = zstd.ZstdCompressor(level=compression_level).compress(raw)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        with open(temporary, 'wb') as f:
            f.write(compressed)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, filename)
        directory_fd = os.open(filename.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)
    return {
        'compressed_bytes': len(compressed),
        'serialized_bytes': len(raw),
        'sha256': hashlib.sha256(compressed).hexdigest(),
    }


def read_chunk(filename, expected_sha256=None):
    """读取并校验一个缓存 chunk。"""
    with open(filename, 'rb') as f:
        compressed = f.read()
    actual_sha256 = hashlib.sha256(compressed).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f'缓存 chunk SHA256 不一致: expected={expected_sha256}, actual={actual_sha256}'
        )
    raw = zstd.ZstdDecompressor().decompress(compressed)
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = json.loads(archive['__metadata__'].tobytes().decode('utf-8'))
        arrays = {name: archive[name] for name in archive.files if name != '__metadata__'}
    if metadata.get('schema_version') != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f'不支持的缓存 schema: {metadata.get("schema_version")}')
    return EncodedChunk(metadata=metadata, arrays=arrays)


def sample_digest(batch, row):
    """计算解码后单条样本的稳定摘要，用于跨进程验证。"""
    digest = hashlib.sha256()
    for value in batch:
        array = np.ascontiguousarray(value[row])
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PackedCacheBatchIterator:
    """按 chunk 打乱并在 CPU/GPU 上解码固定大小训练 batch。"""

    def __init__(self, manifest_file, batch_size, device, shuffle_seed=0, drop_last=False):
        self.manifest_file = Path(manifest_file).resolve()
        with open(self.manifest_file, encoding='utf-8') as f:
            self.manifest = json.load(f)
        if self.manifest.get('status') != 'complete':
            raise RuntimeError('缓存 manifest 尚未完成')
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle_seed = shuffle_seed
        self.drop_last = drop_last

    def __iter__(self):
        rng = random.Random(self.shuffle_seed)
        chunks = list(self.manifest['chunks'])
        rng.shuffle(chunks)
        carry = None
        for entry in chunks:
            filename = self.manifest_file.parent / entry['file']
            chunk = read_chunk(filename, entry['sha256'])
            indices = list(range(chunk.metadata['sample_count']))
            rng.shuffle(indices)
            decoded = decode_chunk_torch(chunk, self.device, indices=np.asarray(indices))
            start = 0
            if carry is not None:
                needed = self.batch_size - carry[0].shape[0]
                taken = tuple(value[:needed] for value in decoded)
                if taken[0].shape[0] == needed:
                    yield tuple(torch.cat((left, right), dim=0) for left, right in zip(carry, taken))
                    carry = None
                    start = needed
                else:
                    carry = tuple(torch.cat((left, right), dim=0) for left, right in zip(carry, taken))
                    continue
            while start + self.batch_size <= decoded[0].shape[0]:
                yield tuple(value[start:start + self.batch_size] for value in decoded)
                start += self.batch_size
            if start < decoded[0].shape[0]:
                carry = tuple(value[start:] for value in decoded)
        if carry is not None and not self.drop_last:
            yield carry
