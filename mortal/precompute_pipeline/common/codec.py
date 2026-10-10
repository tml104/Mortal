"""固定语义的无损六元样本 chunk 格式。"""

from dataclasses import dataclass
import struct

import numpy as np
import zstandard as zstd


OBS_PLANES = 1012
OBS_TILES = 34
ACTION_SPACE = 46
MAGIC = b'MRTLPC01'
SCHEMA_VERSION = 1
HEADER = struct.Struct('<8sII')

SCALAR_BINARY_PLANES = tuple(sorted([
    *range(4, 7),
    *range(15, 23),
    *range(32, 35),
    *[35 + 4 * i + j for i in range(24) for j in (2, 3)],
    *[
        base + 8 * i + j
        for base in (132, 327, 522)
        for i in range(24)
        for j in range(4, 8)
    ],
    *[723 + 7 * i + j for i in range(4) for j in range(4, 7)],
    *[751 + 5 * i + 4 for i in range(16)],
    *[836 + 3 * i + j for i in range(6) for j in (1, 2)],
    *range(854, 860),
    *range(861, 871),
    *range(872, 874),
    *range(878, 885),
    *range(887, 889),
]))
SCALAR_FLOAT_PLANES = tuple([
    *range(7, 15),
    *range(23, 25),
    27,
    *range(717, 723),
    *range(889, 891),
])
TILE_FLOAT_PLANES = tuple([
    131,
    *range(324, 327),
    *range(519, 522),
    *range(714, 717),
    835,
    *range(961, 1012),
])
_RESERVED_PLANES = set(
    SCALAR_BINARY_PLANES + SCALAR_FLOAT_PLANES + TILE_FLOAT_PLANES
)
TILE_BINARY_PLANES = tuple(
    plane for plane in range(OBS_PLANES) if plane not in _RESERVED_PLANES
)

SCALAR_BINARY_BYTES = (len(SCALAR_BINARY_PLANES) + 7) // 8
TILE_BINARY_BITS = len(TILE_BINARY_PLANES) * OBS_TILES
TILE_BINARY_BYTES = (TILE_BINARY_BITS + 7) // 8
MASK_BYTES = (ACTION_SPACE + 7) // 8
BYTES_PER_SAMPLE = (
    SCALAR_BINARY_BYTES
    + TILE_BINARY_BYTES
    + len(SCALAR_FLOAT_PLANES) * 4
    + len(TILE_FLOAT_PLANES) * OBS_TILES * 4
    + 1
    + MASK_BYTES
    + 2
    + 8
    + 1
)

assert [
    len(SCALAR_BINARY_PLANES),
    len(TILE_BINARY_PLANES),
    len(SCALAR_FLOAT_PLANES),
    len(TILE_FLOAT_PLANES),
] == [417, 514, 19, 62]
assert BYTES_PER_SAMPLE == 10_764


@dataclass
class SampleBatch:
    obs: np.ndarray
    actions: np.ndarray
    masks: np.ndarray
    steps_to_done: np.ndarray
    rewards: np.ndarray
    ranks: np.ndarray


@dataclass
class PackedChunk:
    sample_count: int
    scalar_binary_bits: np.ndarray
    tile_binary_bits: np.ndarray
    scalar_float_values: np.ndarray
    tile_float_values: np.ndarray
    actions: np.ndarray
    mask_bits: np.ndarray
    steps_to_done: np.ndarray
    rewards: np.ndarray
    ranks: np.ndarray


def _pack_bits(values):
    values = np.asarray(values, dtype=np.uint8, order='C')
    if values.ndim != 2:
        raise ValueError(f'bit-pack 输入必须是二维数组，实际为 {values.shape}')
    return np.packbits(values, axis=1, bitorder='little')


def _observation_array(value):
    """核对观测形状和精度并提供连续数组，不扫描元素值。"""
    obs = np.asarray(value)
    if obs.dtype != np.float32:
        raise ValueError(f'obs 必须是 float32，实际为 {obs.dtype}')
    obs = np.ascontiguousarray(obs)
    if obs.ndim != 3 or obs.shape[1:] != (OBS_PLANES, OBS_TILES):
        raise ValueError(f'obs shape 必须是 (N,{OBS_PLANES},{OBS_TILES})，实际为 {obs.shape}')
    return obs


def validate_observation_semantics(value):
    """供前置筛选完整检查观测语义，编码阶段仅复用结构约束。"""
    obs = _observation_array(value)
    scalar_binary = obs[:, SCALAR_BINARY_PLANES, :]
    if not np.all(scalar_binary == scalar_binary[:, :, :1]):
        raise ValueError('语义为整行二值的 observation 通道出现逐牌变化')
    scalar_binary = scalar_binary[:, :, 0]
    tile_binary = obs[:, TILE_BINARY_PLANES, :]
    if not np.all((scalar_binary == 0) | (scalar_binary == 1)):
        raise ValueError('语义为整行二值的 observation 通道含非二值')
    if not np.all((tile_binary == 0) | (tile_binary == 1)):
        raise ValueError('语义为逐牌二值的 observation 通道含非二值')
    scalar_float = obs[:, SCALAR_FLOAT_PLANES, :]
    if not np.all(scalar_float == scalar_float[:, :, :1]):
        raise ValueError('语义为整行连续值的 observation 通道出现逐牌变化')
    if not np.isfinite(obs).all():
        raise ValueError('obs 含 NaN/Inf')
    return obs


def _normalize_batch(batch):
    obs = _observation_array(batch.obs)
    sample_count = len(obs)
    if sample_count <= 0:
        raise ValueError('chunk 至少需要一个样本')

    actions = np.asarray(batch.actions, dtype=np.int64)
    masks = np.asarray(batch.masks)
    steps = np.asarray(batch.steps_to_done, dtype=np.int64)
    rewards = np.asarray(batch.rewards, dtype=np.float64)
    ranks = np.asarray(batch.ranks, dtype=np.int64)
    expected_shapes = {
        'actions': (sample_count,),
        'masks': (sample_count, ACTION_SPACE),
        'steps_to_done': (sample_count,),
        'rewards': (sample_count,),
        'ranks': (sample_count,),
    }
    actual_shapes = {
        'actions': actions.shape,
        'masks': masks.shape,
        'steps_to_done': steps.shape,
        'rewards': rewards.shape,
        'ranks': ranks.shape,
    }
    for name, expected in expected_shapes.items():
        if actual_shapes[name] != expected:
            raise ValueError(f'{name} shape 必须是 {expected}，实际为 {actual_shapes[name]}')
    if not np.isfinite(rewards).all():
        raise ValueError('rewards 含 NaN/Inf')
    if np.any(actions < 0) or np.any(actions >= ACTION_SPACE):
        raise ValueError('action 超出 0..45')
    if not np.all((masks == 0) | (masks == 1)):
        raise ValueError('mask 必须是二值数组')
    masks = masks.astype(np.bool_, copy=False)
    if not masks[np.arange(sample_count), actions].all():
        raise ValueError('action 不在对应样本的合法 mask 中')
    if np.any(steps < 0) or np.any(steps > np.iinfo(np.uint16).max):
        raise ValueError('steps_to_done 超出 uint16 范围')
    if np.any(ranks < 0) or np.any(ranks > 3):
        raise ValueError('rank 超出 0..3')

    # 标量只复制首列；逐牌二值先连续提取并缩为 uint8，避免 reshape 搬运 float32 大副本。
    scalar_binary = obs[:, SCALAR_BINARY_PLANES, 0]
    tile_binary = np.take(obs, TILE_BINARY_PLANES, axis=1).astype(np.uint8)
    scalar_float = obs[:, SCALAR_FLOAT_PLANES, 0]

    return (
        sample_count,
        scalar_binary,
        tile_binary,
        scalar_float,
        np.take(obs, TILE_FLOAT_PLANES, axis=1),
        actions,
        masks,
        steps,
        rewards,
        ranks,
    )


def encode_chunk(batch, compression_level=1):
    """按固定语义编码一批样本，并返回单个 Zstd frame。"""
    if not isinstance(compression_level, int):
        raise ValueError('compression_level 必须是整数')
    values = _normalize_batch(batch)
    (
        sample_count,
        scalar_binary,
        tile_binary,
        scalar_float,
        tile_float,
        actions,
        masks,
        steps,
        rewards,
        ranks,
    ) = values
    arrays = (
        _pack_bits(scalar_binary),
        _pack_bits(tile_binary.reshape(sample_count, -1)),
        np.asarray(scalar_float, dtype='<f4', order='C'),
        np.asarray(tile_float, dtype='<f4', order='C'),
        np.asarray(actions, dtype=np.uint8, order='C'),
        _pack_bits(masks),
        np.asarray(steps, dtype='<u2', order='C'),
        np.asarray(rewards, dtype='<f8', order='C'),
        np.asarray(ranks, dtype=np.uint8, order='C'),
    )
    raw = b''.join((
        HEADER.pack(MAGIC, SCHEMA_VERSION, sample_count),
        *(array.tobytes(order='C') for array in arrays),
    ))
    expected_size = HEADER.size + sample_count * BYTES_PER_SAMPLE
    if len(raw) != expected_size:
        raise RuntimeError(f'chunk 编码长度错误: expected={expected_size}, actual={len(raw)}')
    return zstd.ZstdCompressor(level=compression_level).compress(raw)


def _read_array(payload, offset, dtype, shape):
    count = int(np.prod(shape))
    size = count * np.dtype(dtype).itemsize
    end = offset + size
    if end > len(payload):
        raise ValueError('chunk 载荷提前结束')
    value = np.frombuffer(payload, dtype=dtype, count=count, offset=offset).reshape(shape)
    return value, end


def decode_packed_chunk(compressed):
    """读取 chunk 为紧凑 CPU 数组，不展开 observation。"""
    try:
        payload = zstd.ZstdDecompressor().decompress(compressed)
    except zstd.ZstdError as exc:
        raise ValueError(f'chunk Zstd 解压失败: {exc}') from exc
    if len(payload) < HEADER.size:
        raise ValueError('chunk 头部不完整')
    magic, schema_version, sample_count = HEADER.unpack_from(payload)
    if magic != MAGIC:
        raise ValueError(f'chunk magic 不兼容: {magic!r}')
    if schema_version != SCHEMA_VERSION:
        raise ValueError(f'chunk schema 不兼容: {schema_version}')
    if sample_count <= 0:
        raise ValueError('chunk 样本数必须为正数')
    expected_size = HEADER.size + sample_count * BYTES_PER_SAMPLE
    if len(payload) != expected_size:
        raise ValueError(
            f'chunk 长度不兼容: expected={expected_size}, actual={len(payload)}'
        )

    offset = HEADER.size
    scalar_binary, offset = _read_array(
        payload, offset, np.uint8, (sample_count, SCALAR_BINARY_BYTES),
    )
    tile_binary, offset = _read_array(
        payload, offset, np.uint8, (sample_count, TILE_BINARY_BYTES),
    )
    scalar_float, offset = _read_array(
        payload, offset, '<f4', (sample_count, len(SCALAR_FLOAT_PLANES)),
    )
    tile_float, offset = _read_array(
        payload,
        offset,
        '<f4',
        (sample_count, len(TILE_FLOAT_PLANES), OBS_TILES),
    )
    actions, offset = _read_array(payload, offset, np.uint8, (sample_count,))
    masks, offset = _read_array(payload, offset, np.uint8, (sample_count, MASK_BYTES))
    steps, offset = _read_array(payload, offset, '<u2', (sample_count,))
    rewards, offset = _read_array(payload, offset, '<f8', (sample_count,))
    ranks, offset = _read_array(payload, offset, np.uint8, (sample_count,))
    if offset != len(payload):
        raise ValueError('chunk 含未解释的尾部数据')
    return PackedChunk(
        sample_count=sample_count,
        scalar_binary_bits=scalar_binary,
        tile_binary_bits=tile_binary,
        scalar_float_values=scalar_float,
        tile_float_values=tile_float,
        actions=actions,
        mask_bits=masks,
        steps_to_done=steps,
        rewards=rewards,
        ranks=ranks,
    )


def _normalize_indices(indices, sample_count):
    if indices is None:
        return np.arange(sample_count, dtype=np.int64)
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError('indices 必须是一维数组')
    if np.any(indices < 0) or np.any(indices >= sample_count):
        raise ValueError('indices 超出 chunk 范围')
    return indices


def decode_chunk_numpy(chunk, indices=None):
    """将紧凑 chunk 的选定行无损展开为标准 NumPy 六元 batch。"""
    if not isinstance(chunk, PackedChunk):
        chunk = decode_packed_chunk(chunk)
    indices = _normalize_indices(indices, chunk.sample_count)
    sample_count = len(indices)
    obs = np.zeros((sample_count, OBS_PLANES, OBS_TILES), dtype=np.float32)
    scalar_binary = np.unpackbits(
        chunk.scalar_binary_bits[indices],
        axis=1,
        count=len(SCALAR_BINARY_PLANES),
        bitorder='little',
    )
    tile_binary = np.unpackbits(
        chunk.tile_binary_bits[indices],
        axis=1,
        count=TILE_BINARY_BITS,
        bitorder='little',
    ).reshape(sample_count, len(TILE_BINARY_PLANES), OBS_TILES)
    obs[:, SCALAR_BINARY_PLANES, :] = scalar_binary[:, :, None]
    obs[:, TILE_BINARY_PLANES, :] = tile_binary
    obs[:, SCALAR_FLOAT_PLANES, :] = chunk.scalar_float_values[indices, :, None]
    obs[:, TILE_FLOAT_PLANES, :] = chunk.tile_float_values[indices]
    masks = np.unpackbits(
        chunk.mask_bits[indices],
        axis=1,
        count=ACTION_SPACE,
        bitorder='little',
    ).astype(np.bool_, copy=False)
    return SampleBatch(
        obs=obs,
        actions=chunk.actions[indices].copy(),
        masks=masks,
        steps_to_done=chunk.steps_to_done[indices].copy(),
        rewards=chunk.rewards[indices].copy(),
        ranks=chunk.ranks[indices].copy(),
    )


def decode_chunk_torch(chunk, device, indices=None):
    """只传输选定行的紧凑数组，并在目标设备展开六元张量。"""
    import torch

    if not isinstance(chunk, PackedChunk):
        chunk = decode_packed_chunk(chunk)
    indices = _normalize_indices(indices, chunk.sample_count)
    sample_count = len(indices)
    device = torch.device(device)

    def to_device(value):
        return torch.as_tensor(value[indices], device=device)

    def unpack(packed, count):
        shifts = torch.arange(8, dtype=torch.uint8, device=device)
        bits = torch.bitwise_and(
            torch.bitwise_right_shift(packed.unsqueeze(-1), shifts),
            1,
        )
        return bits.reshape(sample_count, -1)[:, :count]

    scalar_binary = unpack(to_device(chunk.scalar_binary_bits), len(SCALAR_BINARY_PLANES))
    tile_binary = unpack(to_device(chunk.tile_binary_bits), TILE_BINARY_BITS).reshape(
        sample_count,
        len(TILE_BINARY_PLANES),
        OBS_TILES,
    )
    scalar_float = to_device(chunk.scalar_float_values)
    tile_float = to_device(chunk.tile_float_values)
    obs = torch.zeros(
        (sample_count, OBS_PLANES, OBS_TILES),
        dtype=torch.float32,
        device=device,
    )
    obs[:, SCALAR_BINARY_PLANES, :] = scalar_binary.to(torch.float32).unsqueeze(-1)
    obs[:, TILE_BINARY_PLANES, :] = tile_binary.to(torch.float32)
    obs[:, SCALAR_FLOAT_PLANES, :] = scalar_float.unsqueeze(-1)
    obs[:, TILE_FLOAT_PLANES, :] = tile_float
    masks = unpack(to_device(chunk.mask_bits), ACTION_SPACE).to(torch.bool)
    return (
        obs,
        to_device(chunk.actions).to(torch.int64),
        masks,
        to_device(chunk.steps_to_done).to(torch.int64),
        to_device(chunk.rewards),
        to_device(chunk.ranks).to(torch.int64),
    )
