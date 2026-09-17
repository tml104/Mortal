"""定义 Transformer-GRP 预计算缓存的内存结构、磁盘格式和训练迭代器。

预计算器传入的逻辑样本为六元组：

``obs, action, mask, steps_to_done, kyoku_reward, player_rank``。

其中 observation 在 schema v1 下固定为 ``(1012, 34)``。直接保存 dense float32 会让
每条样本仅 observation 就占用约 134 KiB，因此本模块利用 observation plane 的值域
特征进行编码：全零/全一 plane 只记录编号，二值数据按 bit 保存，连续值收窄为
float16。动作、mask、折扣步数、奖励和顺位也使用与值域匹配的紧凑类型。

一个 ``EncodedChunk`` 在内存中由解释元数据和若干 NumPy 数组组成。写盘时先用
``np.savez`` 生成未压缩 NPZ，再对完整字节流执行 Zstd 压缩，形成 ``*.npz.zst``。
读取器可以在 CPU 上恢复 NumPy 数组用于验证，也可以把紧凑数组传到目标设备并用
Torch 恢复训练张量。``PackedCacheBatchIterator`` 在此基础上负责确定性洗牌、切 batch
和跨 chunk 拼接尾部样本。

本模块只规定缓存载荷及读取顺序。源文件选择、缓存 contract、抽样验证记录和 manifest
发布由 ``precompute_grp_transformer_cache.py`` 管理；模型优化与 checkpoint 由
``train_with_precomputed_cache.py`` 管理。
"""

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


# schema 版本描述本文件规定的数组名称、dtype 和重构方法。shape 常量与 libriichi 的
# observation v4 及当前 46 维动作空间保持一致；其他 observation 版本不能直接复用格式。
CACHE_SCHEMA_VERSION = 1
OBS_PLANES = 1012
OBS_TILES = 34
ACTION_SPACE = 46


@dataclass(frozen=True)
class EncodedChunk:
    """保存一个 plane-aware 缓存 chunk 的解释信息和实际数组。

    ``metadata`` 是可 JSON 序列化的字典，记录 schema、样本数、observation shape、plane
    分类数量和量化/体积统计。``arrays`` 保存 plane 编号、紧凑 observation、五个其他
    训练字段及三个来源字段。dataclass 冻结只阻止重新绑定两个属性，字典及数组内容仍然
    可变；调用方应把实例视为只读数据。
    """

    metadata: dict
    arrays: dict[str, np.ndarray]


def sha256_file(filename):
    """按 1 MiB 分块计算文件 SHA256，避免把大型 chunk 再复制到单个内存缓冲。"""
    digest = hashlib.sha256()
    with open(filename, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _pack_bits(values):
    """沿二维数组的第二维按 little-endian bit 顺序压缩布尔/0-1 值。

    输入形状约定为 ``(N, logical_width)``，输出宽度为
    ``ceil(logical_width / 8)``。最后一个字节可能含未使用的高位；解码器依靠原始逻辑
    宽度只取有效 bit。统一转成 ``uint8`` 后再调用 NumPy，保证位移和序列化表示稳定。
    """
    values = np.asarray(values, dtype=np.uint8)
    if values.ndim != 2:
        raise ValueError(f'bit-pack 输入必须是二维数组，实际为 {values.shape}')
    return np.packbits(values, axis=1, bitorder='little')


def _validate_sample_arrays(obs, actions, masks, steps_to_done, rewards, ranks, source):
    """在收窄 dtype 前验证一个 chunk 的形状、有限值、业务范围和来源坐标。

    编码会把多个 int64 字段转换为 uint8/uint16/uint32。NumPy 的窄类型转换可能静默截断
    越界值，因此必须先在原始精度上检查。动作还要满足对应行的合法动作 mask；缓存一旦
    固化，训练阶段已经无法回到牌谱上下文修复这些标签。
    """
    sample_count = obs.shape[0]
    # observation 的第 0 维定义 N，所有其他字段必须与同一 N 对齐。
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
    # 只有 observation 会量化为 float16；reward 保持 float64，但两者都必须是有限值。
    if not np.isfinite(obs).all() or not np.isfinite(rewards).all():
        raise ValueError('observation 或 reward 含 NaN/Inf')
    if np.abs(obs).max(initial=0) > np.finfo(np.float16).max:
        raise ValueError('observation 超出 float16 可表示范围')
    if np.any(actions < 0) or np.any(actions >= ACTION_SPACE):
        raise ValueError('action 超出动作空间')
    # 逐行高级索引取出日志动作对应的 mask 位，任何 False 都表示样本标签不合法。
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
    """按 chunk 内每个 observation plane 的值域编码完整训练样本。

    输入 observation 为 ``(N, 1012, 34)``，其余六元组字段及来源坐标共享样本维 N。
    返回的 ``EncodedChunk`` 仍位于内存中；本函数不执行 NPZ 序列化或磁盘写入。

    二值数据保持无损，连续 observation 使用 float16，reward 保留 float64。plane 分类
    基于当前 chunk 的全部样本，因此改变 chunk 边界可能改变分类结果和压缩率。
    """
    # 先统一为编码契约规定的工作 dtype 和 C 连续 observation，再执行形状与值域检查。
    # 来源字段暂时保留 int64，验证后才收窄到各自的磁盘 dtype。
    obs = np.asarray(obs, dtype=np.float32, order='C') # N
    actions = np.asarray(actions, dtype=np.int64)
    masks = np.asarray(masks, dtype=np.bool_)
    steps_to_done = np.asarray(steps_to_done, dtype=np.int64)
    rewards = np.asarray(rewards, dtype=np.float64)
    ranks = np.asarray(ranks, dtype=np.int64)
    source = {name: np.asarray(value, dtype=np.int64) for name, value in source.items()}

    # 关于source: 描述 chunk 中第 r 行样本来自哪份牌谱、哪个玩家视角，以及该视角中的第几个决策？
    '''
        source = {
            'file_ids': np.ndarray,        # (N,)
            'player_ids': np.ndarray,      # (N,)
            'sample_indices': np.ndarray,  # (N,)
        }
    对于任意行 r， 对应的来源就是：
    source['file_ids'][r]
    source['player_ids'][r]
    source['sample_indices'][r]
    '''
    _validate_sample_arrays(obs, actions, masks, steps_to_done, rewards, ranks, source)

    sample_count = obs.shape[0] # N
    # axis=(0, 2) 同时归约样本维 N 和每个 plane 的 34 个位置，结果形状为 (1012,)；
    # 因而 all_zero[p] 表示 plane p 在当前整个 chunk 中处处为 0。
    all_zero = np.all(obs == 0, axis=(0, 2)) # (1012,)
    '''
    all_zero = []

    for plane_id in range(1012):
        当前plane的全部数据 = obs[:, plane_id, :]
        是否全部为0 = np.all(当前plane的全部数据 == 0) # 对每一个plane，检查它在所有样本、所有34个位置中是否全部为0。等价于再写两层for然后沿着x、z遍历检查是否全为0
        all_zero.append(是否全部为0)

    '''

    all_one = np.all(obs == 1, axis=(0, 2)) # (1012,)
    remaining = ~(all_zero | all_one) # 既不是全0，也不是全1，则remaining[p] = True，p是plane_id

    # obs[:, :, :1] 为每个样本、每个 plane 的第一个位置。广播比较后归约 N 与位置维，
    # constant_per_sample[p] 表示 plane p 在每条样本内部都沿 34 个位置保持常量
    constant_per_sample = np.all(obs == obs[:, :, :1], axis=(0, 2)) # (1012,)
    '''
    obs == obs[:, :, :1]会自动将obs[:, :, :1]（形状为(N, 1012, 1)）广播到(N, 1012, 34)，其中在axis=2上复制34次。
    然后再沿着axis=(0, 2)取all，得到(1012,)的布尔数组，表示每个plane是否在所有样本的34个位置上都是常量。
    '''

    scalar_values = obs[:, :, 0] # scalar_values: (N, 1012)
    # 只有所有样本的代表值均为 0/1，逐样本常量 plane 才能进一步使用 bit-pack。
    # (scalar_values == 0) | (scalar_values == 1)会先得到(N,1012)的布尔数组，表示每个样本的每个plane是否为0或1
    # 再沿着样本维度取all，得到(1012,)的布尔数组，表示每个plane是否所有样本都是0或1
    '''
    沿着axis的含义：
    axis是用于判断是否全满足True的条件的内部循环轴。对于不在axis当中指定的轴，它们则决定最终结果的形状（可以理解为外部循环）
    比如，对于scalar_binary这里，np.all沿着axis=0的含义就是：对(N,1012)这个形状的张量(scalar_values == 0) | (scalar_values == 1)（下文简称m）
    设x是在axis=0变化的变量，则

    result = [] # 最终结果形状在这里预期是(1012,)
    for y in range(1012): # 外部循环变量

        flag = True
        for x in range(N): # 内部循环变量

            if m[x, y] == False:
                flag = False
                break
        result[y] = flag

    如果m的形状换成(N, 1012, 34), axis换成 (0, 2)，则上述伪代码变为：
    result = [] # 最终结果形状在这里预期是(1012,)
    for y in range(1012): # 外部循环变量
        flag = True
        for x in range(N): # 内部循环变量
            for z in range(34): # 内部循环变量
                if m[x, y, z] == False:
                    flag = False
                    break
        result[y] = flag
    '''
    scalar_binary = np.all((scalar_values == 0) | (scalar_values == 1), axis=0) # scalar_binary: (1012,)
    constant_binary = remaining & constant_per_sample & scalar_binary # (1012,)
    constant_float = remaining & constant_per_sample & ~scalar_binary # (1012,)
    # 后续 binary/dense 分类只处理每条样本内部仍会沿位置维变化的 plane。
    remaining &= ~constant_per_sample # (1012,)，只保留非全零、非全一（也即为浮点数）、并且至少有一条样本的34个位置会变化（也即浮点数的值在plane这个轴上不完全相同）
    binary = remaining & np.all((obs == 0) | (obs == 1), axis=(0, 2)) # (1012,)，值均为0、1，但是在34个位置上不完全相同的plane
    dense = remaining & ~binary # (1012,)，值均为浮点数，并且在34个位置上不完全相同的plane

    # 分类数组只存原始 plane 编号，uint16 足以覆盖 0..1011。
    '''
    np.flatnonzero:

    zero_plane_ids = []

    for plane_id in range(len(all_zero)):
        if all_zero[plane_id]:
            zero_plane_ids.append(plane_id)

    本质上就是提取出所有为True的plane_id，返回一个一维数组，里面存储了所有满足条件的plane_id。
    '''
    categories = {
        'zero_planes': np.flatnonzero(all_zero).astype(np.uint16),
        'one_planes': np.flatnonzero(all_one).astype(np.uint16),
        'constant_binary_planes': np.flatnonzero(constant_binary).astype(np.uint16),
        'constant_float_planes': np.flatnonzero(constant_float).astype(np.uint16),
        'binary_planes': np.flatnonzero(binary).astype(np.uint16),
        'dense_planes': np.flatnonzero(dense).astype(np.uint16),
    }
    # 六类必须形成 1012 个 plane 的互斥完备分区，否则解码时会遗漏或重复覆盖位置。
    partition = np.concatenate(tuple(categories.values()))
    if len(partition) != OBS_PLANES or len(np.unique(partition)) != OBS_PLANES:
        raise RuntimeError('observation plane 分类没有形成完整互斥分区')

    # constant_* 每个样本每个 plane 只保存位置 0 的代表值；binary/dense 保留全部位置。
    # binary_values 展平 plane 与 tile 两维，以便 _pack_bits 统一沿第二维压缩。
    constant_binary_values = scalar_values[:, categories['constant_binary_planes']]
    binary_values = obs[:, categories['binary_planes'], :].reshape(sample_count, -1)
    constant_float_values = scalar_values[:, categories['constant_float_planes']]
    dense_values = obs[:, categories['dense_planes'], :]

    # 量化统计在实际转换前比较 float32 原值与 float16 往返值。它只覆盖两类连续数据，
    # 全零、全一和二值 plane 不计入 quantized_count。
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
        # plane 目录决定下面各载荷数组应回填到 observation 的哪些通道。
        **categories,
        # observation 的紧凑载荷。
        'constant_binary_bits': _pack_bits(constant_binary_values),
        'constant_float_values': constant_float_values.astype(np.float16),
        'binary_bits': _pack_bits(binary_values),
        'dense_values': dense_values.astype(np.float16),
        # 训练六元组中除 observation 外的五个字段。
        'actions': actions.astype(np.uint8),
        'mask_bits': _pack_bits(masks),
        'steps_to_done': steps_to_done.astype(np.uint16),
        'rewards': rewards.astype(np.float64),
        'ranks': ranks.astype(np.uint8),
        # 来源坐标只用于审计和验证，不由 decode_chunk_* 返回给训练器。
        'file_ids': source['file_ids'].astype(np.uint32),
        'player_ids': source['player_ids'].astype(np.uint8),
        'sample_indices': source['sample_indices'].astype(np.uint16),
    }
    # packed_array_bytes 是所有 .npy 载荷本身的字节和，不包含 NPZ 容器头及 Zstd 开销。
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
    """将 little-endian packed bytes 恢复为指定逻辑宽度的二维 uint8 数组。"""
    if count == 0:
        # np.unpackbits 对零宽逻辑字段没有统一的目标 shape，这里显式保留样本维。
        return np.empty((packed.shape[0], 0), dtype=np.uint8)
    # count 会裁掉 _pack_bits 为字节对齐补在末尾的无效 bit。
    return np.unpackbits(packed, axis=1, count=count, bitorder='little')


def decode_chunk_numpy(chunk, indices=None):
    """在 CPU 上把选定行恢复为标准六元 NumPy batch。

    ``indices=None`` 表示按原顺序解码整个 chunk；显式 indices 可以抽样或重排行。返回值
    不含三个来源数组，字段顺序和 dtype 与主训练接口一致。该实现主要用于预计算阶段把
    原始样本与编码往返结果逐值比较。
    """
    metadata, arrays = chunk.metadata, chunk.arrays
    total = metadata['sample_count']
    rows = np.arange(total) if indices is None else np.asarray(indices, dtype=np.int64)
    # 以全零 dense observation 为起点，所以 zero_planes 无需执行任何回填。
    obs = np.zeros((len(rows), OBS_PLANES, OBS_TILES), dtype=np.float32)

    # one_planes 在所有样本和位置上都为 1，只需按 plane 编号整片赋值。
    one_planes = arrays['one_planes'].astype(np.int64)
    obs[:, one_planes, :] = 1
    # 逐样本常量二值：解包得到 (rows, planes)，再通过末尾单例维广播到 34 个位置。
    planes = arrays['constant_binary_planes'].astype(np.int64)
    values = _unpack_numpy(arrays['constant_binary_bits'][rows], len(planes))
    obs[:, planes, :] = values[:, :, None]
    # 逐样本常量连续值已经量化为 float16；恢复 float32 后同样广播到位置维。
    planes = arrays['constant_float_planes'].astype(np.int64)
    obs[:, planes, :] = arrays['constant_float_values'][rows].astype(np.float32)[:, :, None]
    # 普通二值 plane 的逻辑宽度是 plane_count * 34，解包后再恢复三维布局。
    planes = arrays['binary_planes'].astype(np.int64)
    values = _unpack_numpy(arrays['binary_bits'][rows], len(planes) * OBS_TILES)
    obs[:, planes, :] = values.reshape(len(rows), len(planes), OBS_TILES)
    # 普通连续 plane 只需从 float16 扩展为 float32 并写回原始 plane 位置。
    planes = arrays['dense_planes'].astype(np.int64)
    obs[:, planes, :] = arrays['dense_values'][rows].astype(np.float32)

    # 46 个合法动作位使用与 observation 二值载荷相同的 little-endian 规则。
    masks = _unpack_numpy(arrays['mask_bits'][rows], ACTION_SPACE).astype(np.bool_)
    # 编码时收窄的标量字段恢复为训练器期望的 dtype；float16 observation 的精度损失仍在。
    return (
        obs,
        arrays['actions'][rows].astype(np.int64),
        masks,
        arrays['steps_to_done'][rows].astype(np.int64),
        arrays['rewards'][rows].astype(np.float64),
        arrays['ranks'][rows].astype(np.int64),
    )


def _torch_rows(array, rows, device, dtype=None, pin_memory=False):
    """选择 NumPy 行，按需转换 dtype，并把连续张量传到目标 Torch 设备。

    NumPy 高级索引会产生新数组，``ascontiguousarray`` 进一步保证
    ``torch.from_numpy`` 可以直接共享这块 CPU 存储。CUDA 路径可先把张量放入锁页内存，
    从而允许 ``to(..., non_blocking=True)`` 发起异步副本；本迭代器没有独立预取线程，
    因此这里只优化单次传输条件。
    """
    selected = np.ascontiguousarray(array[rows]) # selected.dtype = float16 并且在 CPU 上
    tensor = torch.from_numpy(selected) # tensor.dtype = float16 且在CPU上
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    if pin_memory and device.type == 'cuda':
        tensor = tensor.pin_memory() # 将CPU张量放入锁页内存，为CPU到GPU传输做准备。
    return tensor.to(device=device, non_blocking=pin_memory and device.type == 'cuda')


def _unpack_torch(packed, count):
    """在 packed 张量所在设备按 little-endian 规则展开指定数量的 bit。"""
    if count == 0:
        return torch.empty((packed.shape[0], 0), dtype=torch.uint8, device=packed.device)
    # 第 k 个逻辑 bit 位于第 k//8 个字节的第 k%8 位，与 np.packbits(bitorder='little')
    # 完全对应。positions 只覆盖 count，因此不会返回末字节的对齐填充位。
    positions = torch.arange(count, device=packed.device)
    return (packed[:, positions // 8] >> (positions % 8)) & 1


def decode_chunk_torch(chunk, device, indices=None, obs_dtype=torch.float32, pin_memory=True):
    """把选定行传到目标设备，并用 Torch 恢复标准六元训练 batch。

    ``indices`` 同时决定选中哪些样本及其返回顺序；缓存迭代器利用这一点在解码前完成
    chunk 内洗牌。observation 直接在目标设备分配为 ``obs_dtype``，bit 展开和 plane
    回填也在该设备执行。默认 CUDA 路径使用锁页 CPU 中间张量；传入 CPU device 时会走
    普通同步路径。

    当前调用者通常一次解码整个 chunk，然后再切成训练 batch，所以瞬时显存取决于 chunk
    样本数，而非只取决于训练 batch size。
    """
    device = torch.device(device)
    metadata, arrays = chunk.metadata, chunk.arrays
    total = metadata['sample_count']
    rows = np.arange(total) if indices is None else np.asarray(indices, dtype=np.int64)
    # 默认零值同时完成 zero_planes 的重构。
    obs = torch.zeros((len(rows), OBS_PLANES, OBS_TILES), dtype=obs_dtype, device=device)

    # 以下回填顺序与 decode_chunk_numpy 保持一致；plane 分类互斥，所以不会相互覆盖。
    one_planes = torch.as_tensor(arrays['one_planes'].astype(np.int64), device=device)
    obs[:, one_planes, :] = 1
    planes = torch.as_tensor(arrays['constant_binary_planes'].astype(np.int64), device=device)
    packed = _torch_rows(arrays['constant_binary_bits'], rows, device, pin_memory=pin_memory)
    values = _unpack_torch(packed, len(planes)).to(obs_dtype)
    obs[:, planes, :] = values[:, :, None] # 常量二值plane
    planes = torch.as_tensor(arrays['constant_float_planes'].astype(np.int64), device=device)
    values = _torch_rows(
        arrays['constant_float_values'], rows, device, dtype=obs_dtype, pin_memory=pin_memory
    ) # 注意这里传入了 obs_dtype ，而其默认值为torch.float32，因此相当于将在CPU上转换为float32，随后再传到device上
    obs[:, planes, :] = values[:, :, None] # 常量连续plane
    planes = torch.as_tensor(arrays['binary_planes'].astype(np.int64), device=device)
    packed = _torch_rows(arrays['binary_bits'], rows, device, pin_memory=pin_memory)
    values = _unpack_torch(packed, len(planes) * OBS_TILES).to(obs_dtype)
    obs[:, planes, :] = values.reshape(len(rows), len(planes), OBS_TILES) # 普通二值plane
    planes = torch.as_tensor(arrays['dense_planes'].astype(np.int64), device=device)
    # 这里传入默认值为 torch.float32 的 obs_dtype，所以先在 CPU 上转换，再传到 device。
    values = _torch_rows(
        arrays['dense_values'], rows, device, dtype=obs_dtype, pin_memory=pin_memory
    )
    obs[:, planes, :] = values # 普通连续plane

    # mask 与其他标量数组先按相同 rows 选取，再恢复训练接口要求的 dtype。
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
    """把 ``EncodedChunk`` 序列化、Zstd 压缩并原子发布到目标路径。

    元数据先编码成 UTF-8 JSON，并作为 ``uint8`` 数组放入 NPZ 的 ``__metadata__`` 条目；
    这样读取时不需要启用 NumPy pickle。``np.savez`` 本身不压缩，各数组和容器头形成的
    完整字节流统一交给 Zstd，避免在 ZIP 与 Zstd 两层重复压缩。

    文件先写到同目录临时路径，经 ``flush/fsync`` 后用 ``os.replace`` 原子替换目标，
    最后同步父目录项。返回压缩前后字节数和最终压缩字节的 SHA256，供 chunks 索引记录。
    本函数会同时在内存持有 NPZ 与 Zstd bytes，峰值内存高于任一单独表示。
    """
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    # 使用规范化 JSON 排版，使同一 metadata 对象产生稳定且紧凑的字节表示。
    metadata_bytes = json.dumps(
        chunk.metadata,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    # BytesIO 接收完整 NPZ；__metadata__ 使用原始字节视图，其他键各自成为一个 .npy 条目。
    payload = io.BytesIO()
    np.savez(
        payload,
        __metadata__=np.frombuffer(metadata_bytes, dtype=np.uint8),
        **chunk.arrays,
    )
    raw = payload.getvalue()
    # 外层 Zstd 是磁盘文件唯一的压缩层，level 由预计算命令纳入 cache contract。
    compressed = zstd.ZstdCompressor(level=compression_level).compress(raw)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        # 数据文件和目录项分别同步，降低进程崩溃或掉电后出现半文件/丢失重命名的概率。
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
        # replace 成功后临时路径已经消失；失败时清理仍存在的未发布文件。
        temporary.unlink(missing_ok=True)
    return {
        'compressed_bytes': len(compressed),
        'serialized_bytes': len(raw),
        'sha256': hashlib.sha256(compressed).hexdigest(),
    }


def read_chunk(filename, expected_sha256=None):
    """读取一个完整压缩 chunk，按需校验摘要，并恢复 ``EncodedChunk``。

    SHA256 针对磁盘上的 Zstd bytes，在解压前检查，因此内容损坏会在进入解析器前失败。
    随后整文件解压到内存，并用 ``allow_pickle=False`` 打开 NPZ。这里只检查 schema version；
    数组 shape、plane 分区和样本业务值依赖生产阶段验证及受摘要保护的已发布文件。
    """
    # 当前实现采用整文件读取/解压，便于后续随机排列 chunk 内全部样本，但内存占用与 chunk
    # 压缩前体积相关。
    with open(filename, 'rb') as f:
        compressed = f.read()
    actual_sha256 = hashlib.sha256(compressed).hexdigest()
    if expected_sha256 is not None and actual_sha256 != expected_sha256:
        raise RuntimeError(
            f'缓存 chunk SHA256 不一致: expected={expected_sha256}, actual={actual_sha256}'
        )
    raw = zstd.ZstdDecompressor().decompress(compressed)
    # 离开 with 前把每个数组实际取出，保证关闭内存中的 NPZ archive 后数据仍可使用。
    with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
        metadata = json.loads(archive['__metadata__'].tobytes().decode('utf-8'))
        arrays = {name: archive[name] for name in archive.files if name != '__metadata__'}
    if metadata.get('schema_version') != CACHE_SCHEMA_VERSION:
        raise RuntimeError(f'不支持的缓存 schema: {metadata.get("schema_version")}')
    return EncodedChunk(metadata=metadata, arrays=arrays)


def sample_digest(batch, row):
    """计算解码后单条六元样本的稳定摘要，供落盘前后抽样验证。

    每个字段依次加入 dtype、shape 和 C 连续原始字节，避免仅拼接数据时出现字段边界或
    不同解释类型的歧义。摘要覆盖调用方传入的 batch；常规用法传入六元训练数据，三个
    来源字段由预计算器另外核对。
    """
    digest = hashlib.sha256()
    for value in batch:
        # 标量、向量和矩阵都统一转成连续 NumPy 表示，再把结构信息与载荷纳入摘要。
        array = np.ascontiguousarray(value[row])
        digest.update(str(array.dtype).encode('ascii'))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


class PackedCacheBatchIterator:
    """从完整 manifest 确定性遍历 packed chunk，并产出固定大小训练 batch。

    随机化分成两级：先打乱 chunk 列表，再打乱当前 chunk 的行号。该策略保留顺序可复现
    性和顺序读取局部性，没有构造全体样本的巨大索引。每个 chunk 会整体解码到目标设备；
    chunk 内完整 batch 直接切片，尾部通过 ``carry`` 与下一 chunk 的头部拼接。

    ``drop_last=False`` 时整个 manifest 的最终余量也会产出。实例本身只保存配置和
    manifest；每次调用 ``iter(instance)`` 都会使用相同 seed 从头重建相同顺序。
    """

    def __init__(self, manifest_file, batch_size, device, shuffle_seed=0, drop_last=False):
        """读取发布清单并保存组批参数；chunk 文件会在实际迭代时按需打开。"""
        self.manifest_file = Path(manifest_file).resolve()
        # manifest 当前整体载入内存，随后复制其中的 chunks 列表用于洗牌。
        with open(self.manifest_file, encoding='utf-8') as f:
            self.manifest = json.load(f)
        if self.manifest.get('status') != 'complete':
            raise RuntimeError('缓存 manifest 尚未完成')
        self.batch_size = batch_size
        self.device = torch.device(device)
        self.shuffle_seed = shuffle_seed
        self.drop_last = drop_last

    def __iter__(self):
        """按固定随机顺序逐 chunk 解码，并跨边界拼成训练六元组。"""
        # 局部 RNG 不改变进程全局 random 状态。同一个 RNG 连续负责 chunk 和行号洗牌，
        # 所以两级随机操作的调用顺序也是可复现契约的一部分。
        rng = random.Random(self.shuffle_seed)
        chunks = list(self.manifest['chunks'])
        rng.shuffle(chunks)
        # carry 始终是位于目标设备的六元张量 tuple，保存上一或多个小 chunk 的未满 batch。
        carry = None
        for entry in chunks:
            # manifest 中的相对路径以 manifest 所在目录为根；每次读取均验证压缩文件摘要。
            filename = self.manifest_file.parent / entry['file']
            chunk = read_chunk(filename, entry['sha256'])
            # 行号先在 CPU 上打乱，再按该顺序一次性解码整个 chunk；后续连续切片不会改变
            # chunk 内已经确定的随机顺序。
            indices = list(range(chunk.metadata['sample_count']))
            rng.shuffle(indices)
            decoded = decode_chunk_torch(chunk, self.device, indices=np.asarray(indices))
            start = 0
            if carry is not None:
                # 优先用当前 chunk 的开头补齐旧余量，保持除最终尾批外的 batch 大小稳定。
                needed = self.batch_size - carry[0].shape[0]
                taken = tuple(value[:needed] for value in decoded)
                if taken[0].shape[0] == needed:
                    # 当前 chunk 足够补齐：产出跨 chunk batch，并从未使用的第 needed 行继续。
                    yield tuple(torch.cat((left, right), dim=0) for left, right in zip(carry, taken))
                    carry = None
                    start = needed
                else:
                    # 整个当前 chunk 仍不足以补齐：全部并入 carry，下一轮继续累积。
                    carry = tuple(torch.cat((left, right), dim=0) for left, right in zip(carry, taken))
                    continue
            # 先零拷贝切出当前 decoded 中所有完整 batch；yield 返回后由训练循环消费。
            while start + self.batch_size <= decoded[0].shape[0]:
                yield tuple(value[start:start + self.batch_size] for value in decoded)
                start += self.batch_size
            if start < decoded[0].shape[0]:
                # 剩余六个字段使用完全相同的起点，作为一个逻辑整体留到下一 chunk。
                carry = tuple(value[start:] for value in decoded)
        # 只有全缓存最后一个不完整 batch 受 drop_last 控制；chunk 中间余量从不丢弃。
        if carry is not None and not self.drop_last:
            yield carry
