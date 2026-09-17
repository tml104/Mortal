"""把原始牌谱预先转换为可直接供主策略训练使用的压缩缓存。

普通 Transformer-GRP 离线训练会在每次读取数据时重复执行以下固定计算：解压并解析
mjai 日志、推进四家状态、构造 observation 和合法动作 mask、运行冻结的
``GRP_Transformer``，以及把顺位概率换算成逐局奖励。本脚本把这些不依赖主策略参数的
工作提前完成，将最终六元样本

``obs, action, mask, steps_to_done, kyoku_reward, player_rank``

按若干源文件组成一个 chunk，交给 ``precomputed_cache.encode_chunk`` 做 plane-aware
编码和 Zstd 压缩。后续 ``train_with_precomputed_cache.py`` 只需读取 chunk、在目标设备上
恢复 dense 张量，再复用原离线训练器的单 batch 优化逻辑。

预处理过程还维护三层一致性信息：

1. ``source_files.json`` 固定本次选择的源文件及其大小和修改时间；
2. ``progress.json`` 与 ``chunks.jsonl`` 记录可恢复的 chunk 提交进度；
3. ``manifest.json`` 只在所有 chunk 的抽样记录均回读验证成功后发布。

缓存内容取决于源文件、observation 版本、名次效用、GRP 权重和参与样本生成的实现。
这些输入会写入带 SHA256 的 contract；恢复时 contract 不一致便拒绝把两个不同版本的
样本混入同一缓存。脚本只读取牌谱，不调用训练加载器的自动隔离移动逻辑。
"""

import argparse
import fcntl
import gc
import hashlib
import json
import logging
import os
import random
import sys
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np

from _bootstrap import MORTAL_DIR, REPOSITORY_ROOT
from _rename_compat import source_identity
from config import config
from dataloader_with_grp_transformer import FileDatasetsIter, is_quarantined_dataset_file
from libriichi.dataset import GameplayLoader
from precomputed_cache import (
    CACHE_SCHEMA_VERSION,
    decode_chunk_numpy,
    encode_chunk,
    read_chunk,
    sample_digest,
    sha256_file,
    write_chunk,
)
from train_mortal_offline_with_grp_transformer import load_file_list


# 连续 observation 会以 float16 保存。该阈值用于阻止量化误差超过当前训练可接受范围
# 的缓存被发布；二值 plane、动作、mask、步数、奖励和顺位仍要求精确往返。
MAX_OBS_ABS_ERROR = 5e-4


def sha256_json(value):
    """对 JSON 可表示对象计算与键顺序无关的稳定摘要。

    ``sort_keys`` 和紧凑分隔符消除字典插入顺序与排版差异，使同一语义对象在预处理初次
    运行和 ``--resume`` 进程中得到相同摘要。列表顺序仍然保留，因为源文件排列等有序
    输入可能直接影响缓存内容。
    """
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(filename, value):
    """原子写入一个完整 JSON 文件，并尽量把文件和目录项同步到磁盘。

    先写同目录临时文件并 ``fsync``，再通过 ``os.replace`` 原子替换目标，避免中断后让
    读取方看见半个 JSON。替换后同步父目录，使新文件名或替换后的目录项尽量跨异常掉电
    持久化。``finally`` 会清理由写入失败遗留的临时路径。
    """
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
            f.write('\n')
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


@contextmanager
def exclusive_lock(filename):
    """在整个预处理期间持有非阻塞排他锁，阻止并发写入同一输出目录。

    锁文件位于缓存目录旁边而不是目录内部，因此首次运行尚未创建输出目录时也能加锁。
    ``LOCK_NB`` 会让第二个进程立即失败，不会在后台等待后又意外接管同一组进度文件。
    """
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True) # 确保父目录存在
    with open(filename, 'a+b') as lock: # 打开锁文件，允许多个进程同时打开但只允许一个进程加锁
        # 通常不会向锁文件写任何内容。打开文件的主要目的是取得文件描述符lock.fileno()
        # 锁状态保存在操作系统内核中，不是通过锁文件内容表示的。
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB) # 排他非阻塞锁
        try:
            yield # 然后开始执行 run(args)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN) # 运行完了以后释放锁


def append_json_line(filename, value):
    """把一个已经完整落盘的 chunk 元数据追加到只增不改的 JSONL 日志。

    每条记录单独换行并 ``fsync``。``run`` 只有在本函数成功后才推进
    ``progress.next_chunk_id``，所以恢复时可以用两者的数量关系检测不完整的提交。
    """
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def read_json_lines(filename):
    """读取 JSONL 中全部非空记录；尚未生成日志时返回空列表。"""
    if not Path(filename).exists():
        return []
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def select_source_files(file_count, seed):
    """按固定 seed 选择指定数量的未隔离训练文件。

    ``load_file_list`` 不只是读取索引，还会按照离线训练配置核对文件数量、顺序摘要及
    glob 集合。这里使用局部 ``Random``，不会污染后续样本处理使用的全局随机状态。
    已经移动到隔离目录的逻辑文件会被跳过；其他缺失文件视为数据集意外变化并立即报错。

    返回 ``(selected, skipped_quarantined)``。前者的顺序决定 chunk 划分和 ``file_id``，
    后者只写入最终 manifest 供审计。
    """
    files = load_file_list(config['dataset'])
    random.Random(seed).shuffle(files)
    selected = []
    skipped_quarantined = []
    for filename in files:
        if is_quarantined_dataset_file(filename):
            skipped_quarantined.append(filename)
            continue
        if not Path(filename).is_file():
            raise FileNotFoundError(f'训练文件不存在且未被隔离: {filename}')
        selected.append(filename)
        if len(selected) == file_count:
            break
    if len(selected) != file_count:
        raise RuntimeError(f'可用训练文件不足: expected={file_count}, actual={len(selected)}')
    return selected, skipped_quarantined


def source_file_records(files):
    """记录源文件路径、字节数和纳秒级修改时间，供缓存 contract 固定数据版本。

    这里有意不逐个计算大型数据集所有牌谱的内容摘要，以免预处理开始前再次完整读取
    数据；代价是 contract 依赖文件系统元数据识别内容变化。
    """
    records = []
    for filename in files:
        stat = Path(filename).stat()
        records.append({
            'path': filename,
            'size': stat.st_size,
            'mtime_ns': stat.st_mtime_ns,
        })
    return records


def build_contract(files, selection_seed, files_per_chunk, compression_level):
    """构造决定缓存语义与字节布局的运行契约及其摘要。

    契约同时覆盖三类依赖：选中的数据和顺序、GRP/奖励/编码参数、参与样本生成的 Python
    与 PyO3 实现。注释或其他不改变行为的源码修改也会改变实现摘要，这是保守设计：脚本
    无法可靠判断任意源码差异是否只影响说明文字。

    返回 ``(contract, contract_sha256, source_records)``。完整 contract 会进入最终
    manifest，摘要同时写入 ``progress.json``，用于恢复兼容性判断。
    """
    pipeline_dir = Path(__file__).resolve().parent
    grp_file = Path(config['grp_transformer']['state_file']).resolve()
    records = source_file_records(files)
    contract = {
        # 这些字段决定样本数值或缓存编码方式。
        'cache_schema_version': CACHE_SCHEMA_VERSION,
        'observation_version': config['control']['version'],
        'selection_seed': selection_seed,
        'source_file_count': len(files),
        'source_files_sha256': sha256_json(records),
        'files_per_chunk': files_per_chunk,
        'compression': {'codec': 'zstd', 'level': compression_level},
        'continuous_dtype': 'float16',
        'reward_dtype': 'float64',
        'pts': config['env']['pts'],
        'grp_network': config['grp_transformer']['network'],
        'grp_file': str(grp_file),
        'grp_sha256': sha256_file(grp_file),
        # 记录实际导入的扩展文件，而非只记录 Rust 源码；这样重新编译出的不同二进制也会
        # 使契约变化。Python 文件均使用绝对定位，避免当前工作目录影响摘要对象。
        'implementation': {
            legacy_name: record['sha256']
            for name, filename in {
                'precompute_grp_transformer_cache.py': Path(__file__).resolve(),
                'precomputed_cache.py': pipeline_dir / 'precomputed_cache.py',
                'dataloader_with_grp_transformer.py': pipeline_dir / 'dataloader_with_grp_transformer.py',
                'dataloader.py': MORTAL_DIR / 'dataloader.py',
                'model.py': MORTAL_DIR / 'model.py',
                'reward_calculator.py': MORTAL_DIR / 'reward_calculator.py',
                'libriichi_extension': Path(sys.modules['libriichi'].__file__).resolve(),
            }.items()
            for legacy_name, record in [source_identity(filename, name)]
        },
    }
    return contract, sha256_json(contract), records


def create_sample_generator():
    """创建只用于预计算的样本生成器，并加载冻结的 Transformer GRP。

    本函数复用 ``FileDatasetsIter.iter_file_samples``，从而与传统 Transformer-GRP 离线
    训练共享奖励和标签语义；但不走基类 ``build_iter/load_files/populate_buffer``，所以
    这里传入的空文件列表、``file_batch_size`` 和 ``reserve_ratio`` 不参与实际遍历。

    预计算固定关闭 Oracle 与花色增强，并提取全部玩家。``GameplayLoader`` 随后由
    ``collect_file_batch`` 直接调用，因此坏文件会令预处理停止，不会触发数据加载器中会
    移动源牌谱的自动隔离流程。
    """
    dataset = FileDatasetsIter( # FileDatasetsIter
        version=config['control']['version'],
        file_list=[],
        pts=config['env']['pts'],
        file_batch_size=1,
        reserve_ratio=0,
        player_names=[],
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        grp_config=config['grp_transformer'],
    )
    # 普通 DataLoader worker 会在首次迭代时延迟初始化 GRP；预计算没有 DataLoader
    # worker，因此必须在主进程显式完成同一步骤。
    dataset.initialize_grp_model() # CPU
    dataset.loader = GameplayLoader(
        version=config['control']['version'],
        oracle=False,
        player_names=[],
        augmented=False,
    )
    return dataset


def collect_file_batch(dataset, filenames, first_file_id):
    """把一组源牌谱展开为一个待编码 chunk 的连续 NumPy 数组。

    ``filenames`` 通常包含 ``--files-per-chunk`` 个文件。Rust 返回的外层列表必须与输入
    文件严格一一对应；每个内层 ``file_games`` 包含该牌谱的玩家视角。数据集再把每个
    决策转换成六元样本，并额外返回玩家编号和视角内的决策位置。

    ``first_file_id`` 是当前文件组在固定源文件列表中的起点，因而来源字段可以稳定定位为
    ``(file_id, player_id, sample_idx)``。有样本时返回六个训练数组加一个来源字典；若这
    组文件没有产生任何合法样本，则返回 ``(None, invalid_count)``。
    """
    # Rust 侧会依据 RAYON_NUM_THREADS 并行处理这一组 gzip 文件，但 Python 侧保持输入
    # 顺序，以便 file_id 与 source_files.json 中的位置一致。
    loaded = dataset.loader.load_gz_log_files(filenames)
    if len(loaded) != len(filenames):
        raise RuntimeError(
            f'批量加载返回文件数异常: expected={len(filenames)}, actual={len(loaded)}'
        )
    # fields 与最终训练样本字段顺序完全相同；先用列表累积不同长度的牌谱，再统一堆叠。
    fields = [[] for _ in range(6)]
    source = {'file_ids': [], 'player_ids': [], 'sample_indices': []}
    invalid_before = dataset.invalid_action_samples # 记录处理前的非法样本数
    for local_file_id, (filename, file_games) in enumerate(zip(filenames, loaded, strict=True)):
        for entry, player_id, sample_idx in dataset.iter_file_samples(filename, file_games):
            for target, value in zip(fields, entry, strict=True):
                target.append(value)
            source['file_ids'].append(first_file_id + local_file_id)
            source['player_ids'].append(player_id)
            source['sample_indices'].append(sample_idx)
    invalid_count = dataset.invalid_action_samples - invalid_before
    if not fields[0]:
        return None, invalid_count
    # 显式固定未压缩阶段的数据类型。编码器会在验证值域后进一步把整型和二值字段收窄，
    # observation 中的连续值也会量化为 float16；reward 始终保留 float64。
    arrays = (
        np.stack(fields[0]).astype(np.float32, copy=False),
        np.asarray(fields[1], dtype=np.int64),
        np.stack(fields[2]).astype(np.bool_, copy=False),
        np.asarray(fields[3], dtype=np.int64),
        np.asarray(fields[4], dtype=np.float64),
        np.asarray(fields[5], dtype=np.int64),
    )
    source = {name: np.asarray(value, dtype=np.int64) for name, value in source.items()}
    return (*arrays, source), invalid_count


def build_validation_records(chunk, original_batch, source, indices, chunk_id):
    """比较原始数据与内存中编码后再解码的数据，并生成抽样验证记录。

    非 observation 字段必须完全相同；observation 的连续值允许 float16 量化误差。每条
    记录保存重构样本摘要、来源坐标和误差统计，稍后 ``validate_written_cache`` 会从磁盘
    重新读取 chunk 再次核对，从而同时覆盖编码逻辑和落盘内容。
    """
    original_obs, actions, masks, steps, rewards, ranks = original_batch
    decoded = decode_chunk_numpy(chunk, indices)
    # decoded[0] 是允许量化误差的 observation；其余字段不应发生任何有损变化。
    expected = (actions, masks, steps, rewards, ranks)
    for actual, original in zip(decoded[1:], expected, strict=True):
        np.testing.assert_array_equal(actual, original[indices])
    records = []
    for local_row, chunk_row in enumerate(indices):
        original = original_obs[chunk_row]
        reconstructed = decoded[0][local_row]
        error = np.abs(original - reconstructed)
        max_abs_error = float(error.max(initial=0))
        if max_abs_error > MAX_OBS_ABS_ERROR:
            raise RuntimeError(
                f'float16 observation 误差超限: actual={max_abs_error}, '
                f'limit={MAX_OBS_ABS_ERROR}'
            )
        record = {
            'chunk_id': chunk_id,
            'row': int(chunk_row),
            'digest': sample_digest(decoded, local_row),
            'file_id': int(source['file_ids'][chunk_row]),
            'player_id': int(source['player_ids'][chunk_row]),
            'sample_idx': int(source['sample_indices'][chunk_row]),
            'obs_max_abs_error': max_abs_error,
            'obs_mean_abs_error': float(error.mean(dtype=np.float64)),
        }
        records.append(record)
    return records


def validate_written_cache(output_dir, chunks):
    """从磁盘回读全部 chunk 的抽样记录，执行缓存发布前的最终验证。

    ``read_chunk`` 会先校验压缩文件 SHA256 并解压完整文件，随后解码器只重构指定抽样
    行。样本摘要验证六元训练数据的稳定往返，来源字段则直接从 chunk 的审计数组中读取。
    返回跨 chunk 汇总的验证样本数、最大误差和平均误差，最终写入 manifest。
    """
    validated = 0
    max_abs_error = 0.0
    mean_abs_error_sum = 0.0
    for entry in chunks:
        validation_file = output_dir / entry['validation_file']
        with open(validation_file, encoding='utf-8') as f:
            records = json.load(f)
        chunk = read_chunk(output_dir / entry['file'], entry['sha256'])
        indices = np.asarray([record['row'] for record in records], dtype=np.int64)
        decoded = decode_chunk_numpy(chunk, indices)
        for row, record in enumerate(records):
            actual = sample_digest(decoded, row)
            if actual != record['digest']:
                raise RuntimeError(
                    f'缓存样本摘要不一致: chunk={entry["id"]}, row={record["row"]}'
                )
            source_row = record['row']
            source_values = (
                int(chunk.arrays['file_ids'][source_row]),
                int(chunk.arrays['player_ids'][source_row]),
                int(chunk.arrays['sample_indices'][source_row]),
            )
            expected_source = (
                record['file_id'],
                record['player_id'],
                record['sample_idx'],
            )
            if source_values != expected_source:
                raise RuntimeError(
                    f'缓存来源索引不一致: expected={expected_source}, actual={source_values}'
                )
            max_abs_error = max(max_abs_error, record['obs_max_abs_error'])
            mean_abs_error_sum += record['obs_mean_abs_error']
            validated += 1
    return {
        'validated_samples': validated,
        'obs_max_abs_error': max_abs_error,
        'obs_mean_abs_error': mean_abs_error_sum / max(validated, 1),
    }


def parse_args():
    """解析缓存输出位置、数据选择、chunk 编码和恢复参数。"""
    parser = argparse.ArgumentParser()
    # file-count 控制实验数据规模；files-per-chunk 是缓存物理分块，不是训练 batch size。
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--file-count', type=int, default=10_000)
    parser.add_argument('--selection-seed', type=int, default=20260721)
    parser.add_argument('--files-per-chunk', type=int, default=6)
    parser.add_argument('--compression-level', type=int, default=1)
    parser.add_argument('--validation-samples-per-chunk', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def run(args):
    """生成或恢复一份缓存，并在完整验证后原子发布 manifest。

    一个非空样本 chunk 的提交顺序为：写压缩文件、写验证 JSON、追加 ``chunks.jsonl``、
    更新 ``progress.json``。只有循环全部完成且回读验证成功，才写 ``manifest.json``。
    因此消费者可以把完整 manifest 的存在与 ``status=complete`` 视为缓存发布边界。
    """
    # 尽早拒绝无意义参数，避免创建输出目录后才发现无法推进。
    if args.file_count <= 0 or args.files_per_chunk <= 0:
        raise ValueError('file-count 和 files-per-chunk 必须为正整数')
    if args.validation_samples_per_chunk <= 0:
        raise ValueError('validation-samples-per-chunk 必须为正整数')
    output_dir = Path(args.output_dir).resolve()
    # chunks 保存训练载荷；validation 保存少量审计样本；其余三个文件分别承担顺序日志、
    # 可恢复游标和最终只读入口的职责。
    chunks_dir = output_dir / 'chunks'
    validation_dir = output_dir / 'validation'
    progress_file = output_dir / 'progress.json'
    chunk_log = output_dir / 'chunks.jsonl'
    manifest_file = output_dir / 'manifest.json'

    # 无论初次运行还是恢复，都重新选择文件并构造当前契约。恢复只有在当前环境能再次得到
    # 完全相同 contract 时才允许继续。
    files, skipped_quarantined = select_source_files(args.file_count, args.selection_seed)
    contract, contract_sha256, source_records = build_contract(
        files,
        args.selection_seed,
        args.files_per_chunk,
        args.compression_level,
    )
    source_file = output_dir / 'source_files.json'

    if args.resume: # 继续
        # progress 是恢复的权威游标；它不保存模型对象，因为 GRP 是冻结的，可由 contract
        # 指定的 checkpoint 在新进程中重新加载。
        if not progress_file.is_file():
            raise FileNotFoundError(f'缺少可恢复进度: {progress_file}')
        with open(progress_file, encoding='utf-8') as f:
            progress = json.load(f)
        if progress['contract_sha256'] != contract_sha256:
            raise RuntimeError('预处理输入或实现已变化，拒绝恢复不兼容缓存')
    else: # 初始化
        # 初次运行不覆盖任何已有内容。调用方若要新建不同契约的缓存，必须使用新目录。
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'输出目录非空，拒绝覆盖: {output_dir}')
        chunks_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(source_file, source_records)
        progress = {
            'status': 'running',
            'contract_sha256': contract_sha256,
            'next_file_index': 0, # 后续程序会从此开始继续处理文件
            'next_chunk_id': 0,
            'total_samples': 0,
            'invalid_action_samples': 0,
            'started_at': datetime.now().astimezone().isoformat(),
        }
        atomic_write_json(progress_file, progress)

    # JSONL 每行代表一个已提交的非空 chunk。数量与 next_chunk_id 不一致意味着上次进程
    # 中断在两个提交步骤之间，当前实现不会猜测哪一侧应被回滚或补写。
    existing_chunks = read_json_lines(chunk_log)
    if len(existing_chunks) != progress['next_chunk_id']:
        raise RuntimeError('chunks.jsonl 与 progress.json 的 chunk 数不一致')
    # 冻结 GRP 只加载一次，并在后续所有文件组之间复用。
    dataset = create_sample_generator()
    started = time.perf_counter()

    for file_start in range(progress['next_file_index'], len(files), args.files_per_chunk):
        # 文件组边界同时是 chunk 边界，也是 progress 可恢复的最小粒度。
        filenames = files[file_start:file_start + args.files_per_chunk]
        chunk_id = progress['next_chunk_id']
        collected, invalid_count = collect_file_batch(dataset, filenames, file_start)
        if collected is None:
            # 空文件组不产生 chunk id 或 JSONL 记录，但仍推进源文件游标，避免恢复后无限重试。
            progress['next_file_index'] = file_start + len(filenames)
            progress['invalid_action_samples'] += invalid_count
            atomic_write_json(progress_file, progress)
            continue
        obs, actions, masks, steps, rewards, ranks, source = collected
        # encode_chunk 负责形状/值域检查、observation plane 分类、bit-pack 和 float16 量化。
        chunk = encode_chunk(obs, actions, masks, steps, rewards, ranks, source)
        relative_chunk = Path('chunks') / f'chunk_{chunk_id:06d}.npz.zst'
        write_stats = write_chunk(
            chunk,
            output_dir / relative_chunk,
            compression_level=args.compression_level,
        )

        # 每个 chunk 使用独立且可复现的随机流。排序只改变验证读取顺序，不改变抽中的行。
        validation_count = min(args.validation_samples_per_chunk, len(obs))
        validation_rng = random.Random(args.selection_seed + chunk_id)
        validation_indices = sorted(validation_rng.sample(range(len(obs)), validation_count))
        validation_records = build_validation_records(
            chunk,
            (obs, actions, masks, steps, rewards, ranks),
            source,
            validation_indices,
            chunk_id,
        )
        relative_validation = Path('validation') / f'chunk_{chunk_id:06d}.json'
        atomic_write_json(output_dir / relative_validation, validation_records)

        elapsed = time.perf_counter() - started
        # entry 既是训练读取所需的 chunk 索引，也是估算压缩率、量化误差和坏样本数量的
        # 审计记录。sample_start 表示生成顺序中的逻辑偏移，不代表训练时的打乱顺序。
        entry = {
            'id': chunk_id,
            'file': str(relative_chunk),
            'validation_file': str(relative_validation),
            'file_start': file_start,
            'file_count': len(filenames),
            'sample_start': progress['total_samples'],
            'sample_count': len(obs),
            'invalid_action_samples': invalid_count,
            'plane_counts': chunk.metadata['plane_counts'],
            'dense_float32_bytes': chunk.metadata['dense_float32_bytes'],
            'packed_array_bytes': chunk.metadata['packed_array_bytes'],
            'serialized_bytes': write_stats['serialized_bytes'],
            'compressed_bytes': write_stats['compressed_bytes'],
            'sha256': write_stats['sha256'],
            'quantized_value_count': chunk.metadata['quantized_value_count'],
            'quantization_abs_sum': chunk.metadata['quantization_abs_sum'],
            'quantization_abs_max': chunk.metadata['quantization_abs_max'],
        }
        # 先提交 chunk 日志，再推进 progress。恢复要求二者严格一致，因此异常窗口不会被
        # 静默跨过；操作者需要保留现场并明确处理，而不是可能重复或漏掉样本。
        append_json_line(chunk_log, entry)
        progress.update({
            'next_file_index': file_start + len(filenames),
            'next_chunk_id': chunk_id + 1,
            'total_samples': progress['total_samples'] + len(obs),
            'invalid_action_samples': progress['invalid_action_samples'] + invalid_count,
            'updated_at': datetime.now().astimezone().isoformat(),
        })
        atomic_write_json(progress_file, progress)
        samples_per_second = (
            progress['total_samples'] - sum(item['sample_count'] for item in existing_chunks)
        ) / elapsed
        logging.info(
            'chunk=%s files=%s/%s samples=%s compressed=%.2fMiB samples/s=%.1f',
            chunk_id,
            progress['next_file_index'],
            len(files),
            f'{progress["total_samples"]:,}',
            write_stats['compressed_bytes'] / 1024 ** 2,
            samples_per_second,
        )
        # 一个 chunk 完成后主动释放巨大的 dense observation 和编码数组，避免完整数据集
        # 预处理过程中常驻内存随 chunk 数增长。gc 主要回收 Python 容器及其引用环。
        del collected, obs, actions, masks, steps, rewards, ranks, source, chunk
        gc.collect()

    # 生成结束后不直接信任内存结果，而是依据 JSONL 重新打开磁盘文件和验证记录。只有
    # 这里成功返回，下面才会出现可供训练器消费的 manifest.json。
    chunks = read_json_lines(chunk_log)
    validation = validate_written_cache(output_dir, chunks)
    # 压缩率以原始 dense float32 observation 字节数为分子，以包含全部字段的最终压缩
    # chunk 字节数为分母；它衡量的是实际磁盘格式相对于主要未压缩载荷的节省程度。
    dense_bytes = sum(entry['dense_float32_bytes'] for entry in chunks)
    compressed_bytes = sum(entry['compressed_bytes'] for entry in chunks)
    quantized_count = sum(entry['quantized_value_count'] for entry in chunks)
    quantized_abs_sum = sum(entry['quantization_abs_sum'] for entry in chunks)
    manifest = {
        'status': 'complete',
        'schema_version': CACHE_SCHEMA_VERSION,
        'created_at': datetime.now().astimezone().isoformat(),
        'contract': contract,
        'contract_sha256': contract_sha256,
        'source_files_file': source_file.name,
        'selected_file_count': len(files),
        'skipped_quarantined_before_selection': skipped_quarantined,
        'sample_count': progress['total_samples'],
        'invalid_action_samples': progress['invalid_action_samples'],
        'dense_float32_bytes': dense_bytes,
        'compressed_bytes': compressed_bytes,
        'compression_ratio': dense_bytes / compressed_bytes,
        'bytes_per_sample': compressed_bytes / progress['total_samples'],
        'quantization_abs_max': max(
            (entry['quantization_abs_max'] for entry in chunks),
            default=0,
        ),
        'quantization_abs_mean': quantized_abs_sum / max(quantized_count, 1),
        'validation': validation,
        'chunks': chunks,
    }
    # manifest 是缓存的发布入口，必须先于 progress 的 complete 状态落盘。读取器还会检查
    # manifest.status，因此未走到此处的中断任务不会被误当作可训练缓存。
    atomic_write_json(manifest_file, manifest)
    progress['status'] = 'complete'
    progress['completed_at'] = manifest['created_at']
    atomic_write_json(progress_file, progress)
    logging.info(
        'PRECOMPUTE_COMPLETED files=%s samples=%s compressed=%.3fGiB ratio=%.2fx validated=%s',
        len(files),
        f'{progress["total_samples"]:,}',
        compressed_bytes / 1024 ** 3,
        manifest['compression_ratio'],
        validation['validated_samples'],
    )


def main():
    """配置日志，在输出目录同级取得运行锁，然后进入预计算主流程。"""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)8s %(filename)s:%(lineno)d %(message)s',
    )
    output_dir = Path(args.output_dir).resolve()
    # 锁文件不放进 output_dir，避免它使“初次运行要求输出目录为空”的检查必然失败。
    with exclusive_lock(output_dir.with_name(f'{output_dir.name}.lock')):
        run(args)


if __name__ == '__main__':
    main()
