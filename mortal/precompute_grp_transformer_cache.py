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

from config import config
from dataloader_grp_transformer import FileDatasetsIter, is_quarantined_dataset_file
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
from train_offline_grp_transformer import load_file_list


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
MAX_OBS_ABS_ERROR = 5e-4


def sha256_json(value):
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


def atomic_write_json(filename, value):
    """原子写入 JSON 并同步目录项。"""
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
    """阻止两个预处理进程写入同一缓存。"""
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'a+b') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def append_json_line(filename, value):
    """追加并同步一个 chunk 记录。"""
    with open(filename, 'a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, sort_keys=True))
        f.write('\n')
        f.flush()
        os.fsync(f.fileno())


def read_json_lines(filename):
    if not Path(filename).exists():
        return []
    with open(filename, encoding='utf-8') as f:
        return [json.loads(line) for line in f if line.strip()]


def select_source_files(file_count, seed):
    """按固定 seed 选择指定数量的未隔离训练文件。"""
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
    """锁定影响缓存内容的输入、模型和实现指纹。"""
    module_dir = Path(__file__).resolve().parent
    grp_file = Path(config['grp_transformer']['state_file']).resolve()
    records = source_file_records(files)
    contract = {
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
        'implementation': {
            name: sha256_file(filename)
            for name, filename in {
                'precompute_grp_transformer_cache.py': Path(__file__).resolve(),
                'precomputed_cache.py': module_dir / 'precomputed_cache.py',
                'dataloader_grp_transformer.py': module_dir / 'dataloader_grp_transformer.py',
                'dataloader.py': module_dir / 'dataloader.py',
                'model.py': module_dir / 'model.py',
                'reward_calculator.py': module_dir / 'reward_calculator.py',
                'libriichi_extension': Path(sys.modules['libriichi'].__file__).resolve(),
            }.items()
        },
    }
    return contract, sha256_json(contract), records


def create_sample_generator():
    """加载冻结 GRP 和 Rust GameplayLoader，但不启用自动隔离移动。"""
    dataset = FileDatasetsIter(
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
    dataset.initialize_grp_model()
    dataset.loader = GameplayLoader(
        version=config['control']['version'],
        oracle=False,
        player_names=[],
        augmented=False,
    )
    return dataset


def collect_file_batch(dataset, filenames, first_file_id):
    """将一组牌谱转换为连续 numpy 数组及来源索引。"""
    loaded = dataset.loader.load_gz_log_files(filenames)
    if len(loaded) != len(filenames):
        raise RuntimeError(
            f'批量加载返回文件数异常: expected={len(filenames)}, actual={len(loaded)}'
        )
    fields = [[] for _ in range(6)]
    source = {'file_ids': [], 'player_ids': [], 'sample_indices': []}
    invalid_before = dataset.invalid_action_samples
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
    """记录原始样本量化后的摘要和量化误差。"""
    original_obs, actions, masks, steps, rewards, ranks = original_batch
    decoded = decode_chunk_numpy(chunk, indices)
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
    """重新读取所有验证记录，并核对文件摘要、样本摘要和来源字段。"""
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--file-count', type=int, default=10_000)
    parser.add_argument('--selection-seed', type=int, default=20260721)
    parser.add_argument('--files-per-chunk', type=int, default=6)
    parser.add_argument('--compression-level', type=int, default=1)
    parser.add_argument('--validation-samples-per-chunk', type=int, default=6)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def run(args):
    if args.file_count <= 0 or args.files_per_chunk <= 0:
        raise ValueError('file-count 和 files-per-chunk 必须为正整数')
    if args.validation_samples_per_chunk <= 0:
        raise ValueError('validation-samples-per-chunk 必须为正整数')
    output_dir = Path(args.output_dir).resolve()
    chunks_dir = output_dir / 'chunks'
    validation_dir = output_dir / 'validation'
    progress_file = output_dir / 'progress.json'
    chunk_log = output_dir / 'chunks.jsonl'
    manifest_file = output_dir / 'manifest.json'

    files, skipped_quarantined = select_source_files(args.file_count, args.selection_seed)
    contract, contract_sha256, source_records = build_contract(
        files,
        args.selection_seed,
        args.files_per_chunk,
        args.compression_level,
    )
    source_file = output_dir / 'source_files.json'

    if args.resume:
        if not progress_file.is_file():
            raise FileNotFoundError(f'缺少可恢复进度: {progress_file}')
        with open(progress_file, encoding='utf-8') as f:
            progress = json.load(f)
        if progress['contract_sha256'] != contract_sha256:
            raise RuntimeError('预处理输入或实现已变化，拒绝恢复不兼容缓存')
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'输出目录非空，拒绝覆盖: {output_dir}')
        chunks_dir.mkdir(parents=True, exist_ok=True)
        validation_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(source_file, source_records)
        progress = {
            'status': 'running',
            'contract_sha256': contract_sha256,
            'next_file_index': 0,
            'next_chunk_id': 0,
            'total_samples': 0,
            'invalid_action_samples': 0,
            'started_at': datetime.now().astimezone().isoformat(),
        }
        atomic_write_json(progress_file, progress)

    existing_chunks = read_json_lines(chunk_log)
    if len(existing_chunks) != progress['next_chunk_id']:
        raise RuntimeError('chunks.jsonl 与 progress.json 的 chunk 数不一致')
    dataset = create_sample_generator()
    started = time.perf_counter()

    for file_start in range(progress['next_file_index'], len(files), args.files_per_chunk):
        filenames = files[file_start:file_start + args.files_per_chunk]
        chunk_id = progress['next_chunk_id']
        collected, invalid_count = collect_file_batch(dataset, filenames, file_start)
        if collected is None:
            progress['next_file_index'] = file_start + len(filenames)
            progress['invalid_action_samples'] += invalid_count
            atomic_write_json(progress_file, progress)
            continue
        obs, actions, masks, steps, rewards, ranks, source = collected
        chunk = encode_chunk(obs, actions, masks, steps, rewards, ranks, source)
        relative_chunk = Path('chunks') / f'chunk_{chunk_id:06d}.npz.zst'
        write_stats = write_chunk(
            chunk,
            output_dir / relative_chunk,
            compression_level=args.compression_level,
        )

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
        del collected, obs, actions, masks, steps, rewards, ranks, source, chunk
        gc.collect()

    chunks = read_json_lines(chunk_log)
    validation = validate_written_cache(output_dir, chunks)
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
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)8s %(filename)s:%(lineno)d %(message)s',
    )
    output_dir = Path(args.output_dir).resolve()
    with exclusive_lock(output_dir.with_name(f'{output_dir.name}.lock')):
        run(args)


if __name__ == '__main__':
    main()
