import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import torch

import _bootstrap
from config import config
from precomputed_cache import (
    PackedCacheBatchIterator,
    decode_chunk_numpy,
    decode_chunk_torch,
    read_chunk,
)
from train_mortal_offline_with_grp_transformer import build_models_and_optimizer, train_batch


ESTIMATED_FULL_DATASET_SAMPLES = 838_885_333


def load_manifest(filename, max_chunks=None):
    filename = Path(filename).resolve()
    with open(filename, encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('status') != 'complete':
        raise RuntimeError('缓存尚未完成')
    chunks = manifest['chunks'][:max_chunks]
    return filename, manifest, chunks


def benchmark_cpu_decode(manifest_file, chunks):
    """测量压缩读取和 CPU dense 重构吞吐。"""
    started = time.perf_counter()
    sample_count = 0
    for entry in chunks:
        chunk = read_chunk(manifest_file.parent / entry['file'], entry['sha256'])
        batch = decode_chunk_numpy(chunk)
        rows = np.arange(len(batch[1]))
        if not batch[2][rows, batch[1]].all():
            raise RuntimeError('CPU 解码结果含非法 action')
        sample_count += len(batch[0])
        del batch, chunk
    elapsed = time.perf_counter() - started
    return {
        'samples': sample_count,
        'seconds': elapsed,
        'samples_per_second': sample_count / elapsed,
    }


def benchmark_gpu_supply(manifest_file, chunks, device):
    """测量压缩读取、传输和 GPU observation 解包吞吐。"""
    torch.cuda.set_device(device)
    torch.empty(0, device=device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    sample_count = 0
    for entry in chunks:
        chunk = read_chunk(manifest_file.parent / entry['file'], entry['sha256'])
        batch = decode_chunk_torch(chunk, device)
        sample_count += len(batch[0])
        del batch, chunk
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    return {
        'samples': sample_count,
        'seconds': elapsed,
        'samples_per_second': sample_count / elapsed,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
    }


def load_training_state(checkpoint, device):
    """在独立进程内恢复模型与优化器，不写回 checkpoint。"""
    state = torch.load(checkpoint, weights_only=True, map_location='cpu')
    components = build_models_and_optimizer(
        device,
        resume=False,
        grp_sha256='',
        file_list_sha256='',
        dataset_manifest_sha256='',
        training_contract={},
        training_contract_sha256='',
    )
    mortal, dqn, aux_net, optimizer, scheduler, scaler, _ = components
    mortal.load_state_dict(state['mortal'], strict=True)
    dqn.load_state_dict(state['current_dqn'], strict=True)
    aux_net.load_state_dict(state['aux_net'], strict=True)
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    return mortal, dqn, aux_net, optimizer, scheduler, scaler


def benchmark_cached_training(
    manifest_file,
    checkpoint,
    device,
    batch_size,
    warmup_steps,
    training_steps,
):
    """测量缓存供数加完整前向、反向和优化器更新的端到端吞吐。"""
    torch.cuda.set_device(device)
    components = load_training_state(checkpoint, device)
    mortal, dqn, aux_net, optimizer, scheduler, scaler = components
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    completed = 0
    timed_steps = 0
    timed_started = None
    epoch = 0
    while timed_steps < training_steps:
        iterator = PackedCacheBatchIterator(
            manifest_file,
            batch_size=batch_size,
            device=device,
            shuffle_seed=20260730 + epoch,
            drop_last=True,
        )
        for batch in iterator:
            if batch[0].shape[0] != batch_size:
                continue
            train_batch(
                batch,
                mortal=mortal,
                dqn=dqn,
                aux_net=aux_net,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                device=device,
            )
            completed += 1
            if completed == warmup_steps:
                torch.cuda.synchronize()
                timed_started = time.perf_counter()
            elif completed > warmup_steps:
                timed_steps += 1
                if timed_steps == training_steps:
                    break
        epoch += 1
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - timed_started
    return {
        'warmup_steps': warmup_steps,
        'training_steps': training_steps,
        'samples': training_steps * batch_size,
        'seconds': elapsed,
        'samples_per_second': training_steps * batch_size / elapsed,
        'milliseconds_per_step': elapsed / training_steps * 1000,
        'peak_allocated_gib': torch.cuda.max_memory_allocated() / 1024 ** 3,
        'peak_reserved_gib': torch.cuda.max_memory_reserved() / 1024 ** 3,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--batch-size', type=int, default=1024)
    parser.add_argument('--warmup-steps', type=int, default=5)
    parser.add_argument('--training-steps', type=int, default=250)
    parser.add_argument('--max-chunks', type=int)
    parser.add_argument('--baseline-samples-per-second', type=float, default=3260.0)
    parser.add_argument('--output')
    parser.add_argument('--skip-cpu', action='store_true')
    parser.add_argument('--skip-gpu', action='store_true')
    parser.add_argument('--skip-training', action='store_true')
    return parser.parse_args()


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO)
    manifest_file, manifest, chunks = load_manifest(args.manifest, args.max_chunks)
    result = {
        'manifest': str(manifest_file),
        'chunk_count': len(chunks),
        'cache_sample_count': manifest['sample_count'],
        'bytes_per_sample': manifest['bytes_per_sample'],
        'projected_full_cache_gib': (
            manifest['bytes_per_sample'] * ESTIMATED_FULL_DATASET_SAMPLES / 1024 ** 3
        ),
        'baseline_samples_per_second': args.baseline_samples_per_second,
    }
    if not args.skip_cpu:
        result['cpu_decode'] = benchmark_cpu_decode(manifest_file, chunks)
    device = torch.device(args.device)
    if not args.skip_gpu or not args.skip_training:
        if device.type != 'cuda' or not torch.cuda.is_available():
            raise RuntimeError('GPU 解包或训练基准要求可用 CUDA')
    if not args.skip_gpu:
        result['gpu_supply'] = benchmark_gpu_supply(manifest_file, chunks, device)
    if not args.skip_training:
        if not args.checkpoint:
            raise ValueError('训练基准需要 --checkpoint')
        result['cached_training'] = benchmark_cached_training(
            manifest_file,
            args.checkpoint,
            device,
            args.batch_size,
            args.warmup_steps,
            args.training_steps,
        )
        result['training_speedup'] = (
            result['cached_training']['samples_per_second']
            / args.baseline_samples_per_second
        )
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        output = Path(args.output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, 'w', encoding='utf-8') as f:
            f.write(rendered)
            f.write('\n')


if __name__ == '__main__':
    main()
