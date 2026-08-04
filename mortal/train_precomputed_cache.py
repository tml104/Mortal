import argparse
import fcntl
import json
import logging
import math
import os
import random
import shutil
import time
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from config import config
from precomputed_cache import PackedCacheBatchIterator
from train_offline_grp_transformer import (
    METRIC_NAMES,
    atomic_torch_save,
    build_models_and_optimizer,
    sha256_file,
    sha256_json,
    train_batch,
)


CHECKPOINT_SCHEMA = 'precomputed-cache-training-v1'


def atomic_write_json(filename, value):
    """原子写入训练元数据，避免中断后留下半个 JSON。"""
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
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def exclusive_lock(filename):
    """防止两个进程同时写入同一个实验目录。"""
    filename = Path(filename).resolve()
    filename.parent.mkdir(parents=True, exist_ok=True)
    with open(filename, 'a+', encoding='utf-8') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'已有缓存训练实例持有运行锁: {filename}') from exc
        try:
            yield
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)


def configure_logging(output_dir):
    """同时记录便于 tail 的训练日志和终端输出。"""
    formatter = logging.Formatter(
        '%(asctime)s %(levelname)8s %(filename)s:%(lineno)d %(message)s'
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers.clear()
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / 'train.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    root.addHandler(stream)
    root.addHandler(file_handler)


def load_manifest(filename):
    """读取并检查只允许训练完整缓存。"""
    filename = Path(filename).resolve()
    with open(filename, encoding='utf-8') as f:
        manifest = json.load(f)
    if manifest.get('status') != 'complete':
        raise RuntimeError('缓存 manifest 尚未完成')
    sample_count = manifest.get('sample_count', 0)
    if sample_count <= 0 or not manifest.get('chunks'):
        raise RuntimeError('缓存 manifest 不含训练样本')
    if sum(item['sample_count'] for item in manifest['chunks']) != sample_count:
        raise RuntimeError('manifest 的 chunk 样本总数不一致')
    return filename, manifest


def build_run_contract(args, manifest_file, manifest, initial_checkpoint):
    """锁定一次训练的数据、初始权重和影响训练语义的参数。"""
    source_config = os.environ.get('MORTAL_CFG')
    contract = {
        'manifest': str(manifest_file),
        'manifest_sha256': sha256_file(manifest_file),
        'cache_contract_sha256': manifest['contract_sha256'],
        'cache_sample_count': manifest['sample_count'],
        'cache_chunk_count': len(manifest['chunks']),
        'initial_checkpoint': str(initial_checkpoint),
        'initial_checkpoint_sha256': sha256_file(initial_checkpoint),
        'source_config': str(Path(source_config).resolve()) if source_config else None,
        'config_sha256': sha256_json(config),
        'batch_size': args.batch_size,
        'shuffle_seed': args.shuffle_seed,
        'device': str(args.device),
        'trainer_sha256': sha256_file(Path(__file__).resolve()),
    }
    return contract, sha256_json(contract)


def create_models(checkpoint, device):
    """创建当前配置的网络，并恢复指定 checkpoint 的全部优化状态。"""
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
    for model in (mortal, dqn, aux_net):
        model.train()
    return state, (mortal, dqn, aux_net, optimizer, scheduler, scaler)


def checkpoint_payload(
    *,
    components,
    source_state,
    run_contract,
    run_contract_sha256,
    progress,
):
    """构造不会被原始文件偏移训练器误恢复的专用 checkpoint。"""
    mortal, dqn, aux_net, optimizer, scheduler, scaler = components
    return {
        'schema_version': CHECKPOINT_SCHEMA,
        'checkpoint_type': 'precomputed_cache_epoch',
        'mortal': mortal.state_dict(),
        'current_dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'steps': source_state.get('steps', 0) + progress['batches_completed'],
        'samples_seen': source_state.get('samples_seen', 0) + progress['samples_completed'],
        'best_perf': source_state.get('best_perf'),
        'base_checkpoint_schema_version': source_state.get('schema_version'),
        'base_checkpoint_steps': source_state.get('steps'),
        'base_checkpoint_samples_seen': source_state.get('samples_seen'),
        'run_contract': run_contract,
        'run_contract_sha256': run_contract_sha256,
        'epoch_progress': progress,
        'timestamp': datetime.now().timestamp(),
        'trainer': 'train_precomputed_cache.py',
        'config': config,
    }


def save_checkpoint(
    filename,
    *,
    components,
    source_state,
    run_contract,
    run_contract_sha256,
    progress,
):
    state = checkpoint_payload(
        components=components,
        source_state=source_state,
        run_contract=run_contract,
        run_contract_sha256=run_contract_sha256,
        progress=progress,
    )
    atomic_torch_save(state, filename)


def render_metrics(metric_sum, sample_count):
    values = (metric_sum / max(sample_count, 1)).tolist()
    return dict(zip(METRIC_NAMES, values, strict=True))


def update_timing(progress, run_started, wall_base, checkpoint_seconds):
    wall_seconds = wall_base + time.perf_counter() - run_started
    progress['wall_seconds'] = wall_seconds
    progress['checkpoint_seconds'] = checkpoint_seconds
    progress['active_seconds'] = wall_seconds - checkpoint_seconds
    progress['updated_at'] = datetime.now().astimezone().isoformat()


def train_one_epoch(args, output_dir, manifest_file, manifest, run_contract, contract_sha256):
    device = torch.device(args.device)
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('本次缓存训练要求可用 CUDA 设备')
    torch.cuda.set_device(device)
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    random.seed(args.shuffle_seed)
    np.random.seed(args.shuffle_seed % 2 ** 32)
    torch.manual_seed(args.shuffle_seed)
    torch.cuda.manual_seed_all(args.shuffle_seed)

    checkpoint_file = output_dir / 'latest.pth'
    setup_started = time.perf_counter()
    if args.resume:
        saved = torch.load(checkpoint_file, weights_only=True, map_location='cpu')
        if saved.get('schema_version') != CHECKPOINT_SCHEMA:
            raise RuntimeError('输出 checkpoint 不是缓存训练格式')
        if saved.get('run_contract_sha256') != contract_sha256:
            raise RuntimeError('训练输入或实现已变化，拒绝恢复')
        source_state, components = create_models(args.initial_checkpoint, device)
        mortal, dqn, aux_net, optimizer, scheduler, scaler = components
        mortal.load_state_dict(saved['mortal'], strict=True)
        dqn.load_state_dict(saved['current_dqn'], strict=True)
        aux_net.load_state_dict(saved['aux_net'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        scaler.load_state_dict(saved['scaler'])
        progress = dict(saved['epoch_progress'])
        logging.info(
            'resumed cache epoch: batches=%s samples=%s',
            progress['batches_completed'],
            progress['samples_completed'],
        )
    else:
        source_state, components = create_models(args.initial_checkpoint, device)
        progress = {
            'status': 'running',
            'epoch': 1,
            'batches_completed': 0,
            'samples_completed': 0,
            'total_batches': math.ceil(manifest['sample_count'] / args.batch_size),
            'total_samples': manifest['sample_count'],
            'metric_sums': [0.0] * len(METRIC_NAMES),
            'wall_seconds': 0.0,
            'active_seconds': 0.0,
            'checkpoint_seconds': 0.0,
            'started_at': datetime.now().astimezone().isoformat(),
        }

    logging.info('model setup seconds: %.3f', time.perf_counter() - setup_started)
    logging.info(
        'CACHE_EPOCH_START samples=%s batches=%s batch_size=%s chunks=%s seed=%s '
        'base_steps=%s base_samples=%s',
        f'{manifest["sample_count"]:,}',
        f'{progress["total_batches"]:,}',
        args.batch_size,
        f'{len(manifest["chunks"]):,}',
        args.shuffle_seed,
        source_state.get('steps'),
        source_state.get('samples_seen'),
    )

    writer = SummaryWriter(output_dir / 'tensorboard', purge_step=progress['batches_completed'])
    cumulative_metrics = torch.tensor(progress['metric_sums'], dtype=torch.float64, device=device)
    interval_metrics = torch.zeros_like(cumulative_metrics)
    interval_samples = 0
    interval_started = time.perf_counter()
    wall_base = progress['wall_seconds']
    checkpoint_seconds = progress['checkpoint_seconds']
    run_started = time.perf_counter()
    skipped_samples = 0

    iterator = PackedCacheBatchIterator(
        manifest_file,
        batch_size=args.batch_size,
        device=device,
        shuffle_seed=args.shuffle_seed,
        drop_last=False,
    )

    try:
        for batch_index, batch in enumerate(iterator):
            if batch_index < progress['batches_completed']:
                skipped_samples += batch[0].shape[0]
                continue
            if batch_index == progress['batches_completed'] and skipped_samples:
                if skipped_samples != progress['samples_completed']:
                    raise RuntimeError('恢复时跳过的样本数与 checkpoint 不一致')

            batch_size, metric_tensor = train_batch(
                batch,
                mortal=components[0],
                dqn=components[1],
                aux_net=components[2],
                optimizer=components[3],
                scheduler=components[4],
                scaler=components[5],
                device=device,
            )
            weighted_metrics = metric_tensor.to(torch.float64) * batch_size
            cumulative_metrics.add_(weighted_metrics)
            interval_metrics.add_(weighted_metrics)
            interval_samples += batch_size
            progress['batches_completed'] += 1
            progress['samples_completed'] += batch_size

            should_log = (
                progress['batches_completed'] % args.log_every_steps == 0
                or progress['batches_completed'] == progress['total_batches']
            )
            if should_log:
                torch.cuda.synchronize()
                update_timing(
                    progress,
                    run_started,
                    wall_base,
                    checkpoint_seconds,
                )
                progress['metric_sums'] = cumulative_metrics.tolist()
                interval_elapsed = time.perf_counter() - interval_started
                metrics = render_metrics(interval_metrics, interval_samples)
                active_throughput = progress['samples_completed'] / max(
                    progress['active_seconds'], 1e-9
                )
                logging.info(
                    'CACHE_TRAIN steps=%s/%s samples=%s/%s loss=%.6f '
                    'interval_samples/s=%.1f active_samples/s=%.1f wall_samples/s=%.1f '
                    'lr=%.8g gpu_allocated=%.3fGiB gpu_reserved=%.3fGiB',
                    f'{progress["batches_completed"]:,}',
                    f'{progress["total_batches"]:,}',
                    f'{progress["samples_completed"]:,}',
                    f'{progress["total_samples"]:,}',
                    metrics['loss'],
                    interval_samples / interval_elapsed,
                    active_throughput,
                    progress['samples_completed'] / max(progress['wall_seconds'], 1e-9),
                    components[4].get_last_lr()[0],
                    torch.cuda.memory_allocated() / 1024 ** 3,
                    torch.cuda.memory_reserved() / 1024 ** 3,
                )
                global_step = source_state.get('steps', 0) + progress['batches_completed']
                for name, value in metrics.items():
                    writer.add_scalar(f'train/{name}', value, global_step)
                writer.add_scalar(
                    'throughput/interval_samples_per_second',
                    interval_samples / interval_elapsed,
                    global_step,
                )
                writer.add_scalar('throughput/active_samples_per_second', active_throughput, global_step)
                writer.add_scalar('param/lr', components[4].get_last_lr()[0], global_step)
                writer.flush()
                interval_metrics.zero_()
                interval_samples = 0
                interval_started = time.perf_counter()

            should_checkpoint = (
                args.checkpoint_every_steps > 0
                and progress['batches_completed'] % args.checkpoint_every_steps == 0
                and progress['batches_completed'] < progress['total_batches']
            )
            if should_checkpoint:
                torch.cuda.synchronize()
                update_timing(
                    progress,
                    run_started,
                    wall_base,
                    checkpoint_seconds,
                )
                progress['metric_sums'] = cumulative_metrics.tolist()
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    checkpoint_file,
                    components=components,
                    source_state=source_state,
                    run_contract=run_contract,
                    run_contract_sha256=contract_sha256,
                    progress=progress,
                )
                elapsed = time.perf_counter() - checkpoint_started
                checkpoint_seconds += elapsed
                logging.info(
                    'cache checkpoint saved: steps=%s samples=%s seconds=%.3f',
                    progress['batches_completed'],
                    progress['samples_completed'],
                    elapsed,
                )

        torch.cuda.synchronize()
        update_timing(progress, run_started, wall_base, checkpoint_seconds)
        progress['metric_sums'] = cumulative_metrics.tolist()
        if progress['batches_completed'] != progress['total_batches']:
            raise RuntimeError('缓存 epoch 的 batch 数不完整')
        if progress['samples_completed'] != progress['total_samples']:
            raise RuntimeError('缓存 epoch 的样本数不完整')
        progress['status'] = 'complete'
        progress['completed_at'] = datetime.now().astimezone().isoformat()
        final_metrics = render_metrics(cumulative_metrics, progress['samples_completed'])

        final_save_started = time.perf_counter()
        save_checkpoint(
            checkpoint_file,
            components=components,
            source_state=source_state,
            run_contract=run_contract,
            run_contract_sha256=contract_sha256,
            progress=progress,
        )
        final_save_seconds = time.perf_counter() - final_save_started
        checkpoint_sha256 = sha256_file(checkpoint_file)
        completion = {
            'status': 'complete',
            'epoch': 1,
            'samples': progress['samples_completed'],
            'batches': progress['batches_completed'],
            'batch_size': args.batch_size,
            'wall_seconds': progress['wall_seconds'],
            'active_seconds': progress['active_seconds'],
            'checkpoint_seconds_during_epoch': progress['checkpoint_seconds'],
            'final_checkpoint_save_seconds': final_save_seconds,
            'wall_samples_per_second': progress['samples_completed'] / progress['wall_seconds'],
            'active_samples_per_second': progress['samples_completed'] / progress['active_seconds'],
            'metrics': final_metrics,
            'checkpoint': str(checkpoint_file),
            'checkpoint_sha256': checkpoint_sha256,
            'run_contract': run_contract,
            'run_contract_sha256': contract_sha256,
            'completed_at': progress['completed_at'],
        }
        atomic_write_json(output_dir / 'epoch_1.completed.json', completion)
        logging.info(
            'CACHE_EPOCH_COMPLETED samples=%s batches=%s wall_seconds=%.3f '
            'active_seconds=%.3f wall_samples/s=%.1f active_samples/s=%.1f '
            'loss=%.6f checkpoint_sha256=%s',
            f'{progress["samples_completed"]:,}',
            f'{progress["batches_completed"]:,}',
            progress['wall_seconds'],
            progress['active_seconds'],
            completion['wall_samples_per_second'],
            completion['active_samples_per_second'],
            final_metrics['loss'],
            checkpoint_sha256,
        )
        return completion
    except BaseException:
        logging.exception('缓存训练未完成')
        if progress['batches_completed'] > 0:
            torch.cuda.synchronize()
            update_timing(progress, run_started, wall_base, checkpoint_seconds)
            progress['metric_sums'] = cumulative_metrics.tolist()
            progress['status'] = 'interrupted'
            save_checkpoint(
                checkpoint_file,
                components=components,
                source_state=source_state,
                run_contract=run_contract,
                run_contract_sha256=contract_sha256,
                progress=progress,
            )
            logging.info('已保存中断 checkpoint: %s', checkpoint_file)
        raise
    finally:
        writer.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--initial-checkpoint', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--device', default=config['control']['device'])
    parser.add_argument('--batch-size', type=int, default=config['control']['batch_size'])
    parser.add_argument('--shuffle-seed', type=int, default=20260804)
    parser.add_argument('--log-every-steps', type=int, default=250)
    parser.add_argument('--checkpoint-every-steps', type=int, default=2048)
    parser.add_argument('--resume', action='store_true')
    return parser.parse_args()


def run(args):
    if args.batch_size <= 0 or args.log_every_steps <= 0:
        raise ValueError('batch-size 和 log-every-steps 必须为正整数')
    if args.checkpoint_every_steps < 0:
        raise ValueError('checkpoint-every-steps 不能为负数')
    output_dir = Path(args.output_dir).resolve()
    initial_checkpoint = Path(args.initial_checkpoint).resolve()
    if not initial_checkpoint.is_file():
        raise FileNotFoundError(f'初始 checkpoint 不存在: {initial_checkpoint}')
    manifest_file, manifest = load_manifest(args.manifest)

    if args.resume:
        if not (output_dir / 'latest.pth').is_file():
            raise FileNotFoundError('缺少可恢复的 latest.pth')
    else:
        if output_dir.exists() and any(output_dir.iterdir()):
            raise FileExistsError(f'输出目录非空，拒绝覆盖: {output_dir}')
        output_dir.mkdir(parents=True, exist_ok=True)

    configure_logging(output_dir)
    run_contract, contract_sha256 = build_run_contract(
        args,
        manifest_file,
        manifest,
        initial_checkpoint,
    )
    run_config_file = output_dir / 'run_config.json'
    if args.resume:
        with open(run_config_file, encoding='utf-8') as f:
            existing = json.load(f)
        if existing['run_contract_sha256'] != contract_sha256:
            raise RuntimeError('run_config 与当前输入不一致，拒绝恢复')
    else:
        atomic_write_json(run_config_file, {
            'run_contract': run_contract,
            'run_contract_sha256': contract_sha256,
            'config': config,
        })
        source_config = os.environ.get('MORTAL_CFG')
        if source_config:
            shutil.copy2(Path(source_config).resolve(), output_dir / 'source_config.toml')

    logging.info('output directory: %s', output_dir)
    logging.info('manifest: %s sha256=%s', manifest_file, run_contract['manifest_sha256'])
    logging.info(
        'initial checkpoint: %s sha256=%s',
        initial_checkpoint,
        run_contract['initial_checkpoint_sha256'],
    )
    logging.info('run contract SHA256: %s', contract_sha256)
    return train_one_epoch(
        args,
        output_dir,
        manifest_file,
        manifest,
        run_contract,
        contract_sha256,
    )


def main():
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    lock_file = output_dir.with_name(f'{output_dir.name}.lock')
    with exclusive_lock(lock_file):
        run(args)


if __name__ == '__main__':
    main()
