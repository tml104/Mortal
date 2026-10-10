"""原生 Conv2d/channels_last + 静态编译/CUDA Graph 的独立训练入口。"""

import argparse
from pathlib import Path
import secrets
import shutil
import sys
import time

MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.optimized_checkpoint import (
    SCHEMA_VERSION, MODEL_FORMAT, METRIC_NAMES, PROGRESS_KEYS,
    atomic_torch_save, load_checkpoint, model_metadata, training_contract,
)
from precompute_pipeline.common.optimized_model import build_models, build_optimizer
from precompute_pipeline.train_optimize.reader import DatasetBatchReader
from precompute_pipeline.train_optimize.settings import configure_threads, load_settings, validate_output
from precompute_pipeline.train_optimize.progress import TrainingProgress
from precompute_pipeline.train_optimize.run_logging import (
    TrainingLog, is_run_log, exception_reported, report_terminal_exception,
)


class LearningModel(nn.Module):
    """将三网络和损失放入同一可编译前向，原始参数仍独立保存。"""

    def __init__(self, models, loss_settings):
        super().__init__()
        self.mortal, self.dqn, self.aux_net = models
        self.gamma = loss_settings.gamma
        self.cql_weight = loss_settings.cql_weight
        self.rank_weight = loss_settings.rank_weight

    def forward(self, obs, actions, masks, steps, rewards, ranks):
        phi = self.mortal(obs)
        q_all = self.dqn(phi, masks)
        q = q_all.gather(1, actions[:, None]).squeeze(1)
        q_target = (self.gamma ** steps * rewards).to(torch.float32)
        dqn_loss = 0.5 * nn.functional.mse_loss(q, q_target)
        cql_loss = q_all.logsumexp(-1).mean() - q.mean()
        rank_logits, = self.aux_net(phi)
        rank_loss = nn.functional.cross_entropy(rank_logits, ranks)
        loss = dqn_loss + self.cql_weight * cql_loss + self.rank_weight * rank_loss
        accuracy = (rank_logits.argmax(-1) == ranks).float().mean()
        metrics = torch.stack((
            loss.detach(), dqn_loss.detach(), cql_loss.detach(), rank_loss.detach(),
            accuracy.detach(), q.detach().mean(), q_target.detach().mean(),
        )).to(torch.float64)
        return loss, metrics


class BatchExecutor:
    """完整 batch 固定编译，小 batch 使用同一 network 的 eager 前向。"""

    def __init__(self, models, settings, device):
        self.network = LearningModel(models, settings.loss)
        self.compiled = torch.compile(self.network, dynamic=False, fullgraph=True, mode='reduce-overhead')
        self.settings, self.device = settings, device

    def forward(self, batch):
        """必要动作约束在图外检查，图边界不产生额外参数更新。"""
        if not batch[2].gather(1, batch[1][:, None]).all():
            raise RuntimeError('训练 batch 含 mask 不允许的 action')
        torch.compiler.cudagraph_mark_step_begin()
        target = self.compiled if len(batch[0]) == self.settings.batch_size else self.network
        with torch.autocast('cuda', dtype=torch.float16, enabled=self.settings.enable_amp):
            return target(*batch)


def _new_progress():
    return {
        'completed_epochs': 0, 'current_epoch': 0, 'epoch_samples_consumed': 0,
        'samples_seen': 0, 'batches_completed': 0, 'last_validation': None,
        'metric_sums': [0.0] * len(METRIC_NAMES), 'metric_samples': 0,
    }


def _payload(models, optimizer, scaler, seed, progress, contract, settings):
    return {
        'schema_version': SCHEMA_VERSION, 'model_format': MODEL_FORMAT,
        'model_spec': model_metadata(settings.model), 'contract': contract,
        **dict(zip(('mortal', 'dqn', 'aux_net'), (model.state_dict() for model in models), strict=True)),
        'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(), 'main_seed': seed,
        'torch_rng_state': torch.get_rng_state(), 'cuda_rng_state_all': torch.cuda.get_rng_state_all(),
        **progress,
    }


def _metric_values(sums, count):
    return dict(zip(METRIC_NAMES, (sums / count).tolist(), strict=True))


def _write_metrics(writer, values, kind, step):
    for name, value in values.items():
        writer.add_scalar(f'{kind}/{name}', value, step)


def _evaluate(settings, executor, *, max_samples=None, log=None, progress=None, kind='epoch', output=None):
    """固定验证顺序，不改变训练游标、BN 或优化器。"""
    started = log.clock() if log else time.monotonic()
    if log:
        log.pause_training()
    mode = executor.network.training
    executor.network.eval()
    sums = torch.zeros(len(METRIC_NAMES), dtype=torch.float64, device=executor.device)
    count = 0
    reader = DatasetBatchReader(
        settings.dataset_progress, split='val', batch_size=settings.batch_size,
        device=executor.device, seed=settings.validation_seed, epoch=0, prefetch=settings.prefetch,
    )
    iterator = iter(reader)
    total = min(reader.total_samples, max_samples) if max_samples is not None else reader.total_samples
    if log:
        log.emit('validation_start', progress, kind=kind, validation_total=total)
    batches = 0
    try:
        with torch.inference_mode():
            for batch in iterator:
                if output is not None:
                    output.raise_if_failed()
                if max_samples is not None:
                    remaining = max_samples - count
                    if remaining <= 0:
                        break
                    batch = tuple(value[:remaining] for value in batch)
                loss, metrics = executor.forward(batch)
                current = len(batch[0])
                sums.add_(metrics * current)
                del loss, metrics
                count += current
                batches += 1
                if log and batches % settings.log_every_batches == 0:
                    log.validation_progress(progress, kind, count, total, started)
                if max_samples is not None and count >= max_samples:
                    break
    finally:
        if hasattr(iterator, 'close'):
            iterator.close()
        executor.network.train(mode)
    if count == 0:
        raise RuntimeError('验证集没有样本')
    result = {'sample_count': count, **_metric_values(sums, count)}
    if log:
        if batches % settings.log_every_batches:
            log.validation_progress(progress, kind, count, total, started)
        log.emit('validation_end', progress, kind=kind, metric_kind='validation', **result)
        log.resume_training()
    return result


def run_training(settings, *, max_batches=None, started_at=None, clock=time.monotonic,
                 wall_clock=time.time, wall_started_at=None):
    """公开入口在安全的模型目录管理本次文件与终端日志。"""
    if started_at is None:
        started_at = clock()
    if wall_started_at is None:
        wall_started_at = wall_clock()
    validate_output(settings)
    with TrainingLog(settings.output_dir, wall_started_at, wall_clock) as output:
        return _run_training(settings, max_batches=max_batches, started_at=started_at,
                             clock=clock, output=output)


def _run_training(settings, *, max_batches=None, started_at=None, clock=time.monotonic, output):
    """正常训练；max_batches 限制本次新增批数，用于有限验收。"""
    log = TrainingProgress(settings, started_at=started_at, clock=clock)
    validate_output(settings)
    configure_threads(settings.torch_threads, compile_threads=settings.compile_threads)
    device = torch.device(settings.device)
    if max_batches is not None and (type(max_batches) is not int or max_batches <= 0):
        raise ValueError('max_batches 必须是正整数')
    dataset = load_dataset_state(settings.dataset_progress)
    contract = training_contract(settings, dataset['generation_id'])
    latest = settings.output_dir / 'latest.pth'
    restored = load_checkpoint(latest, contract) if latest.exists() else None
    if restored is None:
        seed, progress = secrets.randbits(63), _new_progress()
    else:
        seed = restored['main_seed']
        progress = {key: restored[key] for key in PROGRESS_KEYS}
    log.saved_batch = progress['batches_completed'] if restored else 0
    log.emit(
        'resume' if restored else 'start', progress, device=device, batch_size=settings.batch_size,
        execution='conv2d_channels_last_compile_reduce_overhead',
        compile_threads=settings.compile_threads, torch_threads=torch.get_num_threads(),
        epoch_sample=progress['epoch_samples_consumed'],
    )
    if progress['completed_epochs'] >= settings.num_epochs:
        log.emit('complete', progress, already_complete=True)
        return progress
    if device.type != 'cuda' or not torch.cuda.is_available():
        raise RuntimeError('优化训练需要可用的 CUDA 设备')
    if restored is None:
        settings.output_dir.mkdir(parents=True, exist_ok=True)
        for child in settings.output_dir.iterdir():
            if is_run_log(child):
                continue
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    log.emit('prepare', progress, first_compile='pending')
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = settings.enable_cudnn_benchmark
    models = build_models(settings.model, device)
    optimizer = build_optimizer(models, settings.optimizer)
    scaler = torch.amp.GradScaler('cuda', enabled=settings.enable_amp)
    if restored is not None:
        for key, model in zip(('mortal', 'dqn', 'aux_net'), models, strict=True):
            model.load_state_dict(restored[key], strict=True)
        optimizer.load_state_dict(restored['optimizer'])
        scaler.load_state_dict(restored['scaler'])
        torch.set_rng_state(restored['torch_rng_state'])
        torch.cuda.set_rng_state_all(restored['cuda_rng_state_all'])
    executor = BatchExecutor(models, settings, device)
    log.emit('prepared', progress, first_compile='pending')
    sums = torch.tensor(progress['metric_sums'], device=device, dtype=torch.float64)
    metric_samples = progress['metric_samples']

    def save(kind='initial'):
        output.raise_if_failed()
        log.pause_training()
        progress['metric_sums'] = sums.tolist()
        progress['metric_samples'] = metric_samples
        payload = _payload(models, optimizer, scaler, seed, progress, contract, settings)
        output.raise_if_failed()
        atomic_torch_save(payload, latest)
        log.saved_batch = progress['batches_completed']
        log.emit('checkpoint', progress, kind=kind, path=latest)
        log.resume_training()

    if restored is None:
        save('initial')
    writer = SummaryWriter(settings.output_dir / 'tensorboard', purge_step=(progress['batches_completed'] + 1 if restored else None))
    new_batches = 0

    def validate_and_save(kind, max_samples):
        validation = _evaluate(settings, executor, max_samples=max_samples, log=log, progress=progress, kind=kind, output=output)
        _write_metrics(writer, validation, f'validation/{kind}', progress['batches_completed'])
        progress['last_validation'] = {'kind': kind, 'at_batch': progress['batches_completed'], **validation}
        save(kind)

    try:
        for epoch in range(progress['current_epoch'], settings.num_epochs):
            progress['current_epoch'] = epoch
            reader = DatasetBatchReader(
                settings.dataset_progress, split='train', batch_size=settings.batch_size,
                device=device, seed=seed, epoch=epoch, prefetch=settings.prefetch,
            )
            executor.network.train()
            log.resume_training()
            for batch in reader.iter_from(progress['epoch_samples_consumed']):
                output.raise_if_failed()
                if new_batches == 0:
                    log.emit('first_batch_start', progress, actual_batch=len(batch[0]))
                optimizer.zero_grad(set_to_none=True)
                loss, metrics = executor.forward(batch)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                count = len(batch[0])
                sums.add_(metrics * count)
                del loss, metrics
                metric_samples += count
                progress['epoch_samples_consumed'] += count
                progress['samples_seen'] += count
                progress['batches_completed'] += 1
                new_batches += 1
                log.processed(count)
                if new_batches == 1:
                    log.emit('first_batch_end', progress, actual_batch=count, first_window_includes_compile=True)
                if progress['batches_completed'] % settings.log_every_batches == 0:
                    values = _metric_values(sums, metric_samples)
                    _write_metrics(writer, values, 'train', progress['batches_completed'])
                    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], progress['batches_completed'])
                    log.training_progress(progress, reader.total_samples, values)
                    sums.zero_()
                    metric_samples = 0
                epoch_finished = progress['epoch_samples_consumed'] == reader.total_samples
                periodic_due = progress['batches_completed'] % settings.save_every_batches == 0
                if periodic_due and not epoch_finished:
                    validate_and_save('periodic', settings.validation_samples)
                if max_batches is not None and new_batches >= max_batches and not epoch_finished:
                    if not periodic_due:
                        validate_and_save('periodic', settings.validation_samples)
                    if metric_samples:
                        values = _metric_values(sums, metric_samples)
                        _write_metrics(writer, values, 'train', progress['batches_completed'])
                        log.training_progress(progress, reader.total_samples, values, reset=False)
                    log.emit('paused', progress, reason='bounded_run')
                    return dict(progress)
            if progress['batches_completed'] % settings.log_every_batches:
                log.training_progress(progress, reader.total_samples, _metric_values(sums, metric_samples), reset=False)
            validation = _evaluate(settings, executor, log=log, progress=progress, output=output)
            _write_metrics(writer, validation, 'validation/epoch', progress['batches_completed'])
            progress.update(
                completed_epochs=epoch + 1, current_epoch=epoch + 1, epoch_samples_consumed=0,
                last_validation={'kind': 'epoch', 'at_batch': progress['batches_completed'], **validation},
            )
            save('epoch')
            log.emit('epoch_complete', progress, completed_epochs=progress['completed_epochs'])
            if max_batches is not None and new_batches >= max_batches:
                break
        if metric_samples:
            _write_metrics(writer, _metric_values(sums, metric_samples), 'train', progress['batches_completed'])
        log.emit('complete' if progress['completed_epochs'] >= settings.num_epochs else 'paused', progress)
        return dict(progress)
    finally:
        writer.flush()
        writer.close()


def main(argv=None):
    started_at = time.monotonic()
    wall_started_at = time.time()
    parser = argparse.ArgumentParser(description='训练原生 Conv2d 优化主模型')
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        run_training(load_settings(args.config), started_at=started_at, wall_started_at=wall_started_at)
    except (Exception, KeyboardInterrupt) as error:
        if not exception_reported(error):
            report_terminal_exception(error)
        raise SystemExit(130 if isinstance(error, KeyboardInterrupt) else 1) from None


if __name__ == '__main__':
    main()
