"""预计算 chunks 的主模型训练、验证和恢复入口。"""

import argparse
from contextlib import nullcontext
from dataclasses import asdict
import json
import os
from pathlib import Path
import secrets
import shutil
import sys

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter


MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

from model import AuxNet, Brain, DQN
from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.settings import load_settings
from precompute_pipeline.train.reader import DatasetBatchReader


CHECKPOINT_SCHEMA_VERSION = 1
METRIC_NAMES = (
    'loss',
    'dqn_loss',
    'cql_loss',
    'rank_loss',
    'rank_accuracy',
    'q_mean',
    'q_target_mean',
)


def atomic_torch_save(state, filename):
    """在训练目录内原子替换唯一恢复 checkpoint。"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        torch.save(state, temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def _prepare_new_output(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for child in output_dir.iterdir():
        if child.is_dir() and not child.is_symlink():
            shutil.rmtree(child)
        else:
            child.unlink()


def _validate_output_dir(settings):
    """限制从头训练的递归清理只能作用于 artifacts 的具体子目录。"""
    output_dir = settings.output_dir.resolve()
    if output_dir.parent.name != 'artifacts':
        raise ValueError(
            f'训练输出必须是名为 artifacts 的目录下的具体子目录: {output_dir}'
        )
    dataset_progress = settings.dataset_progress.resolve()
    if dataset_progress == output_dir or output_dir in dataset_progress.parents:
        raise ValueError('训练输出目录不能包含预计算数据集状态')
    for protected in settings.protected_paths:
        protected = Path(protected).resolve()
        if protected == output_dir or output_dir in protected.parents:
            raise ValueError(f'训练输出目录不能包含受保护的配置或权重: {protected}')


def _seed_torch(seed, device):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def _build_models(settings, device):
    mortal = Brain(
        version=4,
        conv_channels=settings.model.conv_channels,
        num_blocks=settings.model.num_blocks,
    ).to(device)
    dqn = DQN(version=4).to(device)
    aux_net = AuxNet((4,)).to(device)
    mortal.freeze_bn(settings.model.freeze_bn)
    raw_models = (mortal, dqn, aux_net)
    execution_models = raw_models
    if settings.enable_compile:
        execution_models = tuple(
            torch.compile(model, dynamic=settings.compile_dynamic)
            for model in raw_models
        )
    for model in execution_models:
        model.train()
    return raw_models, execution_models


def _build_optimizer(models, settings):
    decay = []
    no_decay = []
    for model in models:
        for module in model.modules():
            for name, parameter in module.named_parameters(recurse=False):
                if isinstance(module, (nn.Linear, nn.Conv1d)) and name == 'weight':
                    decay.append(parameter)
                else:
                    no_decay.append(parameter)
    return torch.optim.AdamW(
        [
            {'params': decay, 'weight_decay': settings.optimizer.weight_decay},
            {'params': no_decay, 'weight_decay': 0.0},
        ],
        lr=settings.optimizer.learning_rate,
        betas=settings.optimizer.betas,
        eps=settings.optimizer.eps,
        weight_decay=0.0,
    )


def _training_contract(settings, generation_id):
    return {
        'generation_id': generation_id,
        'batch_size': settings.batch_size,
        'enable_amp': settings.enable_amp,
        'validation_samples': settings.validation_samples,
        'validation_seed': settings.validation_seed,
        'model': asdict(settings.model),
        'loss': asdict(settings.loss),
        'optimizer': asdict(settings.optimizer),
    }


def _load_checkpoint(filename, expected_contract):
    """安全读取并核对唯一恢复状态，不修补损坏或不兼容文件。"""
    state = torch.load(filename, weights_only=True, map_location='cpu')
    if state.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
        raise RuntimeError(f'训练 checkpoint schema 不兼容: {state.get("schema_version")}')
    if state.get('contract') != expected_contract:
        raise RuntimeError('训练 checkpoint 与当前数据或训练语义不兼容')
    required = (
        'mortal', 'dqn', 'aux_net', 'optimizer', 'scaler', 'main_seed',
        'completed_epochs', 'current_epoch', 'epoch_samples_consumed',
        'samples_seen', 'batches_completed', 'torch_rng_state',
        'metric_sums', 'metric_samples',
    )
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f'训练 checkpoint 缺少字段: {missing}')
    return state


def _progress_from_checkpoint(state):
    return {
        'completed_epochs': int(state['completed_epochs']),
        'current_epoch': int(state['current_epoch']),
        'epoch_samples_consumed': int(state['epoch_samples_consumed']),
        'samples_seen': int(state['samples_seen']),
        'batches_completed': int(state['batches_completed']),
        'last_validation': state.get('last_validation'),
        'metric_sums': list(state['metric_sums']),
        'metric_samples': int(state['metric_samples']),
    }


def _checkpoint_payload(
    *,
    raw_models,
    optimizer,
    scaler,
    main_seed,
    progress,
    contract,
    device,
):
    mortal, dqn, aux_net = raw_models
    payload = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'mortal': mortal.state_dict(),
        'dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
        'main_seed': main_seed,
        'completed_epochs': progress['completed_epochs'],
        'current_epoch': progress['current_epoch'],
        'epoch_samples_consumed': progress['epoch_samples_consumed'],
        'samples_seen': progress['samples_seen'],
        'batches_completed': progress['batches_completed'],
        'last_validation': progress['last_validation'],
        'metric_sums': progress['metric_sums'],
        'metric_samples': progress['metric_samples'],
        'contract': contract,
        'torch_rng_state': torch.get_rng_state(),
    }
    if device.type == 'cuda':
        payload['cuda_rng_state_all'] = torch.cuda.get_rng_state_all()
    return payload


def _forward_metrics(batch, execution_models, settings, device):
    mortal, dqn, aux_net = execution_models
    obs, actions, masks, steps, rewards, ranks = batch
    sample_count = len(obs)
    rows = torch.arange(sample_count, device=device)
    if not masks[rows, actions].all():
        raise RuntimeError('训练 batch 含 mask 不允许的 action')
    q_target = (settings.loss.gamma ** steps * rewards).to(torch.float32)
    autocast = (
        torch.autocast(device.type, enabled=settings.enable_amp)
        if device.type in {'cuda', 'cpu'}
        else nullcontext()
    )
    with autocast:
        phi = mortal(obs)
        q_all = dqn(phi, masks)
        q = q_all[rows, actions]
        dqn_loss = 0.5 * nn.functional.mse_loss(q, q_target)
        cql_loss = q_all.logsumexp(-1).mean() - q.mean()
        rank_logits, = aux_net(phi)
        rank_loss = nn.functional.cross_entropy(rank_logits, ranks)
        loss = (
            dqn_loss
            + settings.loss.cql_weight * cql_loss
            + settings.loss.rank_weight * rank_loss
        )
    accuracy = (rank_logits.argmax(-1) == ranks).to(torch.float32).mean()
    metrics = torch.stack((
        loss.detach(),
        dqn_loss.detach(),
        cql_loss.detach(),
        rank_loss.detach(),
        accuracy.detach(),
        q.detach().mean(),
        q_target.detach().mean(),
    )).to(torch.float64)
    return sample_count, loss, metrics


def _train_batch(batch, execution_models, optimizer, scaler, settings, device):
    optimizer.zero_grad(set_to_none=True)
    sample_count, loss, metrics = _forward_metrics(
        batch, execution_models, settings, device,
    )
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return sample_count, metrics


def _write_metrics(writer, metric_sums, sample_count, batch_step, optimizer):
    values = (metric_sums / sample_count).tolist()
    for name, value in zip(METRIC_NAMES, values, strict=True):
        writer.add_scalar(f'train/{name}', value, batch_step)
    writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], batch_step)


def _evaluate(
    *,
    settings,
    execution_models,
    device,
    max_samples,
):
    """按固定验证顺序计算样本加权指标，不更新任何网络状态。"""
    previous_modes = [model.training for model in execution_models]
    for model in execution_models:
        model.eval()
    metric_sums = torch.zeros(len(METRIC_NAMES), dtype=torch.float64, device=device)
    sample_count = 0
    reader = DatasetBatchReader(
        settings.dataset_progress,
        split='val',
        batch_size=settings.batch_size,
        device=device,
        seed=settings.validation_seed,
        epoch=0,
        prefetch=settings.prefetch,
    )
    try:
        with torch.inference_mode():
            for batch in reader:
                if max_samples is not None:
                    remaining = max_samples - sample_count
                    if remaining <= 0:
                        break
                    if len(batch[0]) > remaining:
                        batch = tuple(value[:remaining] for value in batch)
                current_count, _, metrics = _forward_metrics(
                    batch,
                    execution_models,
                    settings,
                    device,
                )
                metric_sums.add_(metrics * current_count)
                sample_count += current_count
                if max_samples is not None and sample_count >= max_samples:
                    break
    finally:
        for model, mode in zip(execution_models, previous_modes, strict=True):
            model.train(mode)
    if sample_count == 0:
        raise RuntimeError('验证集没有样本')
    values = (metric_sums / sample_count).tolist()
    return {
        'sample_count': sample_count,
        **dict(zip(METRIC_NAMES, values, strict=True)),
    }


def _record_validation(writer, result, kind, batch_step):
    for name in METRIC_NAMES:
        writer.add_scalar(f'validation/{kind}/{name}', result[name], batch_step)
    writer.add_scalar(
        f'validation/{kind}/sample_count', result['sample_count'], batch_step,
    )


def _save_training_checkpoint(
    latest,
    *,
    raw_models,
    optimizer,
    scaler,
    main_seed,
    progress,
    contract,
    device,
):
    atomic_torch_save(
        _checkpoint_payload(
            raw_models=raw_models,
            optimizer=optimizer,
            scaler=scaler,
            main_seed=main_seed,
            progress=progress,
            contract=contract,
            device=device,
        ),
        latest,
    )


def _validate_and_commit(
    *,
    kind,
    max_samples,
    writer,
    settings,
    execution_models,
    device,
    progress,
    latest,
    raw_models,
    optimizer,
    scaler,
    main_seed,
    contract,
):
    """完整验证后再把对应训练位置提交为唯一恢复状态。"""
    validation = _evaluate(
        settings=settings,
        execution_models=execution_models,
        device=device,
        max_samples=max_samples,
    )
    _record_validation(writer, validation, kind, progress['batches_completed'])
    progress['last_validation'] = {
        'kind': kind,
        'at_batch': progress['batches_completed'],
        **validation,
    }
    _save_training_checkpoint(
        latest,
        raw_models=raw_models,
        optimizer=optimizer,
        scaler=scaler,
        main_seed=main_seed,
        progress=progress,
        contract=contract,
        device=device,
    )


def run_training(settings, *, max_batches=None):
    """从头训练或从唯一 checkpoint 恢复主模型。"""
    dataset = load_dataset_state(settings.dataset_progress)
    _validate_output_dir(settings)
    device = torch.device(settings.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'配置设备不可用: {device}')
    if max_batches is not None and (not isinstance(max_batches, int) or max_batches <= 0):
        raise ValueError('max_batches 必须是正整数')
    latest = settings.output_dir / 'latest.pth'
    torch.backends.cudnn.benchmark = settings.enable_cudnn_benchmark
    contract = _training_contract(settings, dataset['generation_id'])
    restored_state = None
    if latest.exists():
        restored_state = _load_checkpoint(latest, contract)
        progress = _progress_from_checkpoint(restored_state)
        if progress['completed_epochs'] >= settings.num_epochs:
            return dict(progress)
        main_seed = int(restored_state['main_seed'])
    else:
        _prepare_new_output(settings.output_dir)
        main_seed = secrets.randbits(63)
        progress = {
            'completed_epochs': 0,
            'current_epoch': 0,
            'epoch_samples_consumed': 0,
            'samples_seen': 0,
            'batches_completed': 0,
            'last_validation': None,
            'metric_sums': [0.0] * len(METRIC_NAMES),
            'metric_samples': 0,
        }
    _seed_torch(main_seed, device)
    raw_models, execution_models = _build_models(settings, device)
    optimizer = _build_optimizer(raw_models, settings)
    scaler = torch.amp.GradScaler(device.type, enabled=settings.enable_amp)
    if restored_state is None:
        _save_training_checkpoint(
            latest,
            raw_models=raw_models,
            optimizer=optimizer,
            scaler=scaler,
            main_seed=main_seed,
            progress=progress,
            contract=contract,
            device=device,
        )
    else:
        mortal, dqn, aux_net = raw_models
        mortal.load_state_dict(restored_state['mortal'], strict=True)
        dqn.load_state_dict(restored_state['dqn'], strict=True)
        aux_net.load_state_dict(restored_state['aux_net'], strict=True)
        optimizer.load_state_dict(restored_state['optimizer'])
        scaler.load_state_dict(restored_state['scaler'])
        torch.set_rng_state(restored_state['torch_rng_state'])
        if device.type == 'cuda' and 'cuda_rng_state_all' in restored_state:
            torch.cuda.set_rng_state_all(restored_state['cuda_rng_state_all'])
        for model in execution_models:
            model.train()
    writer = SummaryWriter(
        settings.output_dir / 'tensorboard',
        purge_step=(
            progress['batches_completed'] + 1
            if restored_state is not None
            else None
        ),
    )
    metric_sums = torch.as_tensor(
        progress['metric_sums'], dtype=torch.float64, device=device,
    )
    metric_samples = progress['metric_samples']
    stopped_early = False
    try:
        for epoch in range(progress['current_epoch'], settings.num_epochs):
            progress['current_epoch'] = epoch
            reader = DatasetBatchReader(
                settings.dataset_progress,
                split='train',
                batch_size=settings.batch_size,
                device=device,
                seed=main_seed,
                epoch=epoch,
                prefetch=settings.prefetch,
            )
            epoch_start = progress['epoch_samples_consumed']
            for batch in reader.iter_from(epoch_start):
                sample_count, metrics = _train_batch(
                    batch,
                    execution_models,
                    optimizer,
                    scaler,
                    settings,
                    device,
                )
                progress['epoch_samples_consumed'] += sample_count
                progress['samples_seen'] += sample_count
                progress['batches_completed'] += 1
                metric_sums.add_(metrics * sample_count)
                metric_samples += sample_count
                progress['metric_sums'] = metric_sums.tolist()
                progress['metric_samples'] = metric_samples
                if progress['batches_completed'] % settings.log_every_batches == 0:
                    _write_metrics(
                        writer,
                        metric_sums,
                        metric_samples,
                        progress['batches_completed'],
                        optimizer,
                    )
                    metric_sums.zero_()
                    metric_samples = 0
                    progress['metric_sums'] = [0.0] * len(METRIC_NAMES)
                    progress['metric_samples'] = 0
                epoch_finished = progress['epoch_samples_consumed'] == reader.total_samples
                periodic_due = (
                    progress['batches_completed'] % settings.save_every_batches == 0
                )
                if periodic_due and not epoch_finished:
                    _validate_and_commit(
                        kind='periodic',
                        max_samples=settings.validation_samples,
                        writer=writer,
                        settings=settings,
                        execution_models=execution_models,
                        device=device,
                        progress=progress,
                        latest=latest,
                        raw_models=raw_models,
                        optimizer=optimizer,
                        scaler=scaler,
                        main_seed=main_seed,
                        contract=contract,
                    )
                if max_batches is not None and progress['batches_completed'] >= max_batches:
                    if not periodic_due and not epoch_finished:
                        _validate_and_commit(
                            kind='periodic',
                            max_samples=settings.validation_samples,
                            writer=writer,
                            settings=settings,
                            execution_models=execution_models,
                            device=device,
                            progress=progress,
                            latest=latest,
                            raw_models=raw_models,
                            optimizer=optimizer,
                            scaler=scaler,
                            main_seed=main_seed,
                            contract=contract,
                        )
                    stopped_early = True
                    break
            if stopped_early:
                break
            validation = _evaluate(
                settings=settings,
                execution_models=execution_models,
                device=device,
                max_samples=None,
            )
            _record_validation(
                writer, validation, 'epoch', progress['batches_completed'],
            )
            progress['completed_epochs'] = epoch + 1
            progress['current_epoch'] = epoch + 1
            progress['epoch_samples_consumed'] = 0
            progress['last_validation'] = {
                'kind': 'epoch',
                'at_batch': progress['batches_completed'],
                **validation,
            }
            _save_training_checkpoint(
                latest,
                raw_models=raw_models,
                optimizer=optimizer,
                scaler=scaler,
                main_seed=main_seed,
                progress=progress,
                contract=contract,
                device=device,
            )
        if metric_samples:
            _write_metrics(
                writer,
                metric_sums,
                metric_samples,
                progress['batches_completed'],
                optimizer,
            )
        return dict(progress)
    finally:
        writer.flush()
        writer.close()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='训练 Mortal 预计算主模型')
    parser.add_argument('--config', type=Path, required=True)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    pipeline_settings = load_settings(args.config)
    if pipeline_settings.training is None:
        raise ValueError('配置缺少 [training]、[model]、[loss] 或 [optimizer]')
    result = run_training(pipeline_settings.training)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
