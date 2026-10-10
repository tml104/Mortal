"""新训练与消费者共享的原生 Conv2d checkpoint 契约。"""

from dataclasses import asdict
import os
from pathlib import Path

import torch

from model import DQN
from precompute_pipeline.common.optimized_model import BrainConv2d


SCHEMA_VERSION = 2
MODEL_FORMAT = 'conv2d_channels_last_v4'
METRIC_NAMES = ('loss', 'dqn_loss', 'cql_loss', 'rank_loss', 'rank_accuracy', 'q_mean', 'q_target_mean')
PROGRESS_KEYS = (
    'completed_epochs', 'current_epoch', 'epoch_samples_consumed',
    'samples_seen', 'batches_completed', 'last_validation', 'metric_sums', 'metric_samples',
)
TRAINING_KEYS = (*PROGRESS_KEYS, 'aux_net', 'optimizer', 'scaler', 'main_seed', 'torch_rng_state', 'cuda_rng_state_all', 'contract')


def training_contract(settings, generation_id):
    """只锁定数据和学习语义，不锁定日志频率、线程或目标轮数。"""
    return {
        'generation_id': generation_id, 'model_format': MODEL_FORMAT,
        'batch_size': settings.batch_size, 'enable_amp': settings.enable_amp,
        'validation_samples': settings.validation_samples, 'validation_seed': settings.validation_seed,
        'model': asdict(settings.model), 'loss': asdict(settings.loss),
        'optimizer': asdict(settings.optimizer),
    }


def model_metadata(model_settings):
    return {'version': 4, **asdict(model_settings)}


def atomic_torch_save(state, filename):
    """完整序列化并 fsync 后替换唯一恢复文件。"""
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    temporary = filename.with_name(f'{filename.name}.tmp-{os.getpid()}')
    try:
        torch.save(state, temporary)
        with temporary.open('rb') as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, filename)
    finally:
        temporary.unlink(missing_ok=True)


def load_checkpoint(filename, expected_contract=None, *, training=True):
    """只加载明确的新格式，推理不要求训练状态或数据文件。"""
    state = torch.load(filename, weights_only=True, map_location='cpu')
    if not isinstance(state, dict) or state.get('schema_version') != SCHEMA_VERSION or state.get('model_format') != MODEL_FORMAT:
        raise RuntimeError('checkpoint 格式不兼容，需要原生 Conv2d checkpoint')
    required = ('model_spec', 'mortal', 'dqn', *(TRAINING_KEYS if training else ()))
    missing = [key for key in required if key not in state]
    if missing:
        raise RuntimeError(f'checkpoint 缺少字段: {missing}')
    if training and not state['cuda_rng_state_all']:
        raise RuntimeError('checkpoint 缺少 CUDA 随机状态')
    spec = state['model_spec']
    if not isinstance(spec, dict) or spec.get('version') != 4 or not all(key in spec for key in ('conv_channels', 'num_blocks', 'freeze_bn')):
        raise RuntimeError('checkpoint 模型元信息不完整或不兼容')
    if training and spec != model_metadata_from_contract(state['contract']):
        raise RuntimeError('checkpoint 模型元信息与训练契约不兼容')
    if expected_contract is not None and state['contract'] != expected_contract:
        raise RuntimeError('checkpoint 与当前数据或训练语义不兼容')
    return state


def model_metadata_from_contract(contract):
    return {'version': 4, **contract['model']}


def load_policy(filename, *, device='cpu'):
    """只实例化推理所需网络；释放完整文件中的训练状态。"""
    state = load_checkpoint(filename, training=False)
    spec = state['model_spec']
    brain = BrainConv2d(conv_channels=spec['conv_channels'], num_blocks=spec['num_blocks'])
    dqn = DQN(version=4)
    brain.load_state_dict(state['mortal'], strict=True)
    dqn.load_state_dict(state['dqn'], strict=True)
    return brain.to(device).eval(), dqn.to(device).eval(), spec
