"""旧预计算完整 checkpoint 到新目录的显式迁移入口。"""

import argparse
import copy
from pathlib import Path
import sys

MORTAL_DIR = Path(__file__).resolve().parents[2]
if str(MORTAL_DIR) not in sys.path:
    sys.path.insert(0, str(MORTAL_DIR))

import torch

from precompute_pipeline.common.dataset import load_dataset_state
from precompute_pipeline.common.optimized_checkpoint import (
    SCHEMA_VERSION, MODEL_FORMAT, METRIC_NAMES, TRAINING_KEYS,
    atomic_torch_save, model_metadata, training_contract,
)
from precompute_pipeline.common.optimized_model import build_models, build_optimizer
from precompute_pipeline.train_optimize.settings import load_settings, validate_output


def run_migration(settings, source_checkpoint):
    """转换已提交的完整状态，不消费数据、不修改源文件、不开始训练。"""
    source_checkpoint = Path(source_checkpoint).resolve()
    validate_output(settings, source_checkpoint)
    output = settings.output_dir
    if output.exists() and any(output.iterdir()):
        raise ValueError('迁移目标必须是新的或空目录，不能覆盖非空目标')
    source = torch.load(source_checkpoint, weights_only=True, map_location='cpu')
    if source.get('schema_version') != 1 or 'model_format' in source:
        raise RuntimeError('只支持旧预计算链路的完整 checkpoint')
    missing = [key for key in ('mortal', 'dqn', *TRAINING_KEYS) if key not in source]
    if missing:
        raise RuntimeError(f'源 checkpoint 缺少字段: {missing}')
    if not source['cuda_rng_state_all']:
        raise RuntimeError('源 checkpoint 缺少 CUDA 随机状态，不能迁移为完整 CUDA 续训状态')
    dataset = load_dataset_state(settings.dataset_progress)
    contract = training_contract(settings, dataset['generation_id'])
    expected_source = {key: value for key, value in contract.items() if key != 'model_format'}
    expected_source['batch_size'] = source['contract']['batch_size']
    if source['contract'] != expected_source:
        raise RuntimeError('源 checkpoint 与目标数据或训练语义不兼容')
    with torch.random.fork_rng(devices=[]):
        models = build_models(settings.model)
        optimizer = build_optimizer(models, settings.optimizer)
    migrated = copy.deepcopy(source)
    for key, model in zip(('mortal', 'dqn', 'aux_net'), models, strict=True):
        target = model.state_dict()
        converted = {
            name: value.unsqueeze(2) if target[name].ndim == 4 and value.ndim == 3 else value
            for name, value in source[key].items()
        }
        model.load_state_dict(converted, strict=True)
        migrated[key] = model.state_dict()
    parameter_groups = optimizer.param_groups
    source_groups = migrated['optimizer']['param_groups']
    if len(parameter_groups) != len(source_groups):
        raise RuntimeError('优化器参数组不兼容')
    for source_group, target_group in zip(source_groups, parameter_groups, strict=True):
        if len(source_group['params']) != len(target_group['params']):
            raise RuntimeError('优化器参数数量不兼容')
        for parameter_id, parameter in zip(source_group['params'], target_group['params'], strict=True):
            for name, value in migrated['optimizer']['state'].get(parameter_id, {}).items():
                if name == 'step':
                    continue
                if parameter.ndim == 4 and value.ndim == 3:
                    value = value.unsqueeze(2)
                if value.shape != parameter.shape:
                    raise RuntimeError(f'优化器状态形状不兼容: {parameter_id}/{name}')
                migrated['optimizer']['state'][parameter_id][name] = value
    migrated.update(
        schema_version=SCHEMA_VERSION, model_format=MODEL_FORMAT,
        model_spec=model_metadata(settings.model), contract=contract,
        metric_sums=[0.0] * len(METRIC_NAMES), metric_samples=0,
        migration={'source': str(source_checkpoint), 'at_batch': source['batches_completed']},
    )
    destination = output / 'latest.pth'
    atomic_torch_save(migrated, destination)
    step = migrated['batches_completed']
    print(
        f'event=migrated source={source_checkpoint} target={destination} '
        f'batches={step} samples={migrated["samples_seen"]} '
        f'epoch_sample={migrated["epoch_samples_consumed"]} metric_window=cleared '
        f'next_log={(step // settings.log_every_batches + 1) * settings.log_every_batches} '
        f'next_save={(step // settings.save_every_batches + 1) * settings.save_every_batches}',
        flush=True,
    )
    return destination


def main(argv=None):
    parser = argparse.ArgumentParser(description='显式迁移旧预计算模型为原生 Conv2d')
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    args = parser.parse_args(argv)
    run_migration(load_settings(args.config), args.source_checkpoint)


if __name__ == '__main__':
    main()
