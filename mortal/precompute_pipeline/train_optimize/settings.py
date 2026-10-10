"""独立优化训练配置；相对路径以启动 PWD 解析。"""

from dataclasses import dataclass, field
import math
from pathlib import Path

from precompute_pipeline.common.settings import ModelSettings, LossSettings, OptimizerSettings
from precompute_pipeline.common.runtime import (
    read_toml, positive_int, boolean, path_value, optional_threads, configure_threads,
)


@dataclass(frozen=True)
class OptimizedTrainingSettings:
    dataset_progress: Path
    output_dir: Path
    device: str = 'cuda:0'
    batch_size: int = 2048
    num_epochs: int = 30
    enable_amp: bool = True
    enable_cudnn_benchmark: bool = True
    prefetch: bool = True
    log_every_batches: int = 50
    save_every_batches: int = 500
    validation_samples: int = 131072
    validation_seed: int = 0
    compile_threads: int = 2
    torch_threads: int | None = None
    model: ModelSettings = field(default_factory=lambda: ModelSettings(192, 40, False))
    loss: LossSettings = field(default_factory=lambda: LossSettings(1.0, 5.0, 0.2))
    optimizer: OptimizerSettings = field(default_factory=lambda: OptimizerSettings(1e-4, (0.9, 0.999), 1e-8, 0.1))
    protected_paths: tuple[Path, ...] = ()


def load_settings(filename):
    """只解析本入口的配置，不要求 quarantine/reward。"""
    raw = read_toml(filename)
    section = raw['training']
    model, loss, optimizer = (raw.get(key, {}) for key in ('model', 'loss', 'optimizer'))
    model_settings = ModelSettings(
        positive_int(model, 'conv_channels', 192), positive_int(model, 'num_blocks', 40),
        boolean(model, 'freeze_bn', False),
    )
    loss_settings = LossSettings(
        float(loss.get('gamma', 1.0)), float(loss.get('cql_weight', 5.0)), float(loss.get('rank_weight', 0.2)),
    )
    betas = tuple(optimizer.get('betas', (0.9, 0.999)))
    optimizer_settings = OptimizerSettings(
        float(optimizer.get('learning_rate', 1e-4)), betas,
        float(optimizer.get('eps', 1e-8)), float(optimizer.get('weight_decay', 0.1)),
    )
    if not 0 < loss_settings.gamma <= 1 or any(not math.isfinite(value) or value < 0 for value in (loss_settings.cql_weight, loss_settings.rank_weight)):
        raise ValueError('loss 参数无效')
    if len(betas) != 2 or any(not 0 <= value < 1 for value in betas) or not math.isfinite(optimizer_settings.learning_rate) or optimizer_settings.learning_rate <= 0 or not math.isfinite(optimizer_settings.eps) or optimizer_settings.eps <= 0 or not math.isfinite(optimizer_settings.weight_decay) or optimizer_settings.weight_decay < 0:
        raise ValueError('optimizer 参数无效')
    seed = section.get('validation_seed', 0)
    if type(seed) is not int or seed < 0:
        raise ValueError('validation_seed 必须是非负整数')
    return OptimizedTrainingSettings(
        dataset_progress=path_value(section, 'dataset_progress'), output_dir=path_value(section, 'output_dir'),
        device=section.get('device', 'cuda:0'),
        **{key: positive_int(section, key, default) for key, default in (
            ('batch_size', 2048), ('num_epochs', 30), ('log_every_batches', 50),
            ('save_every_batches', 500), ('validation_samples', 131072), ('compile_threads', 2),
        )},
        **{key: boolean(section, key, True) for key in ('enable_amp', 'enable_cudnn_benchmark', 'prefetch')},
        validation_seed=seed, torch_threads=optional_threads(section, 'torch_threads'),
        model=model_settings, loss=loss_settings, optimizer=optimizer_settings,
        protected_paths=(Path(filename).resolve(),),
    )


def validate_output(settings, *protected_paths):
    """从头清理只允许 artifacts 的具体子目录，并排除源数据和配置。"""
    output = settings.output_dir.resolve()
    if output.parent.name != 'artifacts':
        raise ValueError('训练输出必须是 artifacts 下的具体子目录')
    for protected in (settings.dataset_progress, *settings.protected_paths, *protected_paths):
        protected = Path(protected).resolve()
        if protected == output or output in protected.parents:
            raise ValueError(f'训练输出包含受保护路径: {protected}')
