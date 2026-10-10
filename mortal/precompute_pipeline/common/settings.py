"""读取新预计算链路的独立 TOML 配置。"""

from dataclasses import dataclass
from pathlib import Path
import tomllib


@dataclass(frozen=True)
class QuarantineSettings:
    train_dir: Path
    val_dir: Path
    quarantine_dir: Path
    report_file: Path
    progress_file: Path
    commit_every: int
    worker_processes: int
    rayon_threads_per_worker: int
    max_in_flight: int


@dataclass(frozen=True)
class RewardSettings:
    state_file: Path
    d_model: int
    nhead: int
    num_layers: int
    max_seq_len: int
    pts: tuple[float, float, float, float]


@dataclass(frozen=True)
class PrecomputeSettings:
    train_dir: Path
    val_dir: Path
    logical_root: Path
    progress_file: Path
    c_root: Path
    d_root: Path
    c_budget_bytes: int
    d_budget_bytes: int
    chunk_size: int
    file_batch_size: int
    compression_level: int
    rayon_threads: int
    torch_threads: int
    reward: RewardSettings
    log_every_chunks: int = 1


@dataclass(frozen=True)
class ModelSettings:
    conv_channels: int
    num_blocks: int
    freeze_bn: bool


@dataclass(frozen=True)
class LossSettings:
    gamma: float
    cql_weight: float
    rank_weight: float


@dataclass(frozen=True)
class OptimizerSettings:
    learning_rate: float
    betas: tuple[float, float]
    eps: float
    weight_decay: float


@dataclass(frozen=True)
class TrainingSettings:
    dataset_progress: Path
    output_dir: Path
    device: str
    batch_size: int
    num_epochs: int
    enable_amp: bool
    enable_compile: bool
    compile_dynamic: bool
    enable_cudnn_benchmark: bool
    prefetch: bool
    log_every_batches: int
    save_every_batches: int
    validation_samples: int
    validation_seed: int
    model: ModelSettings
    loss: LossSettings
    optimizer: OptimizerSettings
    protected_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class PipelineSettings:
    source_file: Path
    quarantine: QuarantineSettings
    precompute: PrecomputeSettings | None
    training: TrainingSettings | None


def _resolve_path(base_dir, value, key):
    if not isinstance(value, str) or not value:
        raise ValueError(f'{key} 必须是非空路径字符串')
    filename = Path(value).expanduser()
    if not filename.is_absolute():
        filename = base_dir / filename
    return filename.resolve()


def load_settings(filename):
    """读取配置，并以启动进程的当前工作目录解析相对路径。"""
    filename = Path(filename).resolve()
    with open(filename, 'rb') as f:
        raw = tomllib.load(f)
    try:
        section = raw['quarantine']
    except KeyError as exc:
        raise ValueError('配置缺少 [quarantine]') from exc
    if not isinstance(section, dict):
        raise ValueError('[quarantine] 必须是 TOML table')

    base_dir = Path.cwd()
    quarantine_dir = _resolve_path(
        base_dir,
        section.get('quarantine_dir'),
        'quarantine.quarantine_dir',
    )
    commit_every = section.get('commit_every', 100)
    if not isinstance(commit_every, int) or isinstance(commit_every, bool) or commit_every <= 0:
        raise ValueError('quarantine.commit_every 必须是正整数')

    def quarantine_positive_int(key, default):
        value = section.get(key, default)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f'quarantine.{key} 必须是正整数')
        return value

    worker_processes = quarantine_positive_int('worker_processes', 8)
    rayon_threads_per_worker = quarantine_positive_int(
        'rayon_threads_per_worker', 2,
    )
    max_in_flight = quarantine_positive_int('max_in_flight', 16)
    if max_in_flight < worker_processes:
        raise ValueError(
            'quarantine.max_in_flight 不能小于 quarantine.worker_processes'
        )

    quarantine = QuarantineSettings(
        train_dir=_resolve_path(base_dir, section.get('train_dir'), 'quarantine.train_dir'),
        val_dir=_resolve_path(base_dir, section.get('val_dir'), 'quarantine.val_dir'),
        quarantine_dir=quarantine_dir,
        report_file=_resolve_path(
            base_dir,
            section.get('report_file', str(quarantine_dir / 'report.jsonl')),
            'quarantine.report_file',
        ),
        progress_file=_resolve_path(
            base_dir,
            section.get('progress_file', str(quarantine_dir / 'progress.json')),
            'quarantine.progress_file',
        ),
        commit_every=commit_every,
        worker_processes=worker_processes,
        rayon_threads_per_worker=rayon_threads_per_worker,
        max_in_flight=max_in_flight,
    )

    precompute = None
    if 'precompute' in raw:
        section = raw['precompute']
        reward_raw = raw.get('reward')
        if not isinstance(section, dict) or not isinstance(reward_raw, dict):
            raise ValueError('[precompute] 和 [reward] 必须同时是 TOML table')

        def positive_int(key, default=None):
            value = section.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'precompute.{key} 必须是正整数')
            return value

        def budget_bytes(key):
            value = section.get(key)
            if (
                not isinstance(value, (int, float))
                or isinstance(value, bool)
                or value <= 0
            ):
                raise ValueError(f'precompute.{key} 必须是正数 GiB')
            return int(value * 2 ** 30)

        pts = reward_raw.get('pts')
        if (
            not isinstance(pts, list)
            or len(pts) != 4
            or not all(isinstance(value, (int, float)) for value in pts)
        ):
            raise ValueError('reward.pts 必须是四个数')
        reward = RewardSettings(
            state_file=_resolve_path(
                base_dir, reward_raw.get('state_file'), 'reward.state_file',
            ),
            d_model=int(reward_raw.get('d_model', 128)),
            nhead=int(reward_raw.get('nhead', 4)),
            num_layers=int(reward_raw.get('num_layers', 4)),
            max_seq_len=int(reward_raw.get('max_seq_len', 60)),
            pts=tuple(float(value) for value in pts),
        )
        for key in ('d_model', 'nhead', 'num_layers', 'max_seq_len'):
            if getattr(reward, key) <= 0:
                raise ValueError(f'reward.{key} 必须为正数')
        logical_root = _resolve_path(
            base_dir, section.get('logical_root'), 'precompute.logical_root',
        )
        precompute = PrecomputeSettings(
            train_dir=_resolve_path(
                base_dir,
                section.get('train_dir', str(quarantine.train_dir)),
                'precompute.train_dir',
            ),
            val_dir=_resolve_path(
                base_dir,
                section.get('val_dir', str(quarantine.val_dir)),
                'precompute.val_dir',
            ),
            logical_root=logical_root,
            progress_file=_resolve_path(
                base_dir,
                section.get('progress_file', str(logical_root / 'progress.json')),
                'precompute.progress_file',
            ),
            c_root=_resolve_path(base_dir, section.get('c_root'), 'precompute.c_root'),
            d_root=_resolve_path(base_dir, section.get('d_root'), 'precompute.d_root'),
            c_budget_bytes=budget_bytes('c_budget_gib'),
            d_budget_bytes=budget_bytes('d_budget_gib'),
            chunk_size=positive_int('chunk_size', 16_384),
            file_batch_size=positive_int('file_batch_size', 4),
            compression_level=positive_int('compression_level', 1),
            rayon_threads=positive_int('rayon_threads', 16),
            torch_threads=positive_int('torch_threads', 1),
            reward=reward,
            log_every_chunks=positive_int('log_every_chunks', 1),
        )
    training = None
    if 'training' in raw:
        section = raw['training']
        model_raw = raw.get('model')
        loss_raw = raw.get('loss')
        optimizer_raw = raw.get('optimizer')
        if not all(isinstance(value, dict) for value in (
            section, model_raw, loss_raw, optimizer_raw,
        )):
            raise ValueError(
                '[training]、[model]、[loss] 和 [optimizer] 必须同时是 TOML table'
            )

        def training_positive_int(key, default=None):
            value = section.get(key, default)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f'training.{key} 必须是正整数')
            return value

        def boolean(table, key, default, prefix):
            value = table.get(key, default)
            if not isinstance(value, bool):
                raise ValueError(f'{prefix}.{key} 必须是布尔值')
            return value

        betas = optimizer_raw.get('betas', [0.9, 0.999])
        if (
            not isinstance(betas, list)
            or len(betas) != 2
            or not all(isinstance(value, (int, float)) for value in betas)
        ):
            raise ValueError('optimizer.betas 必须是两个数')
        model = ModelSettings(
            conv_channels=int(model_raw.get('conv_channels', 192)),
            num_blocks=int(model_raw.get('num_blocks', 40)),
            freeze_bn=boolean(model_raw, 'freeze_bn', False, 'model'),
        )
        if model.conv_channels <= 0 or model.num_blocks <= 0:
            raise ValueError('model 网络规模必须是正整数')
        training = TrainingSettings(
            dataset_progress=_resolve_path(
                base_dir,
                section.get('dataset_progress'),
                'training.dataset_progress',
            ),
            output_dir=_resolve_path(
                base_dir, section.get('output_dir'), 'training.output_dir',
            ),
            device=str(section.get('device', 'cuda:0')),
            batch_size=training_positive_int('batch_size', 1024),
            num_epochs=training_positive_int('num_epochs', 1),
            enable_amp=boolean(section, 'enable_amp', True, 'training'),
            enable_compile=boolean(section, 'enable_compile', False, 'training'),
            compile_dynamic=boolean(section, 'compile_dynamic', True, 'training'),
            enable_cudnn_benchmark=boolean(
                section, 'enable_cudnn_benchmark', True, 'training',
            ),
            prefetch=boolean(section, 'prefetch', True, 'training'),
            log_every_batches=training_positive_int('log_every_batches', 100),
            save_every_batches=training_positive_int('save_every_batches', 1000),
            validation_samples=training_positive_int('validation_samples', 131_072),
            validation_seed=int(section.get('validation_seed', 0)),
            model=model,
            loss=LossSettings(
                gamma=float(loss_raw.get('gamma', 1.0)),
                cql_weight=float(loss_raw.get('cql_weight', 5.0)),
                rank_weight=float(loss_raw.get('rank_weight', 0.2)),
            ),
            optimizer=OptimizerSettings(
                learning_rate=float(optimizer_raw.get('learning_rate', 1e-4)),
                betas=(float(betas[0]), float(betas[1])),
                eps=float(optimizer_raw.get('eps', 1e-8)),
                weight_decay=float(optimizer_raw.get('weight_decay', 0.1)),
            ),
            protected_paths=(
                filename,
                *((precompute.reward.state_file,) if precompute is not None else ()),
            ),
        )
        if not training.device:
            raise ValueError('training.device 不能为空')
    return PipelineSettings(
        source_file=filename,
        quarantine=quarantine,
        precompute=precompute,
        training=training,
    )
