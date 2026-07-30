import argparse
from contextlib import contextmanager
import fcntl
import gc
import hashlib
import json
import logging
import os
import random
import shutil
import stat
import sys
import time
from datetime import datetime
from glob import glob, iglob
from itertools import chain
from os import path

import torch
from torch import nn, optim
from torch.amp import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import prelude
import libriichi
from common import filtered_trimmed_lines, parameter_count
from config import config
from dataloader_grp_transformer import (
    FileDatasetsIter,
    dataset_file_stat,
    is_quarantined_dataset_file,
    worker_init_fn,
)
from libriichi.consts import obs_shape
from lr_scheduler import LinearWarmUpCosineAnnealingLR
from model import AuxNet, Brain, DQN


CHECKPOINT_SCHEMA_VERSION = 1
INVALID_ACTION_FIX_LEGACY_CONTRACT_SHA256 = (
    '3ed8bafd0dc6b2f5b0a8e7b2f4e29e999b461d6c7f85bae331178db23e101a0b'
)
DATASET_QUARANTINE_LEGACY_CONTRACT_SHA256 = (
    '113c51969c23f847b2a271f638d796b10aaaddd09cc90b6ac3460b28ffd82c12'
)
ALLOWED_IMPLEMENTATION_MIGRATIONS = {
    INVALID_ACTION_FIX_LEGACY_CONTRACT_SHA256: '非法动作过滤与坏数据隔离',
    DATASET_QUARANTINE_LEGACY_CONTRACT_SHA256: '坏数据隔离',
}
IMPLEMENTATION_MIGRATION_FILES = frozenset((
    'train_offline_grp_transformer.py',
    'dataloader_grp_transformer.py',
))
METRIC_NAMES = (
    'loss',
    'dqn_loss',
    'cql_loss',
    'next_rank_loss',
    'q_mean',
    'q_target_mean',
)


def sha256_file(filename):
    """流式计算文件摘要，避免把大型 checkpoint 一次性读入内存。"""
    digest = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_paths(paths):
    """对确定顺序的文件列表计算摘要，用于保证断点恢复时仍是同一 epoch。"""
    digest = hashlib.sha256()
    for filename in paths:
        digest.update(os.fsencode(filename))
        digest.update(b'\0')
    return digest.hexdigest()


def dataset_manifest_record(filename):
    """读取单个训练文件的稳定元数据，并编码为可哈希记录。"""
    try:
        file_stat = dataset_file_stat(filename)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f'离线训练文件不存在: {filename}') from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f'离线训练路径不是普通文件: {filename}')
    return b''.join((
        os.fsencode(filename),
        b'\0',
        str(file_stat.st_size).encode('ascii'),
        b'\0',
        str(file_stat.st_mtime_ns).encode('ascii'),
        b'\0',
    ))


def sha256_dataset_manifest(paths):
    """用路径、大小和纳秒修改时间锁定本轮训练所见的数据清单。"""
    digest = hashlib.sha256()
    for filename in paths:
        digest.update(dataset_manifest_record(filename))
    return digest.hexdigest()


def build_dataset_manifests(paths, shard_size):
    """一次遍历生成全量及逐分片清单摘要，供训练前后交叉核验。"""
    if shard_size <= 0:
        raise ValueError('file_shard_size 必须为正整数')
    full_digest = hashlib.sha256()
    shard_digest = hashlib.sha256()
    shard_digests = []
    for index, filename in enumerate(paths, start=1):
        record = dataset_manifest_record(filename)
        full_digest.update(record)
        shard_digest.update(record)
        if index % shard_size == 0:
            shard_digests.append(shard_digest.hexdigest())
            shard_digest = hashlib.sha256()
    if len(paths) % shard_size:
        shard_digests.append(shard_digest.hexdigest())
    return full_digest.hexdigest(), shard_digests


def sha256_json(value):
    """对可 JSON 序列化的训练契约生成稳定摘要。"""
    payload = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
    ).encode('utf-8')
    return hashlib.sha256(payload).hexdigest()


@contextmanager
def training_lock(filename):
    """持有进程级排他锁，防止两个训练实例并发写同一 checkpoint。"""
    filename = path.abspath(filename)
    os.makedirs(path.dirname(filename), exist_ok=True)
    with open(filename, 'a+', encoding='utf-8') as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'已有离线训练实例持有运行锁: {filename}') from exc
        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f'pid={os.getpid()} started={datetime.now().astimezone().isoformat()}\n')
        lock_file.flush()
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def fsync_parent_directory(filename):
    """同步父目录元数据，使原子替换在主机异常后也更可靠。"""
    directory = path.dirname(path.abspath(filename))
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def atomic_copy_file(source, destination):
    """复制文件到同目录临时路径，同步后再原子替换目标。"""
    destination = path.abspath(destination)
    os.makedirs(path.dirname(destination), exist_ok=True)
    temporary = f'{destination}.tmp-{os.getpid()}'
    try:
        shutil.copyfile(source, temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, destination)
        fsync_parent_directory(destination)
    finally:
        if path.exists(temporary):
            os.unlink(temporary)


def validate_output_paths():
    """拒绝会覆盖奖励权重、索引或其他输出文件的危险路径组合。"""
    control_cfg = config['control']
    grp_cfg = config['grp_transformer']
    named_paths = {
        '主 checkpoint': control_cfg['state_file'],
        '完成标记': control_cfg['completion_file'],
        '运行日志': control_cfg['run_log_file'],
        'GRP 源权重': grp_cfg['source_state_file'],
        'GRP 冻结快照': grp_cfg['state_file'],
        '数据索引': config['dataset']['file_index'],
    }
    resolved = {name: path.realpath(filename) for name, filename in named_paths.items()}
    reverse = {}
    for name, filename in resolved.items():
        if filename in reverse:
            raise ValueError(f'路径冲突: {reverse[filename]} 与 {name} 都指向 {filename}')
        reverse[filename] = name


def prepare_grp_snapshot(preflight):
    """正式训练前冻结 GRP 权重，保证数天内所有 worker 使用同一份模型。"""
    grp_cfg = config['grp_transformer']
    source_file = path.abspath(grp_cfg['source_state_file'])
    snapshot_file = path.abspath(grp_cfg['state_file'])
    expected_sha256 = grp_cfg['expected_sha256']

    source_sha256 = sha256_file(source_file)
    if source_sha256 != expected_sha256:
        raise RuntimeError(
            f'Transformer GRP 源权重摘要不匹配: expected={expected_sha256}, actual={source_sha256}'
        )
    if preflight:
        return source_file, source_sha256

    if not path.exists(snapshot_file):
        atomic_copy_file(source_file, snapshot_file)
    snapshot_sha256 = sha256_file(snapshot_file)
    if snapshot_sha256 != expected_sha256:
        raise RuntimeError(
            f'Transformer GRP 冻结快照摘要不匹配: expected={expected_sha256}, actual={snapshot_sha256}'
        )
    return snapshot_file, snapshot_sha256


def configure_file_logging(filename):
    """把长时间训练日志同时写入独立文件，便于中断后审计进度。"""
    filename = path.abspath(filename)
    os.makedirs(path.dirname(filename), exist_ok=True)

    handler = logging.FileHandler(filename, encoding='utf-8')
    handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)8s %(filename)28s:%(lineno)-4s %(message)s'
    ))
    logging.getLogger().addHandler(handler)
    return filename


def load_player_names(dataset_cfg):
    """读取可选的玩家白名单；空列表表示使用日志中的全部四位玩家。"""
    player_names = set()
    for filename in dataset_cfg['player_names_files']:
        with open(filename, encoding='utf-8') as f:
            player_names.update(filtered_trimmed_lines(f))
    return sorted(player_names)


def load_file_list(dataset_cfg):
    """加载已有文件索引；仅在指定普通 file_list 索引不存在时才新建。"""
    index_file = dataset_cfg['file_index']
    index_key = dataset_cfg.get('file_index_key', 'file_list')

    if path.exists(index_file):
        index = torch.load(index_file, weights_only=True, map_location='cpu')
        if index_key not in index:
            raise KeyError(f'文件索引 {index_file} 中不存在键 {index_key!r}')
        file_list = list(index[index_key])
    else:
        if index_key != 'file_list':
            raise FileNotFoundError(
                f'复用索引不存在，且不允许用 {index_key!r} 覆盖式新建: {index_file}'
            )
        logging.info('building offline file index: %s', index_file)
        file_list = []
        for pattern in dataset_cfg['globs']:
            file_list.extend(glob(pattern, recursive=True))
        file_list.sort(reverse=True)
        os.makedirs(path.dirname(path.abspath(index_file)), exist_ok=True)
        torch.save({'file_list': file_list}, index_file)

    if not file_list:
        raise RuntimeError('离线训练文件列表为空，不能把空迭代误判为 epoch 完成')
    validate_full_file_list(file_list, dataset_cfg)
    return file_list


def validate_full_file_list(file_list, dataset_cfg):
    """把索引与固化基准及配置 glob 做集合核验，防止截短或重复索引。"""
    expected_count = dataset_cfg['expected_file_count']
    if len(file_list) != expected_count:
        raise RuntimeError(
            f'训练索引数量不匹配: actual={len(file_list):,}, expected={expected_count:,}'
        )
    if not all(isinstance(filename, str) for filename in file_list):
        raise TypeError('训练索引中的路径必须全部是字符串')

    unique_files = set(file_list)
    if len(unique_files) != expected_count:
        raise RuntimeError(
            f'训练索引含重复路径: total={len(file_list):,}, unique={len(unique_files):,}'
        )

    expected_sha256 = dataset_cfg['expected_file_list_sha256']
    actual_sha256 = sha256_paths(file_list)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            f'训练索引摘要不匹配: expected={expected_sha256}, actual={actual_sha256}'
        )

    discovered_files = set()
    for pattern in dataset_cfg['globs']:
        discovered_files.update(iglob(pattern, recursive=True))
    quarantined_files = {
        filename
        for filename in unique_files - discovered_files
        if is_quarantined_dataset_file(filename)
    }
    discovered_logical_files = discovered_files | quarantined_files
    missing = next(iter(unique_files - discovered_logical_files), None)
    extra = next(iter(discovered_logical_files - unique_files), None)
    if (
        len(discovered_logical_files) != expected_count
        or missing is not None
        or extra is not None
    ):
        raise RuntimeError(
            '训练索引与配置 glob 的文件集合不一致: '
            f'index={len(unique_files):,}, glob={len(discovered_files):,}, '
            f'quarantined={len(quarantined_files):,}, '
            f'missing_from_glob={missing!r}, missing_from_index={extra!r}'
        )


def build_implementation_manifest():
    """记录会影响样本、奖励或参数更新语义的代码、扩展与运行时版本。"""
    module_dir = path.dirname(path.abspath(__file__))
    source_files = {
        'train_offline_grp_transformer.py': __file__,
        'dataloader_grp_transformer.py': path.join(module_dir, 'dataloader_grp_transformer.py'),
        'dataloader.py': path.join(module_dir, 'dataloader.py'),
        'model.py': path.join(module_dir, 'model.py'),
        'reward_calculator.py': path.join(module_dir, 'reward_calculator.py'),
        'lr_scheduler.py': path.join(module_dir, 'lr_scheduler.py'),
        'common.py': path.join(module_dir, 'common.py'),
        'libriichi_extension': libriichi.__file__,
    }
    return {
        'files': {
            name: {
                'path': path.abspath(filename),
                'sha256': sha256_file(filename),
            }
            for name, filename in source_files.items()
        },
        'runtime': {
            'python': sys.version,
            'torch': str(torch.__version__),
            'torch_cuda': torch.version.cuda,
            'cudnn': torch.backends.cudnn.version(),
            'libriichi': getattr(libriichi, '__version__', None),
            'libriichi_profile': getattr(libriichi, '__profile__', None),
        },
    }


def build_training_contract(
    player_names,
    grp_sha256,
    file_list_sha256,
    dataset_manifest_sha256,
    implementation_manifest,
):
    """提取所有会影响奖励、样本顺序或参数更新语义的配置。"""
    control_cfg = config['control']
    contract = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'control': {
            key: control_cfg[key]
            for key in (
                'version',
                'device',
                'enable_cudnn_benchmark',
                'enable_amp',
                'enable_compile',
                'compile_dynamic',
                'batch_size',
                'opt_step_every',
                'target_epochs',
                'epoch_seed',
            )
        },
        'dataset': dict(config['dataset']),
        'env': dict(config['env']),
        'resnet': dict(config['resnet']),
        'cql': dict(config['cql']),
        'aux': dict(config['aux']),
        'freeze_bn': dict(config['freeze_bn']),
        'optim': dict(config['optim']),
        'grp_transformer': dict(config['grp_transformer']),
        'player_names': player_names,
        'grp_transformer_sha256': grp_sha256,
        'file_list_sha256': file_list_sha256,
        'dataset_manifest_sha256': dataset_manifest_sha256,
        'implementation': implementation_manifest,
    }
    return contract, sha256_json(contract)


def validate_resume_training_contract(state, current_contract, current_sha256):
    """仅允许白名单 checkpoint 迁移到当前数据修复实现。"""
    saved_sha256 = state.get('training_contract_sha256')
    saved_contract = state.get('training_contract')
    if not isinstance(saved_contract, dict) or sha256_json(saved_contract) != saved_sha256:
        raise RuntimeError('checkpoint 内嵌训练契约与摘要不一致')
    if saved_sha256 == current_sha256:
        return
    migration = ALLOWED_IMPLEMENTATION_MIGRATIONS.get(saved_sha256)
    if migration is None:
        raise RuntimeError('关键训练配置已变化，拒绝在同一 epoch 中混用训练语义')

    saved_semantics = {
        key: value
        for key, value in saved_contract.items()
        if key != 'implementation'
    }
    current_semantics = {
        key: value
        for key, value in current_contract.items()
        if key != 'implementation'
    }
    if saved_semantics != current_semantics:
        raise RuntimeError('旧 checkpoint 在实现文件之外还包含其他训练语义变化')

    saved_implementation = saved_contract.get('implementation', {})
    current_implementation = current_contract.get('implementation', {})
    if saved_implementation.get('runtime') != current_implementation.get('runtime'):
        raise RuntimeError('训练实现迁移时运行时版本发生变化')
    saved_files = saved_implementation.get('files', {})
    current_files = current_implementation.get('files', {})
    if saved_files.keys() != current_files.keys():
        raise RuntimeError('训练实现迁移时文件集合发生变化')
    changed_files = {
        filename
        for filename in saved_files
        if saved_files[filename] != current_files[filename]
    }
    if changed_files != IMPLEMENTATION_MIGRATION_FILES:
        raise RuntimeError(
            '训练实现迁移包含未授权的文件变化: '
            f'{sorted(changed_files)}'
        )
    logging.warning(
        'migrating exact legacy training contract %s for %s; '
        'the next checkpoint will persist contract %s',
        saved_sha256,
        migration,
        current_sha256,
    )


def build_models_and_optimizer(
    device,
    resume,
    grp_sha256,
    file_list_sha256,
    dataset_manifest_sha256,
    training_contract,
    training_contract_sha256,
):
    """创建主模型与优化器，并按需恢复本专用入口生成的 checkpoint。"""
    control_cfg = config['control']
    optim_cfg = config['optim']
    version = control_cfg['version']

    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    mortal.freeze_bn(config['freeze_bn']['mortal'])

    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for module_name, module in model.named_modules():
            for name, param in module.named_parameters(prefix=module_name, recurse=False):
                params_dict[name] = param
                if isinstance(module, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))

    param_groups = [
        {'params': decay_params, 'weight_decay': optim_cfg['weight_decay']},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(
        param_groups,
        lr=1,
        weight_decay=0,
        betas=optim_cfg['betas'],
        eps=optim_cfg['eps'],
    )
    scheduler = LinearWarmUpCosineAnnealingLR(
        optimizer,
        **optim_cfg['scheduler'],
    )
    scaler = GradScaler(device.type, enabled=control_cfg['enable_amp'])

    progress = {
        'steps': 0,
        'samples_seen': 0,
        'completed_epochs': 0,
        'next_file_offset': 0,
        'epoch_started_at': datetime.now().timestamp(),
        'pending_batch': None,
    }

    state_file = control_cfg['state_file']
    if resume and path.exists(state_file):
        state = torch.load(state_file, weights_only=True, map_location='cpu')
        if state.get('schema_version') != CHECKPOINT_SCHEMA_VERSION:
            raise RuntimeError(f'不支持的离线 checkpoint schema: {state.get("schema_version")}')
        if state.get('grp_transformer_sha256') != grp_sha256:
            raise RuntimeError('Transformer GRP checkpoint 已变化，拒绝继续混合奖励来源的 epoch')
        if state.get('file_list_sha256') != file_list_sha256:
            raise RuntimeError('训练文件顺序已变化，拒绝从不兼容的文件偏移继续')
        if state.get('dataset_manifest_sha256') != dataset_manifest_sha256:
            raise RuntimeError('训练数据清单已变化，拒绝在同一 epoch 中混用数据内容')
        validate_resume_training_contract(
            state,
            training_contract,
            training_contract_sha256,
        )

        mortal.load_state_dict(state['mortal'], strict=True)
        dqn.load_state_dict(state['current_dqn'], strict=True)
        aux_net.load_state_dict(state['aux_net'], strict=True)
        optimizer.load_state_dict(state['optimizer'])
        scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        for key in progress:
            progress[key] = state[key]
        logging.info(
            'resumed checkpoint: steps=%s samples=%s next_file=%s completed_epochs=%s',
            progress['steps'],
            progress['samples_seen'],
            progress['next_file_offset'],
            progress['completed_epochs'],
        )

    if control_cfg['enable_compile']:
        for model in all_models:
            model.compile(dynamic=control_cfg['compile_dynamic'])

    logging.info('version: %s', version)
    logging.info('obs shape: %s', obs_shape(version))
    logging.info('mortal params: %s', f'{parameter_count(mortal):,}')
    logging.info('dqn params: %s', f'{parameter_count(dqn):,}')
    logging.info('aux params: %s', f'{parameter_count(aux_net):,}')
    return mortal, dqn, aux_net, optimizer, scheduler, scaler, progress


def atomic_torch_save(state, filename):
    """持久化并原子替换 checkpoint，避免暴露半写入文件。"""
    filename = path.abspath(filename)
    os.makedirs(path.dirname(filename), exist_ok=True)
    temporary = f'{filename}.tmp-{os.getpid()}'
    try:
        torch.save(state, temporary)
        with open(temporary, 'rb') as f:
            os.fsync(f.fileno())
        os.replace(temporary, filename)
        fsync_parent_directory(filename)
    finally:
        if path.exists(temporary):
            os.unlink(temporary)


def save_checkpoint(
    *,
    mortal,
    dqn,
    aux_net,
    optimizer,
    scheduler,
    scaler,
    progress,
    grp_sha256,
    file_list_sha256,
    dataset_manifest_sha256,
    training_contract,
    training_contract_sha256,
    file_count,
    reason,
):
    """保存可恢复训练状态，并记录奖励模型和数据顺序的不可变指纹。"""
    state = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'mortal': mortal.state_dict(),
        'current_dqn': dqn.state_dict(),
        'aux_net': aux_net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        **progress,
        'timestamp': datetime.now().timestamp(),
        'file_count': file_count,
        'expected_file_count': config['dataset']['expected_file_count'],
        'grp_model_type': 'transformer',
        'grp_transformer_state_file': path.abspath(config['grp_transformer']['state_file']),
        'grp_transformer_sha256': grp_sha256,
        'file_list_sha256': file_list_sha256,
        'dataset_manifest_sha256': dataset_manifest_sha256,
        'training_contract': training_contract,
        'training_contract_sha256': training_contract_sha256,
        'checkpoint_reason': reason,
        'best_perf': {'avg_rank': 4.0, 'avg_pt': -135.0},
        'trainer': 'train_offline_grp_transformer.py',
        'config': config,
    }
    atomic_torch_save(state, config['control']['state_file'])


def create_data_loader(file_list, player_names, seed, num_workers=None):
    """为一个可恢复文件分片创建 DataLoader，样本尾批也参与训练。"""
    dataset_cfg = config['dataset']
    if num_workers is None:
        num_workers = dataset_cfg['num_workers']

    file_data = FileDatasetsIter(
        version=config['control']['version'],
        file_list=list(file_list),
        pts=config['env']['pts'],
        file_batch_size=dataset_cfg['file_batch_size'],
        reserve_ratio=dataset_cfg['reserve_ratio'],
        player_names=player_names,
        num_epochs=1,
        enable_augmentation=False,
        augmented_first=False,
        grp_config=config['grp_transformer'],
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader_kwargs = {
        'dataset': file_data,
        'batch_size': config['control']['batch_size'],
        'drop_last': False,
        'num_workers': num_workers,
        'pin_memory': dataset_cfg['pin_memory'],
        'worker_init_fn': worker_init_fn if num_workers > 0 else None,
        'generator': generator,
    }
    if num_workers > 0:
        loader_kwargs['prefetch_factor'] = dataset_cfg['prefetch_factor']
        loader_kwargs['multiprocessing_context'] = dataset_cfg['multiprocessing_context']
    return DataLoader(**loader_kwargs)


def train_batch(
    batch,
    *,
    mortal,
    dqn,
    aux_net,
    optimizer,
    scheduler,
    scaler,
    device,
):
    """执行与原离线入口一致的 DQN、CQL 和下一顺位联合优化。"""
    obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks = batch
    batch_size = obs.shape[0]

    validation_rows = torch.arange(batch_size, device=masks.device)
    if not masks[validation_rows, actions].all():
        raise RuntimeError('数据集中存在不合法动作，拒绝继续训练')

    obs = obs.to(dtype=torch.float32, device=device, non_blocking=True)
    actions = actions.to(dtype=torch.int64, device=device, non_blocking=True)
    masks = masks.to(dtype=torch.bool, device=device, non_blocking=True)
    steps_to_done = steps_to_done.to(dtype=torch.int64, device=device, non_blocking=True)
    kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device, non_blocking=True)
    player_ranks = player_ranks.to(dtype=torch.int64, device=device, non_blocking=True)

    row_indices = torch.arange(batch_size, device=device)
    gamma = config['env']['gamma']
    q_target = (gamma ** steps_to_done * kyoku_rewards).to(torch.float32)
    optimizer.zero_grad(set_to_none=True)

    with torch.autocast(device.type, enabled=config['control']['enable_amp']):
        phi = mortal(obs)
        q_out = dqn(phi, masks)
        q = q_out[row_indices, actions]
        dqn_loss = 0.5 * nn.functional.mse_loss(q, q_target)
        cql_loss = q_out.logsumexp(-1).mean() - q.mean()
        next_rank_logits, = aux_net(phi)
        next_rank_loss = nn.functional.cross_entropy(next_rank_logits, player_ranks)
        loss = (
            dqn_loss
            + cql_loss * config['cql']['min_q_weight']
            + next_rank_loss * config['aux']['next_rank_weight']
        )

    scaler.scale(loss).backward()
    max_grad_norm = config['optim']['max_grad_norm']
    if max_grad_norm > 0:
        scaler.unscale_(optimizer)
        params = chain.from_iterable(group['params'] for group in optimizer.param_groups)
        clip_grad_norm_(params, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    metrics = torch.stack((
        loss.detach(),
        dqn_loss.detach(),
        cql_loss.detach(),
        next_rank_loss.detach(),
        q.detach().mean(),
        q_target.detach().mean(),
    ))
    return batch_size, metrics


def merge_partial_batch(pending_batch, batch, target_size):
    """跨文件分片合并尾 batch，返回至多一个完整 batch 和新的待合并余量。"""
    if pending_batch is None:
        combined = batch
    else:
        combined = tuple(
            torch.cat((pending, current), dim=0)
            for pending, current in zip(pending_batch, batch, strict=True)
        )
    combined_size = combined[0].shape[0]
    if combined_size < target_size:
        return None, combined

    ready = tuple(tensor[:target_size] for tensor in combined)
    if combined_size == target_size:
        return ready, None
    remainder = tuple(tensor[target_size:] for tensor in combined)
    return ready, remainder


def validate_completed_checkpoint(
    state_file,
    *,
    file_count,
    grp_sha256,
    file_list_sha256,
    dataset_manifest_sha256,
    training_contract_sha256,
):
    """回读并验证所有 epoch 完成不变量，不能只信任 completed_epochs 字段。"""
    state = torch.load(state_file, weights_only=True, map_location='cpu')
    expected = {
        'schema_version': CHECKPOINT_SCHEMA_VERSION,
        'completed_epochs': 1,
        'next_file_offset': file_count,
        'file_count': file_count,
        'expected_file_count': config['dataset']['expected_file_count'],
        'grp_transformer_sha256': grp_sha256,
        'file_list_sha256': file_list_sha256,
        'dataset_manifest_sha256': dataset_manifest_sha256,
        'training_contract_sha256': training_contract_sha256,
        'checkpoint_reason': 'epoch_completed',
    }
    for key, expected_value in expected.items():
        if state.get(key) != expected_value:
            raise RuntimeError(
                f'最终 checkpoint 不变量失败: {key}={state.get(key)!r}, '
                f'expected={expected_value!r}'
            )
    if state.get('steps', 0) <= 0 or state.get('samples_seen', 0) <= 0:
        raise RuntimeError('最终 checkpoint 没有有效的优化步骤或训练样本')
    if state.get('pending_batch') is not None:
        raise RuntimeError('最终 checkpoint 仍含未训练的尾 batch')
    return state, sha256_file(state_file)


def write_completion_marker(
    checkpoint_sha256,
    grp_sha256,
    file_list_sha256,
    dataset_manifest_sha256,
    training_contract_sha256,
    progress,
    file_count,
):
    """仅在最终 checkpoint 回读通过后写入 epoch 完成标记。"""
    marker_file = path.abspath(config['control']['completion_file'])
    os.makedirs(path.dirname(marker_file), exist_ok=True)
    marker = {
        'status': 'completed',
        'completed_epochs': progress['completed_epochs'],
        'completed_at': datetime.now().astimezone().isoformat(),
        'steps': progress['steps'],
        'samples_seen': progress['samples_seen'],
        'file_count': file_count,
        'expected_file_count': config['dataset']['expected_file_count'],
        'checkpoint': path.abspath(config['control']['state_file']),
        'checkpoint_sha256': checkpoint_sha256,
        'grp_transformer_sha256': grp_sha256,
        'file_list_sha256': file_list_sha256,
        'dataset_manifest_sha256': dataset_manifest_sha256,
        'training_contract_sha256': training_contract_sha256,
    }
    temporary = f'{marker_file}.tmp-{os.getpid()}'
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(marker, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, marker_file)
        fsync_parent_directory(marker_file)
    finally:
        if path.exists(temporary):
            os.unlink(temporary)


def invalidate_stale_completion_marker():
    """未完成训练启动时保留并改名旧标记，避免它继续冒充当前完成证据。"""
    marker_file = path.abspath(config['control']['completion_file'])
    if not path.exists(marker_file):
        return
    timestamp = datetime.now().strftime('%Y%m%d-%H%M%S-%f')
    stale_file = f'{marker_file}.stale-{timestamp}'
    os.replace(marker_file, stale_file)
    fsync_parent_directory(marker_file)
    logging.warning('moved stale completion marker to %s', stale_file)


def run(preflight_files=0, disable_compile=False):
    """运行全量首个离线 epoch，或执行不落盘的真实数据预检。"""
    control_cfg = config['control']
    dataset_cfg = config['dataset']
    grp_cfg = config['grp_transformer']

    if control_cfg.get('online', False):
        raise ValueError('该新增入口只允许 control.online=false 的离线训练')
    if dataset_cfg['num_epochs'] != 1 or dataset_cfg['enable_augmentation']:
        raise ValueError('首个完整 epoch 要求 num_epochs=1 且 enable_augmentation=false')
    if control_cfg['opt_step_every'] != 1:
        raise ValueError('该可恢复入口当前要求 opt_step_every=1')
    if control_cfg['target_epochs'] != 1:
        raise ValueError('当前目标固定为完成第一个 epoch，target_epochs 必须为 1')
    if disable_compile and preflight_files == 0:
        raise ValueError('--disable-compile 只允许与 --preflight-files 一起使用')
    validate_output_paths()
    if preflight_files == 0:
        invalidate_stale_completion_marker()

    device = torch.device(control_cfg['device'])
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'配置要求 {device}，但当前 CUDA 不可用')
    torch.backends.cudnn.benchmark = control_cfg['enable_cudnn_benchmark']
    torch.set_float32_matmul_precision('high')

    active_grp_file, grp_sha256 = prepare_grp_snapshot(preflight_files > 0)
    grp_cfg['state_file'] = active_grp_file
    file_list = load_file_list(dataset_cfg)
    random.Random(control_cfg['epoch_seed']).shuffle(file_list)
    if preflight_files > 0:
        file_list = file_list[:preflight_files]
    file_count = len(file_list)
    shard_size = dataset_cfg['file_shard_size']
    stop_after_file_offset = control_cfg.get('stop_after_file_offset')
    if stop_after_file_offset is not None:
        if isinstance(stop_after_file_offset, bool) or not isinstance(stop_after_file_offset, int):
            raise ValueError('stop_after_file_offset 必须是整数文件偏移')
        if not 0 < stop_after_file_offset <= file_count:
            raise ValueError(
                f'stop_after_file_offset 越界: {stop_after_file_offset}/{file_count}'
            )
        if stop_after_file_offset != file_count and stop_after_file_offset % shard_size != 0:
            raise ValueError('stop_after_file_offset 必须位于 checkpoint 分片边界')
    file_list_sha256 = sha256_paths(file_list)
    dataset_manifest_sha256, shard_manifest_sha256s = build_dataset_manifests(
        file_list,
        shard_size,
    )
    player_names = load_player_names(dataset_cfg)
    implementation_manifest = build_implementation_manifest()
    implementation_manifest_sha256 = sha256_json(implementation_manifest)
    training_contract, training_contract_sha256 = build_training_contract(
        player_names,
        grp_sha256,
        file_list_sha256,
        dataset_manifest_sha256,
        implementation_manifest,
    )

    logging.info('device: %s', device)
    logging.info('Transformer GRP file: %s', active_grp_file)
    logging.info('Transformer GRP SHA256: %s', grp_sha256)
    logging.info('file list size: %s', f'{file_count:,}')
    logging.info('file list SHA256: %s', file_list_sha256)
    logging.info('dataset manifest SHA256: %s', dataset_manifest_sha256)
    logging.info('implementation manifest SHA256: %s', implementation_manifest_sha256)
    logging.info('training contract SHA256: %s', training_contract_sha256)
    logging.info('player filter size: %s', len(player_names))

    state_file = control_cfg['state_file']
    if preflight_files == 0 and path.exists(state_file):
        existing = torch.load(state_file, weights_only=True, map_location='cpu')
        if existing.get('completed_epochs', 0) >= 1:
            verified, checkpoint_sha256 = validate_completed_checkpoint(
                state_file,
                file_count=file_count,
                grp_sha256=grp_sha256,
                file_list_sha256=file_list_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                training_contract_sha256=training_contract_sha256,
            )
            write_completion_marker(
                checkpoint_sha256,
                grp_sha256,
                file_list_sha256,
                dataset_manifest_sha256,
                training_contract_sha256,
                verified,
                file_count,
            )
            logging.info(
                'EPOCH_COMPLETED_VERIFIED: files=%s samples=%s steps=%s checkpoint_sha256=%s',
                f'{file_count:,}',
                f'{verified["samples_seen"]:,}',
                f'{verified["steps"]:,}',
                checkpoint_sha256,
            )
            return

    original_compile = control_cfg['enable_compile']
    if disable_compile:
        control_cfg['enable_compile'] = False
    try:
        components = build_models_and_optimizer(
            device,
            resume=preflight_files == 0,
            grp_sha256=grp_sha256,
            file_list_sha256=file_list_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            training_contract=training_contract,
            training_contract_sha256=training_contract_sha256,
        )
    finally:
        control_cfg['enable_compile'] = original_compile

    mortal, dqn, aux_net, optimizer, scheduler, scaler, progress = components

    if preflight_files > 0:
        logging.info('PRECHECK only: %s files; no checkpoint or TensorBoard will be written', preflight_files)
        loader = create_data_loader(file_list, player_names, control_cfg['epoch_seed'])
        loader_iter = iter(loader)
        try:
            batch = next(loader_iter)
            for _ in loader_iter:
                pass
        finally:
            del loader_iter, loader
            gc.collect()
        batch_size, metric_tensor = train_batch(
            batch,
            mortal=mortal,
            dqn=dqn,
            aux_net=aux_net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        if not torch.isfinite(metric_tensor).all():
            raise RuntimeError(f'真实数据预检得到非有限指标: {metric_tensor}')
        metric_values = metric_tensor.cpu().tolist()
        metrics = {'batch_size': batch_size, **dict(zip(METRIC_NAMES, metric_values, strict=True))}
        logging.info('PRECHECK_SUCCESS: %s', metrics)
        return

    start_offset = progress['next_file_offset']
    if not 0 <= start_offset <= file_count:
        raise RuntimeError(f'checkpoint 文件偏移越界: {start_offset}/{file_count}')
    if stop_after_file_offset is not None and start_offset >= stop_after_file_offset:
        logging.info(
            'TRAINING_ALREADY_PAUSED_AT_CHECKPOINT: files=%s steps=%s samples=%s',
            f'{start_offset:,}',
            f'{progress["steps"]:,}',
            f'{progress["samples_seen"]:,}',
        )
        return

    writer = SummaryWriter(control_cfg['tensorboard_dir'])
    log_every_steps = control_cfg['log_every_steps']
    target_batch_size = control_cfg['batch_size']
    running_batches = 0
    running_values = torch.zeros(len(METRIC_NAMES), dtype=torch.float64, device=device)
    started = time.perf_counter()
    run_start_samples = progress['samples_seen']
    current_shard_start = start_offset

    def flush_metrics(force=False):
        """按日志周期一次性同步 GPU 指标，避免每个 batch 触发六次同步。"""
        nonlocal running_batches
        if running_batches == 0:
            return
        if not force and progress['steps'] % log_every_steps != 0:
            return
        elapsed = time.perf_counter() - started
        run_samples = progress['samples_seen'] - run_start_samples
        averages = (running_values / running_batches).cpu().tolist()
        if not all(torch.isfinite(torch.tensor(averages))):
            raise RuntimeError('训练得到非有限损失，拒绝继续并写入后续断点')
        values = dict(zip(METRIC_NAMES, averages, strict=True))
        throughput = run_samples / elapsed
        logging.info(
            'steps=%s samples=%s files=%s/%s loss=%.6f samples/s=%.1f',
            f'{progress["steps"]:,}',
            f'{progress["samples_seen"]:,}',
            f'{current_shard_start:,}',
            f'{file_count:,}',
            values['loss'],
            throughput,
        )
        for key in ('loss', 'dqn_loss', 'cql_loss', 'next_rank_loss'):
            writer.add_scalar(f'loss/{key}', values[key], progress['steps'])
        writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], progress['steps'])
        writer.add_scalar('q/predicted_mean', values['q_mean'], progress['steps'])
        writer.add_scalar('q/target_mean', values['q_target_mean'], progress['steps'])
        writer.add_scalar('throughput/samples_per_second', throughput, progress['steps'])
        writer.flush()
        running_values.zero_()
        running_batches = 0

    def optimize(batch):
        """训练一个已整理 batch，并统一更新断点进度与延迟指标。"""
        nonlocal running_batches
        batch_size, metric_tensor = train_batch(
            batch,
            mortal=mortal,
            dqn=dqn,
            aux_net=aux_net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            device=device,
        )
        progress['steps'] += 1
        progress['samples_seen'] += batch_size
        running_values.add_(metric_tensor.to(torch.float64))
        running_batches += 1
        flush_metrics()

    try:
        for shard_start in range(start_offset, file_count, shard_size):
            current_shard_start = shard_start
            shard_end = min(shard_start + shard_size, file_count)
            shard_files = file_list[shard_start:shard_end]
            shard_index = shard_start // shard_size
            expected_shard_manifest = shard_manifest_sha256s[shard_index]
            before_shard_manifest = sha256_dataset_manifest(shard_files)
            if before_shard_manifest != expected_shard_manifest:
                raise RuntimeError(
                    f'训练分片 {shard_index} 在读取前已变化: '
                    f'expected={expected_shard_manifest}, actual={before_shard_manifest}'
                )
            loader = create_data_loader(
                shard_files,
                player_names,
                control_cfg['epoch_seed'] + shard_start,
            )

            for batch in loader:
                batch_size = batch[0].shape[0]
                if batch_size == target_batch_size:
                    optimize(batch)
                    continue
                if batch_size > target_batch_size:
                    raise RuntimeError(f'DataLoader 返回超大 batch: {batch_size}>{target_batch_size}')
                ready, progress['pending_batch'] = merge_partial_batch(
                    progress['pending_batch'],
                    batch,
                    target_batch_size,
                )
                if ready is not None:
                    optimize(ready)

            after_shard_manifest = sha256_dataset_manifest(shard_files)
            if after_shard_manifest != expected_shard_manifest:
                raise RuntimeError(
                    f'训练分片 {shard_index} 在读取期间发生变化: '
                    f'expected={expected_shard_manifest}, actual={after_shard_manifest}'
                )
            del loader
            gc.collect()
            flush_metrics(force=True)
            progress['next_file_offset'] = shard_end
            save_checkpoint(
                mortal=mortal,
                dqn=dqn,
                aux_net=aux_net,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                progress=progress,
                grp_sha256=grp_sha256,
                file_list_sha256=file_list_sha256,
                dataset_manifest_sha256=dataset_manifest_sha256,
                training_contract=training_contract,
                training_contract_sha256=training_contract_sha256,
                file_count=file_count,
                reason='file_shard_completed',
            )
            logging.info(
                'shard checkpoint saved: files=%s/%s (%.3f%%), steps=%s, samples=%s',
                f'{shard_end:,}',
                f'{file_count:,}',
                shard_end / file_count * 100,
                f'{progress["steps"]:,}',
                f'{progress["samples_seen"]:,}',
            )
            if stop_after_file_offset is not None and shard_end >= stop_after_file_offset:
                logging.info(
                    'TRAINING_PAUSED_AT_CHECKPOINT: files=%s steps=%s samples=%s',
                    f'{shard_end:,}',
                    f'{progress["steps"]:,}',
                    f'{progress["samples_seen"]:,}',
                )
                return

        if progress['pending_batch'] is not None:
            optimize(progress['pending_batch'])
            progress['pending_batch'] = None
        flush_metrics(force=True)
        final_dataset_manifest = sha256_dataset_manifest(file_list)
        if final_dataset_manifest != dataset_manifest_sha256:
            raise RuntimeError(
                '完成前数据清单复核失败: '
                f'expected={dataset_manifest_sha256}, actual={final_dataset_manifest}'
            )
        final_implementation_manifest = build_implementation_manifest()
        if final_implementation_manifest != implementation_manifest:
            raise RuntimeError('训练期间实现代码、扩展或运行时版本发生变化，拒绝标记 epoch 完成')
        final_grp_sha256 = sha256_file(active_grp_file)
        if final_grp_sha256 != grp_sha256:
            raise RuntimeError(
                f'训练期间 Transformer GRP 快照发生变化: '
                f'expected={grp_sha256}, actual={final_grp_sha256}'
            )
        if progress['steps'] <= 0 or progress['samples_seen'] <= 0:
            raise RuntimeError('没有完成任何优化步骤，拒绝写入 epoch 完成状态')
        progress['completed_epochs'] = 1
        progress['next_file_offset'] = file_count
        save_checkpoint(
            mortal=mortal,
            dqn=dqn,
            aux_net=aux_net,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            progress=progress,
            grp_sha256=grp_sha256,
            file_list_sha256=file_list_sha256,
            dataset_manifest_sha256=dataset_manifest_sha256,
            training_contract=training_contract,
            training_contract_sha256=training_contract_sha256,
            file_count=file_count,
            reason='epoch_completed',
        )
    finally:
        writer.close()

    verified, checkpoint_sha256 = validate_completed_checkpoint(
        state_file,
        file_count=file_count,
        grp_sha256=grp_sha256,
        file_list_sha256=file_list_sha256,
        dataset_manifest_sha256=dataset_manifest_sha256,
        training_contract_sha256=training_contract_sha256,
    )
    write_completion_marker(
        checkpoint_sha256,
        grp_sha256,
        file_list_sha256,
        dataset_manifest_sha256,
        training_contract_sha256,
        verified,
        file_count,
    )
    logging.info(
        'EPOCH_COMPLETED: files=%s samples=%s steps=%s checkpoint_sha256=%s',
        f'{file_count:,}',
        f'{verified["samples_seen"]:,}',
        f'{verified["steps"]:,}',
        checkpoint_sha256,
    )


def parse_args():
    """解析开发预检参数；不传参数时执行并恢复全量训练。"""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--preflight-files',
        type=int,
        default=0,
        help='只用前 N 个真实日志执行一个不落盘的训练 batch',
    )
    parser.add_argument(
        '--disable-compile',
        action='store_true',
        help='预检时跳过 torch.compile；全量训练不建议使用',
    )
    return parser.parse_args()


def main():
    """配置日志后启动训练；中断会保留最近文件分片 checkpoint 并返回非零状态。"""
    args = parse_args()
    if args.preflight_files < 0:
        raise ValueError('--preflight-files 不能为负数')
    if args.preflight_files == 0:
        lock_file = f'{config["control"]["state_file"]}.lock'
        with training_lock(lock_file):
            log_file = configure_file_logging(config['control']['run_log_file'])
            logging.info('run log: %s', log_file)
            run(
                preflight_files=0,
                disable_compile=args.disable_compile,
            )
    else:
        run(
            preflight_files=args.preflight_files,
            disable_compile=args.disable_compile,
        )


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        logging.exception('训练被中断；下次将从最近完成的文件分片继续')
        sys.exit(130)
