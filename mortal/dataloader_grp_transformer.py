import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from config import config
from dataloader import FileDatasetsIter as BaseFileDatasetsIter
from dataloader import worker_init_fn
from model import GRP_Transformer
from reward_calculator import RewardCalculator


REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOTS = (
    REPOSITORY_ROOT / 'data' / 'grp' / 'train',
    REPOSITORY_ROOT / 'data' / 'grp' / 'val',
    REPOSITORY_ROOT / 'data' / 'grp' / 'test',
)
DATASET_QUARANTINE_ROOT = REPOSITORY_ROOT / 'data' / 'grp' / 'quarantine'
QUARANTINABLE_ERROR_MARKERS = (
    'empty or invalid game log',
    'failed to fill whole buffer',
    'failed to parse log',
    'invalid gzip header',
    'invalid log:',
    'invalid utf-8',
    'on event ',
    'stream did not contain valid utf-8',
    'corrupt deflate stream',
    'rule violation:',
    'unexpected end of file',
)


def dataset_quarantine_path(filename):
    """返回训练、验证或测试日志在统一隔离目录中的对应路径。"""
    source = Path(filename).resolve(strict=False)
    for dataset_root in DATASET_ROOTS:
        try:
            relative = source.relative_to(dataset_root)
        except ValueError:
            continue
        return DATASET_QUARANTINE_ROOT / dataset_root.name / relative
    return None


def dataset_file_stat(filename):
    """读取原文件元数据；文件已隔离时读取其保留元数据。"""
    source = Path(filename).resolve(strict=False)
    try:
        return source.stat()
    except FileNotFoundError as error:
        quarantined = dataset_quarantine_path(filename)
        if quarantined is not None and quarantined.is_file():
            return quarantined.stat()
        raise error


def is_quarantined_dataset_file(filename):
    """判断索引路径是否已被移动到隔离目录。"""
    source = Path(filename).resolve(strict=False)
    quarantined = dataset_quarantine_path(filename)
    return not source.exists() and quarantined is not None and quarantined.is_file()


def _sha256_file(filename):
    digest = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory):
    """同步目录项，保证隔离移动在主机异常后仍尽量可恢复。"""
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_quarantine_metadata(source, target, error, file_stat, file_sha256):
    metadata_file = Path(f'{target}.quarantine.json')
    temporary = Path(f'{metadata_file}.tmp-{os.getpid()}')
    metadata = {
        'schema_version': 1,
        'quarantined_at': datetime.now().astimezone().isoformat(),
        'original_path': str(source),
        'quarantine_path': str(target),
        'size': file_stat.st_size,
        'mtime_ns': file_stat.st_mtime_ns,
        'sha256': file_sha256,
        'error': str(error),
    }
    try:
        with open(temporary, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temporary, metadata_file)
        _fsync_directory(target.parent)
    finally:
        if temporary.exists():
            temporary.unlink()


def quarantine_dataset_file(filename, error):
    """不覆盖地隔离确定损坏的日志，并允许并发调用幂等完成。"""
    source = Path(filename).resolve(strict=False)
    target = dataset_quarantine_path(filename)
    if target is None:
        raise RuntimeError(f'拒绝隔离训练/验证/测试目录外的文件: {source}')
    if not source.exists():
        if not target.is_file():
            raise FileNotFoundError(f'待隔离数据不存在: {source}')

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        try:
            os.link(source, target, follow_symlinks=False)
        except FileExistsError:
            if not target.is_file():
                raise FileExistsError(f'隔离目标已存在，拒绝覆盖: {target}')
        except FileNotFoundError:
            if not target.is_file():
                raise

    if source.exists():
        if not os.path.samefile(source, target):
            raise FileExistsError(f'隔离目标已存在且内容来源不同，拒绝覆盖: {target}')
        try:
            source.unlink()
        except FileNotFoundError:
            pass

    _fsync_directory(source.parent)
    if source.parent != target.parent:
        _fsync_directory(target.parent)

    file_stat = target.stat()
    file_sha256 = _sha256_file(target)
    try:
        _write_quarantine_metadata(source, target, error, file_stat, file_sha256)
    except Exception:
        logging.exception('写入隔离元数据失败，但数据文件已移动到 %s', target)
    return target


def is_quarantinable_dataset_error(error):
    """仅把明确的数据格式、压缩或规则错误视为可隔离错误。"""
    message = str(error).lower()
    return 'error when reading ' in message and any(
        marker in message
        for marker in QUARANTINABLE_ERROR_MARKERS
    )


class FileDatasetsIter(BaseFileDatasetsIter):
    """使用 Transformer GRP 计算局间奖励，其余样本生成逻辑复用原加载器。"""

    def __init__(self, *args, grp_config=None, **kwargs):
        """保存可随 spawn worker 传递的 GRP 配置快照。"""
        super().__init__(*args, **kwargs)
        self.invalid_action_samples = 0
        source = grp_config or config['grp_transformer']
        self.grp_config = {
            'state_file': source['state_file'],
            'expected_sha256': source.get('expected_sha256'),
            'network': dict(source['network']),
        }

    def initialize_grp_model(self):
        """在当前进程加载冻结 GRP，供训练 worker 和预处理器复用。"""
        grp_cfg = self.grp_config
        state_file = grp_cfg['state_file']

        expected_sha256 = grp_cfg.get('expected_sha256')
        if expected_sha256:
            digest = hashlib.sha256()
            with open(state_file, 'rb') as f:
                for chunk in iter(lambda: f.read(1024 * 1024), b''):
                    digest.update(chunk)
            actual_sha256 = digest.hexdigest()
            if actual_sha256 != expected_sha256:
                raise RuntimeError(
                    'Transformer GRP worker 加载的权重摘要不一致: '
                    f'expected={expected_sha256}, actual={actual_sha256}'
                )

        self.grp = GRP_Transformer(**grp_cfg['network'])
        state = torch.load(
            state_file,
            weights_only=True,
            map_location=torch.device('cpu'),
        )
        if 'model' not in state:
            raise KeyError(f'Transformer GRP checkpoint 缺少 model 字段: {state_file}')
        self.grp.load_state_dict(state['model'], strict=True)
        self.grp.eval()
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        logging.info(
            'Transformer GRP loaded in data worker: %s (steps=%s)',
            state_file,
            state.get('steps', 'unknown'),
        )

    def build_iter(self):
        """在 DataLoader worker 内加载 GRP，避免跨进程传递 PyTorch 模型。"""
        self.initialize_grp_model()

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def _build_shared_grp_targets(self, grp_feature, rank_by_player, final_scores):
        """一次计算同一日志四个玩家共享的局间奖励与顺位序列。"""
        rank_by_player = np.fromiter(rank_by_player, dtype=np.int64, count=4)
        final_scores = np.asarray(final_scores, dtype=np.int64)

        with torch.inference_mode():
            matrix = self.reward_calc.calc_grp(grp_feature)
            expected_pts = matrix @ self.reward_calc.pts
            final_expected_pts = self.reward_calc.pts[
                torch.as_tensor(rank_by_player, dtype=torch.int64)
            ]
            expected_pts = torch.cat((expected_pts, final_expected_pts.unsqueeze(0)))
            kyoku_rewards_by_player = torch.diff(expected_pts, dim=0).cpu().numpy()

        scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
        rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
        return rank_by_player, final_scores, kyoku_rewards_by_player, rank_by_player_seq

    def _load_files_with_quarantine(self, file_list):
        """批量加载失败时逐文件定位并隔离确定损坏的日志。"""
        active_files = []
        for filename in file_list:
            if is_quarantined_dataset_file(filename):
                logging.warning('skipped quarantined dataset file: %s', filename)
                continue
            active_files.append(filename)
        if not active_files:
            return [], []

        try:
            return active_files, self.loader.load_gz_log_files(active_files)
        except RuntimeError:
            loaded_files = []
            loaded_data = []
            for filename in active_files:
                try:
                    file_data = self.loader.load_gz_log_files([filename])
                except RuntimeError as error:
                    if is_quarantined_dataset_file(filename):
                        logging.warning('skipped quarantined dataset file: %s', filename)
                        continue
                    if not is_quarantinable_dataset_error(error):
                        raise
                    target = quarantine_dataset_file(filename, error)
                    logging.warning(
                        'quarantined invalid dataset file: source=%s target=%s error=%s',
                        filename,
                        target,
                        str(error).replace('\n', ' | '),
                    )
                    continue
                if len(file_data) != 1:
                    raise RuntimeError(
                        f'单文件加载返回数量异常: file={filename}, actual={len(file_data)}'
                    )
                loaded_files.append(filename)
                loaded_data.append(file_data[0])
            return loaded_files, loaded_data

    def iter_file_samples(self, filename, file_games):
        """把单个牌谱转换为带来源定位信息的合法训练样本。"""
        shared = None
        shared_feature = None

        for game in file_games:
            obs = game.take_obs()
            if self.oracle:
                invisible_obs = game.take_invisible_obs()
            actions = game.take_actions()
            masks = game.take_masks()
            at_kyoku = game.take_at_kyoku()
            dones = game.take_dones()
            apply_gamma = game.take_apply_gamma()

            grp = game.take_grp()
            player_id = game.take_player_id()
            grp_feature = grp.take_feature()
            rank_by_player = grp.take_rank_by_player()
            final_scores = grp.take_final_scores()

            if shared is None:
                shared_feature = grp_feature.copy()
                shared = self._build_shared_grp_targets(
                    grp_feature,
                    rank_by_player,
                    final_scores,
                )
            else:
                cached_ranks, cached_scores, _, _ = shared
                current_ranks = np.fromiter(rank_by_player, dtype=np.int64, count=4)
                current_scores = np.asarray(final_scores, dtype=np.int64)
                if (
                    not np.array_equal(grp_feature, shared_feature)
                    or not np.array_equal(current_ranks, cached_ranks)
                    or not np.array_equal(current_scores, cached_scores)
                ):
                    raise RuntimeError('同一日志的玩家视角包含不一致的 GRP 数据')

            _, _, kyoku_rewards_by_player, rank_by_player_seq = shared
            kyoku_rewards = kyoku_rewards_by_player[:, player_id]
            player_ranks = rank_by_player_seq[:, player_id]
            game_size = len(obs)
            if len(kyoku_rewards) < at_kyoku[-1] + 1:
                raise RuntimeError('GRP 奖励序列短于训练样本所需的局数')

            steps_to_done = np.zeros(game_size, dtype=np.int64)
            for i in reversed(range(game_size)):
                if not dones[i]:
                    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

            for i in range(game_size):
                action = int(actions[i])
                mask = masks[i]
                if action < 0 or action >= len(mask) or not bool(mask[action]):
                    self.invalid_action_samples += 1
                    logging.warning(
                        'skipped invalid-action sample: file=%s player_id=%s '
                        'sample_idx=%s action=%s legal_actions=%s',
                        filename,
                        player_id,
                        i,
                        action,
                        np.flatnonzero(mask).tolist(),
                    )
                    continue
                entry = [
                    obs[i],
                    action,
                    mask,
                    steps_to_done[i],
                    kyoku_rewards[at_kyoku[i]],
                    player_ranks[at_kyoku[i] + 1],
                ]
                if self.oracle:
                    entry.insert(1, invisible_obs[i])
                yield entry, int(player_id), i

    def populate_buffer(self, file_list):
        """加载日志，并让同一日志的所有玩家视角复用 Transformer 前向结果。"""
        loaded_files, data = self._load_files_with_quarantine(file_list)
        for filename, file_games in zip(loaded_files, data, strict=True):
            for entry, _, _ in self.iter_file_samples(filename, file_games):
                self.buffer.append(entry)


__all__ = [
    'FileDatasetsIter',
    'dataset_file_stat',
    'dataset_quarantine_path',
    'is_quarantined_dataset_file',
    'quarantine_dataset_file',
    'worker_init_fn',
]
