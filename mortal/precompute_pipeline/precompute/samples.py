"""从单份牌谱构造冻结 Transformer-GRP 六元样本。"""

from pathlib import Path

import numpy as np
import torch

from libriichi.dataset import GameplayLoader
from model import GRP_Transformer
from precompute_pipeline.common.codec import ACTION_SPACE, OBS_PLANES, OBS_TILES, SampleBatch
from precompute_pipeline.precompute.progress import StageTimings


def _stable_ranks(scores):
    return (-scores).argsort(-1, kind='stable').argsort(-1, kind='stable')


def _as_int_vector(value, count):
    if isinstance(value, (bytes, bytearray, memoryview)):
        array = np.frombuffer(value, dtype=np.uint8)
    else:
        array = np.fromiter(value, dtype=np.int64, count=count)
    if array.shape != (count,):
        raise ValueError(f'预期 {count} 个整数，实际 shape={array.shape}')
    return array.astype(np.int64, copy=False)


def build_shared_targets(
    grp_feature,
    *,
    rank_by_player,
    final_scores,
    rank_matrix,
    pts,
):
    """将各局边界的顺位概率转换为四家局奖励和真实顺位序列。"""
    grp_feature = np.asarray(grp_feature, dtype=np.float64)
    rank_by_player = np.asarray(rank_by_player, dtype=np.int64)
    final_scores = np.asarray(final_scores, dtype=np.int64)
    rank_matrix = np.asarray(rank_matrix, dtype=np.float64)
    pts = np.asarray(pts, dtype=np.float64)
    kyoku_count = len(grp_feature)
    if grp_feature.shape != (kyoku_count, 7):
        raise ValueError(f'GRP feature shape 不兼容: {grp_feature.shape}')
    if rank_matrix.shape != (kyoku_count, 4, 4):
        raise ValueError(f'GRP rank matrix shape 不兼容: {rank_matrix.shape}')
    if rank_by_player.shape != (4,) or sorted(rank_by_player.tolist()) != [0, 1, 2, 3]:
        raise ValueError('终局顺位必须是 0..3 的排列')
    if final_scores.shape != (4,) or pts.shape != (4,):
        raise ValueError('终局分数和名次效用都必须含四个值')
    if not np.isfinite(rank_matrix).all() or not np.isfinite(pts).all():
        raise ValueError('GRP 概率或名次效用含 NaN/Inf')

    expected_pts = rank_matrix @ pts
    final_expected_pts = pts[rank_by_player]
    expected_pts = np.concatenate((expected_pts, final_expected_pts[None, :]), axis=0)
    rewards = np.diff(expected_pts, axis=0)
    score_sequence = np.concatenate((grp_feature[:, 3:] * 10_000, final_scores[None, :]))
    rank_sequence = _stable_ranks(score_sequence)
    return rewards.astype(np.float64, copy=False), rank_sequence.astype(np.uint8)


def _empty_batch():
    return SampleBatch(
        obs=np.empty((0, OBS_PLANES, OBS_TILES), dtype=np.float32),
        actions=np.empty(0, dtype=np.uint8),
        masks=np.empty((0, ACTION_SPACE), dtype=np.bool_),
        steps_to_done=np.empty(0, dtype=np.uint16),
        rewards=np.empty(0, dtype=np.float64),
        ranks=np.empty(0, dtype=np.uint8),
    )


class SampleGenerator:
    """持有冻结奖励模型，并按文件生成标准六元样本。"""

    def __init__(
        self,
        *,
        state_file,
        d_model,
        nhead,
        num_layers,
        max_seq_len,
        pts,
    ):
        self.timings = StageTimings()
        self.max_seq_len = int(max_seq_len)
        self.pts = np.asarray(pts, dtype=np.float64)
        if self.pts.shape != (4,) or not np.isfinite(self.pts).all():
            raise ValueError('pts 必须是四个有限数')
        self.grp = GRP_Transformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            max_seq_len=max_seq_len,
        ).cpu().eval()
        state = torch.load(Path(state_file), weights_only=True, map_location='cpu')
        if not isinstance(state, dict) or not isinstance(state.get('model'), dict):
            raise ValueError('GRP checkpoint 缺少 model state_dict')
        self.grp.load_state_dict(state['model'], strict=True)
        self.loader = GameplayLoader(
            4,
            oracle=False,
            player_names=[],
            excludes=[],
            trust_seed=False,
            always_include_kan_select=True,
            augmented=False,
        )

    def _predict_rank_matrix(self, grp_feature):
        kyoku_count = len(grp_feature)
        if not 0 < kyoku_count <= self.max_seq_len:
            raise ValueError(
                f'牌谱局数 {kyoku_count} 超出奖励模型容量 1..{self.max_seq_len}'
            )
        prefixes = [
            torch.as_tensor(grp_feature[:end], dtype=torch.float64)
            for end in range(1, kyoku_count + 1)
        ]
        with torch.inference_mode():
            logits = self.grp(prefixes)
            matrix = self.grp.calc_matrix(logits)
        return matrix.cpu().numpy()

    def load_files(self, filenames):
        """由 Rust 并行读取一批文件，再按输入顺序逐份计算奖励。"""
        filenames = [str(Path(filename)) for filename in filenames]
        with self.timings.measure('rust'):
            loaded = self.loader.load_gz_log_files(filenames)
        if len(loaded) != len(filenames):
            raise RuntimeError(
                f'GameplayLoader 返回文件数错误: expected={len(filenames)}, actual={len(loaded)}'
            )
        return [self._games_to_batch(games) for games in loaded]

    def load_file(self, filename):
        """读取一份 gzip 牌谱，按玩家与原决策顺序返回六元样本。"""
        return self.load_files([filename])[0]

    def _games_to_batch(self, games):
        if not games:
            return _empty_batch()

        shared_feature = None
        shared_ranks = None
        shared_scores = None
        rewards_by_player = None
        ranks_by_player = None
        output = []
        for game in games:
            obs = game.take_obs()
            actions = game.take_actions()
            masks = game.take_masks()
            at_kyoku = game.take_at_kyoku()
            dones = game.take_dones()
            apply_gamma = game.take_apply_gamma()
            grp = game.take_grp()
            player_id = int(game.take_player_id())
            feature = np.asarray(grp.take_feature(), dtype=np.float64)
            final_ranks = _as_int_vector(grp.take_rank_by_player(), 4)
            final_scores = _as_int_vector(grp.take_final_scores(), 4)

            if shared_feature is None:
                shared_feature = feature.copy()
                shared_ranks = final_ranks.copy()
                shared_scores = final_scores.copy()
                with self.timings.measure('grp'):
                    rank_matrix = self._predict_rank_matrix(shared_feature)
                rewards_by_player, ranks_by_player = build_shared_targets(
                    shared_feature,
                    rank_by_player=shared_ranks,
                    final_scores=shared_scores,
                    rank_matrix=rank_matrix,
                    pts=self.pts,
                )
            elif (
                not np.array_equal(feature, shared_feature)
                or not np.array_equal(final_ranks, shared_ranks)
                or not np.array_equal(final_scores, shared_scores)
            ):
                raise ValueError('同一牌谱的玩家视角包含不一致的 GRP 数据')

            sample_count = len(obs)
            if sample_count == 0:
                continue
            field_lengths = {
                'actions': len(actions),
                'masks': len(masks),
                'at_kyoku': len(at_kyoku),
                'dones': len(dones),
                'apply_gamma': len(apply_gamma),
            }
            for name, actual in field_lengths.items():
                if actual != sample_count:
                    raise ValueError(
                        f'玩家 {player_id} 的 {name} 长度 {actual} 与 obs {sample_count} 不一致'
                    )
            obs = np.asarray(obs, dtype=np.float32)
            actions = np.asarray(actions, dtype=np.int64)
            masks = np.asarray(masks, dtype=np.bool_)
            at_kyoku = _as_int_vector(at_kyoku, sample_count)
            dones = np.asarray(dones, dtype=np.bool_)
            apply_gamma = np.asarray(apply_gamma, dtype=np.bool_)
            if obs.shape != (sample_count, OBS_PLANES, OBS_TILES):
                raise ValueError(f'玩家 {player_id} 的 obs shape 不兼容: {obs.shape}')
            if masks.shape != (sample_count, ACTION_SPACE):
                raise ValueError(f'玩家 {player_id} 的 mask shape 不兼容: {masks.shape}')
            if np.any(actions < 0) or np.any(actions >= ACTION_SPACE):
                raise ValueError(f'玩家 {player_id} 含越界 action')
            if not masks[np.arange(sample_count), actions].all():
                raise ValueError(f'玩家 {player_id} 含 mask 不允许的 action')
            kyoku_count = len(shared_feature)
            if np.any(at_kyoku < 0) or np.any(at_kyoku >= kyoku_count):
                raise ValueError(f'玩家 {player_id} 的样本局索引越界')
            if not dones[-1]:
                raise ValueError(f'玩家 {player_id} 的最后样本没有结束局边界')

            steps = np.zeros(sample_count, dtype=np.int64)
            for index in range(sample_count - 2, -1, -1):
                if not dones[index]:
                    steps[index] = steps[index + 1] + int(apply_gamma[index])
            if steps.max(initial=0) > np.iinfo(np.uint16).max:
                raise ValueError(f'玩家 {player_id} 的折扣步数超出 uint16')
            output.append(SampleBatch(
                obs=obs,
                actions=actions.astype(np.uint8),
                masks=masks,
                steps_to_done=steps.astype(np.uint16),
                rewards=rewards_by_player[at_kyoku, player_id].astype(np.float64),
                ranks=ranks_by_player[at_kyoku + 1, player_id].astype(np.uint8),
            ))

        if not output:
            return _empty_batch()
        return SampleBatch(
            obs=np.concatenate([batch.obs for batch in output]),
            actions=np.concatenate([batch.actions for batch in output]),
            masks=np.concatenate([batch.masks for batch in output]),
            steps_to_done=np.concatenate([batch.steps_to_done for batch in output]),
            rewards=np.concatenate([batch.rewards for batch in output]),
            ranks=np.concatenate([batch.ranks for batch in output]),
        )
