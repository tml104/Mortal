"""在不运行奖励模型的前提下检查牌谱能否产生预计算样本。"""

from dataclasses import dataclass

import numpy as np

from libriichi.dataset import GameplayLoader
from precompute_pipeline.common.codec import (
    ACTION_SPACE,
    OBS_PLANES,
    OBS_TILES,
    validate_observation_semantics,
)
from precompute_pipeline.quarantine_logs.log_validation import InvalidLogData


KNOWN_GAMEPLAY_ERROR_MARKERS = (
    'empty or invalid game log',
    'invalid log:',
    'on event ',
    'rule violation:',
)


@dataclass(frozen=True)
class ApplicabilityResult:
    sample_count: int
    nonempty_players: int
    kyoku_count: int


def _int_vector(value, count, field):
    if isinstance(value, (bytes, bytearray, memoryview)):
        array = np.frombuffer(value, dtype=np.uint8)
    else:
        array = np.fromiter(value, dtype=np.int64, count=count)
    if array.shape != (count,):
        raise InvalidLogData(f'{field} 应有 {count} 项，实际 shape={array.shape}')
    return array.astype(np.int64, copy=False)


def validate_gameplays(games, *, max_seq_len):
    """检查 Rust 玩家视角的样本字段与奖励模型输入容量。"""
    total_samples = 0
    nonempty_players = 0
    shared_feature = None
    shared_ranks = None
    shared_scores = None
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
        kyoku_count = len(feature)
        ranks = _int_vector(grp.take_rank_by_player(), 4, 'rank_by_player')
        scores = _int_vector(grp.take_final_scores(), 4, 'final_scores')
        if feature.shape != (kyoku_count, 7) or not np.isfinite(feature).all():
            raise InvalidLogData(f'玩家 {player_id} 的 GRP feature 不兼容: {feature.shape}')
        if not 0 < kyoku_count <= max_seq_len:
            raise InvalidLogData(
                f'牌谱局数 {kyoku_count} 超出奖励模型容量 1..{max_seq_len}'
            )
        if sorted(ranks.tolist()) != [0, 1, 2, 3]:
            raise InvalidLogData('终局顺位必须是 0..3 的排列')
        if shared_feature is None:
            shared_feature = feature.copy()
            shared_ranks = ranks.copy()
            shared_scores = scores.copy()
        elif (
            not np.array_equal(feature, shared_feature)
            or not np.array_equal(ranks, shared_ranks)
            or not np.array_equal(scores, shared_scores)
        ):
            raise InvalidLogData('同一牌谱的玩家视角包含不一致的 GRP 数据')

        sample_count = len(obs)
        if sample_count == 0:
            continue
        nonempty_players += 1
        total_samples += sample_count
        fields = {
            'actions': len(actions),
            'masks': len(masks),
            'at_kyoku': len(at_kyoku),
            'dones': len(dones),
            'apply_gamma': len(apply_gamma),
        }
        for field, actual in fields.items():
            if actual != sample_count:
                raise InvalidLogData(
                    f'玩家 {player_id} 的 {field} 长度 {actual} 与 obs {sample_count} 不一致'
                )
        obs = np.asarray(obs, dtype=np.float32)
        try:
            validate_observation_semantics(obs)
        except ValueError as exc:
            raise InvalidLogData(f'玩家 {player_id} 的 observation 不兼容: {exc}') from exc
        actions = np.asarray(actions, dtype=np.int64)
        masks = np.asarray(masks)
        at_kyoku = _int_vector(at_kyoku, sample_count, 'at_kyoku')
        dones = np.asarray(dones, dtype=np.bool_)
        apply_gamma = np.asarray(apply_gamma, dtype=np.bool_)
        if masks.shape != (sample_count, ACTION_SPACE):
            raise InvalidLogData(f'玩家 {player_id} 的 mask shape 不兼容: {masks.shape}')
        if np.any(actions < 0) or np.any(actions >= ACTION_SPACE):
            raise InvalidLogData(f'玩家 {player_id} 含越界 action')
        if not masks[np.arange(sample_count), actions].all():
            raise InvalidLogData(f'玩家 {player_id} 含 mask 不允许的 action')
        if np.any(at_kyoku < 0) or np.any(at_kyoku >= kyoku_count):
            raise InvalidLogData(f'玩家 {player_id} 的样本局索引越界')
        if not dones[-1]:
            raise InvalidLogData(f'玩家 {player_id} 的最后样本没有结束局边界')
        steps = np.zeros(sample_count, dtype=np.int64)
        for index in range(sample_count - 2, -1, -1):
            if not dones[index]:
                steps[index] = steps[index + 1] + int(apply_gamma[index])
        if steps.max(initial=0) > np.iinfo(np.uint16).max:
            raise InvalidLogData(f'玩家 {player_id} 的折扣步数超出 uint16')

    if shared_feature is None or total_samples == 0:
        raise InvalidLogData('四家合计零样本，本训练链路不可用')
    return ApplicabilityResult(
        sample_count=total_samples,
        nonempty_players=nonempty_players,
        kyoku_count=len(shared_feature),
    )


class SampleApplicabilityValidator:
    def __init__(self, max_seq_len=60):
        self.max_seq_len = int(max_seq_len)
        self.loader = GameplayLoader(
            4,
            oracle=False,
            player_names=[],
            excludes=[],
            trust_seed=False,
            always_include_kan_select=True,
            augmented=False,
        )

    def validate_raw_log(self, raw_log):
        try:
            games = self.loader.load_log(raw_log)
        except RuntimeError as exc:
            message = str(exc).lower()
            if any(marker in message for marker in KNOWN_GAMEPLAY_ERROR_MARKERS):
                raise InvalidLogData(f'libriichi 无法提取样本: {exc}') from exc
            raise
        return validate_gameplays(games, max_seq_len=self.max_seq_len)
