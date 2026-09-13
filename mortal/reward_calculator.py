"""把 GRP 的最终顺位预测转换为主策略使用的按局奖励。

GRP 输入是一场牌谱在每局开始时记录的 7 维特征序列：
``[grand_kyoku, honba, kyotaku, score_0 / 10000, ..., score_3 / 10000]``。
模型对每个历史前缀预测四家最终顺位的 24 种可能排列。本模块先把联合排列概率边缘化
为每名玩家的四种顺位概率，再按照 ``pts`` 中定义的一至四位效用计算期望值。

相邻局边界的期望效用差形成长度为 K 的局间奖励。主策略数据加载器随后把第 k 个奖励
分配给第 k 局内的所有决策，并根据 ``steps_to_done`` 再做动作级折扣。本模块只负责
按局计算，不处理 observation、动作 mask 或折扣步数。

普通 ``dataloader.py`` 会通过 ``calc_delta_pt`` 分别计算玩家奖励；优化后的
``transformer_pipeline/dataloader_grp_transformer.py`` 则调用 ``calc_grp`` 一次取得
四家全部概率，避免同一牌谱重复执行四次 Transformer。
"""

import torch
import numpy as np


class RewardCalculator:
    """在 CPU 上使用冻结 GRP 计算顺位概率和局间奖励。

    ``grp`` 可以是 GRU 版 ``GRP`` 或 ``GRP_Transformer``，但必须支持两个接口：

    - ``grp(inputs)``：接收变长前缀张量列表，返回 ``(N, 24)`` 排列 logits；
    - ``grp.calc_matrix(logits)``：返回 ``(N, 4, 4)`` 的玩家顺位边缘概率。

    ``pts`` 按从一位到四位排列，索引 0..3 与模型的 rank 编码一致。例如
    ``[6, 4, 2, 0]`` 表示最终四个名次的训练效用，而不是麻将终局点棒。
    """

    def __init__(self, grp=None, pts=None, uniform_init=False):
        """固定奖励模型的执行设备和名次效用。

        传入的 ``grp`` 会被原地移动到 CPU 并切换到 eval 模式。每个 DataLoader worker
        各自持有一个计算器，因此不会与主策略所在 GPU 共享模型。``uniform_init`` 为真时，
        第一局开始前的顺位分布强制设为均匀分布，使初始期望效用不依赖 GRP 的首个预测。
        """
        # 奖励推理固定放在 CPU，避免多个数据 worker 争用主策略训练使用的 CUDA 设备。
        self.device = torch.device('cpu')
        self.grp = grp.to(self.device).eval()
        self.uniform_init = uniform_init

        # rank=0..3 分别对应一至四位；GRP 模型本身使用 float64，因此效用也保持双精度。
        pts = pts or [3, 1, -1, -3]
        self.pts = torch.tensor(pts, dtype=torch.float64, device=self.device)

    def calc_grp(self, grp_feature):
        """预测每个历史前缀下四名玩家的最终顺位概率。

        参数 ``grp_feature`` 通常是 NumPy ``float64`` 数组，形状为 ``(K, 7)``，K 是
        牌谱包含的局数。模型需要看到第 0 局、第 0..1 局、……、第 0..K-1 局，因此这里
        构造 K 个长度依次为 1..K 的前缀，而不是只把完整序列送入一次。

        返回 ``matrix`` 的形状为 ``(K, 4, 4)``：
        ``matrix[k, player, rank]`` 表示观察到前 k+1 个局初特征后，玩家 ``player``
        最终取得 ``rank`` 的概率。rank 使用 0..3 表示一至四位。
        """
        # as_tensor 在 CPU 上会尽可能复用 NumPy 切片的存储，并保留输入的 float64 dtype。
        # 列表元素形状依次为 (1, 7)、(2, 7)、...、(K, 7)，由模型内部进行 padding/packing。
        seq = list(map(
            lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
            range(len(grp_feature)),
        ))

        # GRP 只提供固定奖励信号，不建立 autograd 图，也不会由主策略优化器更新。
        with torch.inference_mode():
            logits = self.grp(seq)
        # logits: (K, 24) -> matrix: (K, player=4, rank=4)。24 类是四家顺位的
        # 联合排列，calc_matrix 会把包含同一 player/rank 的排列概率相加得到边缘概率。
        matrix = self.grp.calc_matrix(logits)
        return matrix

    def calc_rank_prob(self, player_id, grp_feature, rank_by_player):
        """构造指定玩家从各局开始到真实终局的顺位概率序列。

        ``rank_by_player`` 长度为 4，``rank_by_player[p]`` 是玩家 p 的真实最终顺位。
        GRP 只能为 K 个局初边界提供预测，因此末尾追加一行真实终局 one-hot，返回形状
        为 ``(K + 1, 4)``。这样相邻两行正好对应一局开始前和该局结束后。
        """
        # 先保留当前玩家在 K 个局初边界上的四种顺位概率。
        matrix = self.calc_grp(grp_feature)

        # 终局已经发生，不再使用预测分布：真实名次位置概率为 1，其余为 0。
        final_ranking = torch.zeros((1, 4), device=self.device)
        final_ranking[0, rank_by_player[player_id]] = 1.
        rank_prob = torch.cat((matrix[:, player_id], final_ranking))
        if self.uniform_init:
            # 均匀初始分布对应固定的 mean(pts) 期望效用，可消除模型对开局面的偏置。
            rank_prob[0, :] = 1 / 4
        return rank_prob

    def calc_delta_pt(self, player_id, grp_feature, rank_by_player):
        """计算指定玩家每一局带来的期望顺位效用变化。

        设 ``P[k, r]`` 是第 k 个边界取得名次 r 的概率，则：

        ``expected_pt[k] = sum_r P[k, r] * pts[r]``

        ``reward[k] = expected_pt[k + 1] - expected_pt[k]``

        输入有 K 个局初特征，概率序列追加终局后长度为 K+1，因此返回 NumPy
        ``float64`` 数组形状为 ``(K,)``。这些值是名次效用增量，不是实际点棒增量。
        """
        rank_prob = self.calc_rank_prob(player_id, grp_feature, rank_by_player)
        # (K+1, 4) @ (4,) -> (K+1,)，得到每个局边界的期望名次效用。
        exp_pts = rank_prob @ self.pts
        # 相邻边界相减后得到每一局的稠密奖励；全部 K 项相加会发生望远镜消去。
        reward = exp_pts[1:] - exp_pts[:-1]
        return reward.cpu().numpy()

    def calc_delta_points(self, player_id, grp_feature, final_scores):
        """计算指定玩家每一局的真实点棒变化，不经过 GRP。

        ``grp_feature[:, 3 + player_id]`` 保存该玩家每局开始时以一万点为单位的分数。
        还原为点棒并追加 ``final_scores[player_id]`` 后，相邻差分得到形状 ``(K,)`` 的
        实际点棒增减。该函数与 ``calc_delta_pt`` 的量纲和语义不同，目前主策略奖励路径
        使用后者；本函数保留用于分析或其他奖励方案。
        """
        # K 个局初分数加一个真实终局分数，共形成 K+1 个局边界。
        seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
        delta_points = seq[1:] - seq[:-1]
        return delta_points
