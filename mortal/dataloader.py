"""把 mjai gzip 牌谱转换为 Mortal 主策略使用的逐动作训练样本。

Rust ``GameplayLoader`` 负责解析事件、推进玩家状态，并按玩家视角提取观测、实际
动作和合法动作掩码；本模块再用冻结的 GRU 版 ``GRP`` 估计局间顺位价值变化，组装
可由 PyTorch ``DataLoader`` 自动合并成 batch 的样本。

普通样本的字段顺序为 ``obs, action, mask, steps_to_done, kyoku_reward,
player_rank``。启用 Oracle 时，会在 ``obs`` 后插入 ``invisible_obs``。其中观测形状
由 ``version`` 决定，动作掩码长度为 46，其余字段均为标量。
"""

import random
import torch
import numpy as np
from torch.utils.data import IterableDataset
from model import GRP
from reward_calculator import RewardCalculator
from libriichi.dataset import GameplayLoader
from config import config


class FileDatasetsIter(IterableDataset):
    """按文件流式解析牌谱，并产生单次决策级训练样本。

    该数据集不预先把全部牌谱展开到内存，而是每次解析若干文件，将得到的玩家视角
    样本放入局部缓冲区、打乱后逐条产出。PyTorch ``DataLoader`` 随后负责把这些列表
    按位置整理成 batch。对象内部缓存同一个生成器，因此一个数据集实例只设计为完整
    消费一次。
    """

    def __init__(
        self,
        version,
        file_list,
        pts,
        oracle = False,
        file_batch_size = 20, # 经验上每份牌谱约产生 660 个玩家决策样本
        reserve_ratio = 0,
        player_names = None,
        excludes = None,
        num_epochs = 1,
        enable_augmentation = False,
        augmented_first = False,
    ):
        """保存数据选择、缓冲和增强参数，等待首次迭代时初始化重型资源。

        ``version`` 选择 Rust 侧观测编码版本；``file_list`` 是可变路径列表，本类会
        原地打乱它，多 worker 时还会在各 worker 的数据集副本中将其裁成文件子集。
        ``pts`` 是一至四位对应的四个效用值，供 GRP 顺位概率换算局间奖励。

        ``oracle`` 控制是否额外提取不可见信息观测。``file_batch_size`` 是一次交给
        Rust 解析器的牌谱文件数，与训练的神经网络 batch size 无关；
        ``reserve_ratio`` 表示每批新增样本中暂留到下一批继续混洗的比例，调用方应
        传入 ``0..1``。``player_names`` 非空时只提取指定玩家，且优先于 ``excludes``；
        否则 ``excludes`` 指定不提取的玩家。

        ``num_epochs`` 决定单次迭代内部遍历文件列表的次数。每个内部 epoch 先按
        ``augmented_first`` 选择是否交换万子与筒子；``enable_augmentation`` 为真时，
        再用相反的增强状态遍历一次同一批文件。
        """
        super().__init__()
        self.version = version
        self.file_list = file_list
        self.pts = pts
        self.oracle = oracle
        self.file_batch_size = file_batch_size
        self.reserve_ratio = reserve_ratio
        self.player_names = player_names
        self.excludes = excludes
        self.num_epochs = num_epochs
        self.enable_augmentation = enable_augmentation
        self.augmented_first = augmented_first
        self.iterator = None

    def build_iter(self):
        """在实际消费数据的进程内加载 GRP，并按配置顺序生成所有样本。

        本方法是生成器，函数体会在首次 ``next()`` 时执行。延迟到这里创建 PyTorch
        模型，可确保多进程 DataLoader 先完成数据集复制，再由每个 worker 各自在 CPU
        上加载一份 GRP；若放进 ``__init__``，模型需要跨进程传递，在 Windows 的
        ``spawn`` 模式下无法按现有方式工作。
        """
        # 标准主策略固定使用 config.grp 下的 GRU 模型；RewardCalculator 会将其切到
        # CPU/eval 模式，并在 inference mode 下把顺位概率转换为局间奖励。
        self.grp = GRP(**config['grp']['network'])
        grp_state = torch.load(config['grp']['state_file'], weights_only=True, map_location=torch.device('cpu'))
        self.grp.load_state_dict(grp_state['model'])
        self.reward_calc = RewardCalculator(self.grp, self.pts)

        # augmentation 开启时，每个内部 epoch 会产出两遍规则等价的数据；首遍是否增强
        # 由 augmented_first 决定，第二遍始终采用相反状态。
        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def load_files(self, augmented):
        """以指定增强状态遍历文件，并通过保留部分样本实现跨文件批次混洗。

        文件顺序会在每次调用时原地打乱。每轮只在内存中保存当前文件批次产生的样本
        和上轮保留样本，全部文件处理完成后再清空尾部缓冲区。
        """
        # 每个增强轮次重新打乱文件，而不沿用索引文件或上一轮的固定顺序。
        random.shuffle(self.file_list)

        # augmented 属于 GameplayLoader 的构造配置，因此每个增强轮次新建一个解析器。
        # 增强在 Rust 侧统一交换整份日志中的万子与筒子，观测、动作和 mask 会同步重算。
        self.loader = GameplayLoader(
            version = self.version,
            oracle = self.oracle,
            player_names = self.player_names,
            excludes = self.excludes,
            augmented = augmented,
        )
        self.buffer = []

        for start_idx in range(0, len(self.file_list), self.file_batch_size):
            # old_buffer_size 是上一文件批次留下的混洗样本；populate_buffer 只追加本批
            # 新样本，所以两次长度之差就是本批实际新增的样本数。
            old_buffer_size = len(self.buffer)
            self.populate_buffer(self.file_list[start_idx:start_idx + self.file_batch_size])
            buffer_size = len(self.buffer)

            # 保留数量按“本批新增量”而非缓冲区总量计算。打乱新旧样本的联合缓冲区后，
            # 产出后半段并留下前 reserved_size 条，使留下的具体样本可以来自任一批次。
            reserved_size = int((buffer_size - old_buffer_size) * self.reserve_ratio)
            if reserved_size > buffer_size:
                continue

            random.shuffle(self.buffer)
            yield from self.buffer[reserved_size:] # 产出大部分样本
            del self.buffer[reserved_size:] # 删除已经产出的部分，只留下保留样本
        # 最后一批没有后继批次可继续混洗，因此随机化后产出全部暂留样本。
        random.shuffle(self.buffer)
        yield from self.buffer
        self.buffer.clear()

    def populate_buffer(self, file_list):
        """解析一批 gzip 牌谱，把每个玩家视角展开并追加到样本缓冲区。

        ``load_gz_log_files`` 的外层结果与输入文件一一对应，每个文件的内层列表包含
        通过玩家过滤条件的若干 ``Gameplay`` 视角。Rust ``take_*`` 方法会把对应向量
        的所有权一次性移交给 Python，同一字段不能在后续再次读取复用。
        """
        data = self.loader.load_gz_log_files(file_list)
        for file in data:
            for game in file:
                # 以下数组均按决策样本排列并具有相同长度。obs 的单样本形状为
                # obs_shape(version)，mask 的单样本形状为 (46,)；at_kyoku 把动作映射到
                # 从 0 开始的局索引，dones 标记该视角在一局中的最后一条决策样本。
                obs = game.take_obs()
                if self.oracle:
                    # Oracle 观测包含牌山、对手手牌等不可见信息，只在显式启用时提取。
                    invisible_obs = game.take_invisible_obs()
                actions = game.take_actions()
                masks = game.take_masks()
                at_kyoku = game.take_at_kyoku()
                dones = game.take_dones()
                apply_gamma = game.take_apply_gamma()

                # grp 和 player_id 属于整个玩家视角。前者保存每局开始时的 7 维特征及
                # 终局信息，后者指定当前视角在四名玩家中的固定编号。
                grp = game.take_grp()
                player_id = game.take_player_id()

                game_size = len(obs)

                # GRP 根据每个局开始前缀预测最终顺位，再以 pts 换算相邻时刻的期望
                # 效用差。返回值按局排列，同一局内的所有动作共享同一个未折扣奖励。
                grp_feature = grp.take_feature()
                rank_by_player = grp.take_rank_by_player()
                kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
                # 奖励数通常等于当前视角出现动作的局数；若末局没有该玩家的决策样本，
                # GRP 仍会包含末局特征，因此奖励数组可能更长。
                assert len(kyoku_rewards) >= at_kyoku[-1] + 1

                # grp_feature 的后四列是以一万点为单位的各家局初分数；恢复为点棒后，
                # 再追加真实终局分数，得到“每局开始 + 终局”的 (K + 1, 4) 序列。
                final_scores = grp.take_final_scores()
                scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
                # 第一次稳定 argsort 按分数从高到低得到玩家顺序，第二次把该顺序反转为
                # 每名玩家的 0..3 顺位。稳定排序使同分时玩家编号较小者排在前面。
                rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
                player_ranks = rank_by_player_seq[:, player_id]

                # 从后向前计算当前动作到本局结束的有效折扣步数。Rust 侧保证每局末条
                # 样本以及全局末条样本的 done=True，因此这些位置保持 0，也不会访问
                # i + 1。apply_gamma 当前只对动作编号 0..37 为真，即打牌/杠牌种选择及
                # 立直；吃碰、杠决定、和牌、流局和跳过不增加折扣步数。
                steps_to_done = np.zeros(game_size, dtype=np.int64)
                for i in reversed(range(game_size)):
                    if not dones[i]:
                        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

                # player_ranks 有 K + 1 个时刻：索引 k 是第 k 局开始前的顺位，索引
                # k + 1 是该局结束后的顺位（最后一局对应真实终局）。因此奖励和辅助
                # 顺位标签分别使用 at_kyoku[i] 与 at_kyoku[i] + 1。
                for i in range(game_size):
                    entry = [
                        obs[i],
                        actions[i],
                        masks[i],
                        steps_to_done[i],
                        kyoku_rewards[at_kyoku[i]],
                        player_ranks[at_kyoku[i] + 1], # +1的原因：下一局开始时玩家的顺位
                    ]
                    if self.oracle:
                        # 保持 Oracle 样本契约：[obs, invisible_obs, action, mask, ...]。
                        entry.insert(1, invisible_obs[i])
                    self.buffer.append(entry)

    def __iter__(self):
        """返回本实例缓存的惰性生成器。

        首次调用只创建生成器，GRP 会在生成器真正前进时初始化。后续调用返回同一对象；
        若它已经耗尽，不会自动从头读取文件。每个 DataLoader worker 持有自己的数据集
        副本和生成器，不会与其他 worker 共享迭代进度。
        """
        if self.iterator is None:
            self.iterator = self.build_iter()
        return self.iterator


def worker_init_fn(*args, **kwargs):
    """按文件数量把当前数据集的路径连续切分给各 DataLoader worker。

    使用向上取整可覆盖全部文件；最后一个或文件不足时的部分 worker 可能获得较短列表
    乃至空列表。这里只平衡文件数，不保证各 worker 的玩家视角数或动作样本数相同，
    因为不同牌谱长度不同。调用方若需要随机分片，应在创建 DataLoader 前先打乱列表。
    """
    # 回调参数中的 worker_id 未直接使用；get_worker_info() 同时提供当前 worker 编号、
    # worker 总数和已经复制到本进程的数据集对象。
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    per_worker = int(np.ceil(len(dataset.file_list) / worker_info.num_workers))
    start = worker_info.id * per_worker
    end = start + per_worker
    # Python 切片允许 end 超过列表长度，因此最后一段无需额外取 min。
    dataset.file_list = dataset.file_list[start:end]
