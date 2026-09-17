"""使用 Transformer GRP 为 Mortal 主策略生成逐动作训练样本。

Rust ``GameplayLoader`` 负责解压并解析 mjai 日志、推进四家状态，以及提取 observation、
实际动作和合法动作 mask。基类 ``dataloader.FileDatasetsIter`` 负责文件遍历、局部缓冲、
样本打乱、数据增强和 DataLoader worker 分片；本模块只替换以下业务环节：

1. 在每个数据进程中加载冻结的 ``GRP_Transformer``；
2. 让同一牌谱的所有玩家视角共享一次 GRP 推理；
3. 把按局的期望顺位效用变化映射到逐动作样本；
4. 识别并隔离确定损坏的牌谱，过滤 action 与 mask 不一致的样本。

普通样本契约为 ``obs, action, mask, steps_to_done, kyoku_reward, player_rank``；启用
Oracle 时在 ``obs`` 后插入 ``invisible_obs``。``train_mortal_offline_with_grp_transformer.py``
通过 ``populate_buffer`` 使用完整的数据加载流程；预计算入口会复用 GRP 初始化和
``iter_file_samples``，但会绕过本模块的自动隔离移动逻辑。
"""

import hashlib
import json
import logging
import os
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from _bootstrap import REPOSITORY_ROOT
from config import config
from dataloader import FileDatasetsIter as BaseFileDatasetsIter
from dataloader import worker_init_fn
from model import GRP_Transformer
from reward_calculator import RewardCalculator


# 隔离功能只管理仓库内这三棵数据目录，避免异常路径导致任意文件被移动。
DATASET_ROOTS = (
    REPOSITORY_ROOT / 'data' / 'grp' / 'train',
    REPOSITORY_ROOT / 'data' / 'grp' / 'val',
    REPOSITORY_ROOT / 'data' / 'grp' / 'test',
)
DATASET_QUARANTINE_ROOT = REPOSITORY_ROOT / 'data' / 'grp' / 'quarantine'

# PyO3 会把 Rust 错误链转换为 RuntimeError 文本。这里只列出能够明确归因于数据文件
# 损坏或牌谱非法的标记；OOM、程序错误等未知异常必须继续抛出，不能误删训练数据。
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
    """计算数据文件对应的隔离路径，目录外文件返回 ``None``。

    ``data/grp/train/a.json.gz`` 会映射到
    ``data/grp/quarantine/train/a.json.gz``。保留 train/val/test 这一层可以避免不同
    数据集中的同名相对路径相互覆盖。``strict=False`` 允许源文件已经被其他 worker
    移走时仍然完成纯路径计算。
    """
    source = Path(filename).resolve(strict=False)
    for dataset_root in DATASET_ROOTS:
        try:
            relative = source.relative_to(dataset_root)
        except ValueError:
            continue
        return DATASET_QUARANTINE_ROOT / dataset_root.name / relative
    return None


def dataset_file_stat(filename):
    """读取逻辑数据文件的元数据，已隔离时透明地读取隔离副本。

    离线训练器会使用文件大小和 ``mtime_ns`` 构建数据清单摘要。隔离采用硬链接后删除
    原路径，文件内容和这些关键元数据保持不变，因此坏文件被隔离后，原始文件索引仍可
    用于恢复训练和校验完整数据集。
    """
    source = Path(filename).resolve(strict=False)
    try:
        return source.stat()
    except FileNotFoundError as error:
        quarantined = dataset_quarantine_path(filename)
        if quarantined is not None and quarantined.is_file():
            return quarantined.stat()
        raise error


def is_quarantined_dataset_file(filename):
    """判断索引中的原路径是否已被移动到预期隔离位置。

    只有“原路径不存在且隔离文件存在”才返回真；源文件仍在时，即使隔离目录出现同名
    文件，也不能把它视为一次已经完成的隔离操作。
    """
    source = Path(filename).resolve(strict=False)
    quarantined = dataset_quarantine_path(filename)
    return not source.exists() and quarantined is not None and quarantined.is_file()


def _sha256_file(filename):
    """流式计算牌谱摘要，避免一次性读取整个压缩文件。"""
    digest = hashlib.sha256()
    with open(filename, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(directory):
    """同步目录项，使硬链接、删除或原子替换尽量跨主机异常持久化。"""
    directory_fd = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


def _write_quarantine_metadata(source, target, error, file_stat, file_sha256):
    """原子写入隔离审计记录，记录来源、内容摘要和触发隔离的错误。"""
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
    """不覆盖地隔离确定损坏的日志，并允许多个 worker 幂等调用。

    操作顺序是“为源 inode 创建隔离硬链接 -> 验证两条路径指向同一文件 -> 删除原
    路径”，而不是先删除再复制。这样任何时刻至少保留一个数据文件名；若另一 worker
    已经完成相同步骤，只要隔离目标确实存在，当前调用就可以继续完成元数据记录。
    隔离文件不会被训练消费，但不会被物理删除，后续仍可根据 JSON 审计或人工恢复。
    """
    source = Path(filename).resolve(strict=False)
    target = dataset_quarantine_path(filename)
    if target is None:
        raise RuntimeError(f'拒绝隔离训练/验证/测试目录外的文件: {source}')
    if not source.exists():
        if not target.is_file():
            raise FileNotFoundError(f'待隔离数据不存在: {source}')

    target.parent.mkdir(parents=True, exist_ok=True)
    if source.exists():
        # 硬链接先把同一 inode 暴露到隔离目录；FileExistsError 可能来自并发 worker，
        # 后续 samefile 校验会区分“同一隔离操作”与“危险的同名文件冲突”。
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
        # 只有目标已经确认指向相同 inode 后才移除原目录项。
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
        # 数据隔离已经成功时，审计 JSON 写入失败不应伪装成数据文件仍在原位置。
        _write_quarantine_metadata(source, target, error, file_stat, file_sha256)
    except Exception:
        logging.exception('写入隔离元数据失败，但数据文件已移动到 %s', target)
    return target


def is_quarantinable_dataset_error(error):
    """仅把明确归属于某个输入文件的数据、压缩或规则错误视为可隔离错误。

    ``error when reading`` 来自 Rust ``GameplayLoader`` 为具体文件添加的上下文；第二层
    marker 再限制根因。两者同时满足可避免把模型、内存或并发错误误判成坏牌谱。
    """
    message = str(error).lower()
    return 'error when reading ' in message and any(
        marker in message
        for marker in QUARANTINABLE_ERROR_MARKERS
    )


class FileDatasetsIter(BaseFileDatasetsIter):
    """使用 Transformer GRP 计算局间奖励的流式逐动作数据集。

    文件遍历和缓冲策略继承自基类。对每份牌谱，Rust 返回若干 ``Gameplay`` 玩家视角；
    本类先用第一个视角携带的共享 GRP 数据计算四家全部目标，再按 ``player_id`` 选择
    当前视角的奖励和顺位标签。这样一份四人牌谱只执行一次 Transformer，而不是四次。

    设牌谱共有 K 局、某玩家视角共有 M 个决策样本，则主要张量为：

    - ``grp_feature``：``(K, 7)``，每局开始时的场况与四家分数；
    - ``kyoku_rewards_by_player``：``(K, 4)``，每局、每家的顺位效用变化；
    - ``rank_by_player_seq``：``(K + 1, 4)``，局初以及真实终局的顺位；
    - ``obs``：长度 M 的 observation 序列，单项形状由 observation version 决定；
    - ``masks``：长度 M 的合法动作 mask，单项形状为 ``(46,)``。
    """

    def __init__(self, *args, grp_config=None, **kwargs):
        """初始化基类，并保存可随 ``spawn`` worker 传递的 GRP 配置快照。

        此处不能提前创建 PyTorch 模型：DataLoader 需要先序列化数据集，再让每个 worker
        在自己的进程和 CPU 地址空间中加载一份 GRP。只复制必要字段也避免 worker 启动后
        观察到调用方对全局 ``config`` 的后续修改。
        """
        super().__init__(*args, **kwargs)
        # 该计数属于当前数据集副本；多 worker 时各进程独立维护，不是全局汇总值。
        self.invalid_action_samples = 0
        source = grp_config or config['grp_transformer']
        self.grp_config = {
            'state_file': source['state_file'],
            'expected_sha256': source.get('expected_sha256'),
            'network': dict(source['network']),
        }

    def initialize_grp_model(self):
        """在当前进程的 CPU 上加载冻结 GRP，并创建奖励计算器。

        训练 worker 由 ``build_iter`` 延迟调用；预计算器则显式调用本方法后，再直接使用
        ``iter_file_samples``。checkpoint 必须包含 ``model``，并以 ``strict=True`` 校验
        网络结构，避免缺失参数或错误模型产生无法察觉的奖励漂移。
        """
        grp_cfg = self.grp_config
        state_file = grp_cfg['state_file']

        expected_sha256 = grp_cfg.get('expected_sha256')
        if expected_sha256:
            # 摘要按块计算，既限制内存占用，也确保每个 spawn worker 实际打开的权重
            # 与主训练进程写入训练契约的冻结快照完全一致。
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
        # 奖励生成固定在 CPU；主策略训练使用的 GPU 由训练器管理，二者互不共享模型。
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

    def build_iter(self): # 重写
        """在实际消费数据的进程内加载 GRP，再进入基类的文件遍历流程。

        本方法是生成器，首次 ``next()`` 时才真正执行。基类 ``load_files`` 会按
        ``file_batch_size`` 调用本类的 ``populate_buffer``；启用 augmentation 时，同一
        文件列表还会以相反花色映射再遍历一次。
        """
        self.initialize_grp_model()

        for _ in range(self.num_epochs):
            yield from self.load_files(self.augmented_first)
            if self.enable_augmentation:
                yield from self.load_files(not self.augmented_first)

    def _build_shared_grp_targets(self, grp_feature, rank_by_player, final_scores):
        """一次计算同一日志四个玩家共享的局间奖励与顺位序列。

        ``grp_feature`` 形状为 ``(K, 7)``。GRP 会对长度为 1..K 的所有历史前缀预测
        24 种最终顺位排列；``calc_matrix`` 再得到 ``matrix[k, player, rank]``。若
        ``pts[rank]`` 是名次效用，则每局边界的期望效用为：

        ``expected_pts[k, player] = sum_r matrix[k, player, r] * pts[r]``。

        追加真实终局名次对应的效用后，相邻行做差得到 ``(K, 4)`` 局间奖励。辅助任务
        标签不使用 GRP 预测，而是从 K 个局初分数和真实终局分数计算 ``(K + 1, 4)``
        顺位序列。
        """
        rank_by_player = np.fromiter(rank_by_player, dtype=np.int64, count=4)
        final_scores = np.asarray(final_scores, dtype=np.int64)

        with torch.inference_mode():
            # matrix: (K, 4, 4)，最后两维依次是玩家和可能的最终顺位。
            matrix = self.reward_calc.calc_grp(grp_feature)
            # (K, 4, 4) @ (4,) -> (K, 4)，得到每个局初前缀下四家的期望效用。
            expected_pts = matrix @ self.reward_calc.pts
            # 最后一个边界没有 GRP 预测，直接使用真实终局顺位对应的确定效用。
            final_expected_pts = self.reward_calc.pts[
                torch.as_tensor(rank_by_player, dtype=torch.int64)
            ]
            expected_pts = torch.cat((expected_pts, final_expected_pts.unsqueeze(0)))
            # reward[k] 表示从第 k 局开始到下一局边界的期望效用增量。
            kyoku_rewards_by_player = torch.diff(expected_pts, dim=0).cpu().numpy()

        # GRP 特征中的分数以一万点为单位（也即1.0表示10000点）；追加真实终局分数后得到 K+1 个边界。
        # 这部分写法和FileDatasetsIter中的populate_buffer的部分代码179-184是一样的
        scores_seq = np.concatenate((grp_feature[:, 3:] * 1e4, [final_scores]))
        # 两次稳定 argsort 把降序玩家编号变成每名玩家的 0..3 顺位；同分时保持玩家编号
        # 较小者在前，与数据集现有的顺位约定一致。
        rank_by_player_seq = (-scores_seq).argsort(-1, kind='stable').argsort(-1, kind='stable')
        return rank_by_player, final_scores, kyoku_rewards_by_player, rank_by_player_seq

    def _load_files_with_quarantine(self, file_list):
        """批量加载牌谱；失败时逐文件定位并隔离确定损坏的输入。

        正常路径保持一次 Rust 批量调用，以利用 Rayon 并行解析。只有批量调用抛出
        ``RuntimeError`` 时才退化为逐文件重试，因此坏文件定位不会持续拖慢健康数据。
        返回的文件名与 ``Gameplay`` 结果保持相同顺序和长度。
        """
        active_files = []
        for filename in file_list:
            # 文件索引保持原始路径不变；已经隔离的逻辑文件在后续恢复时直接跳过。
            if is_quarantined_dataset_file(filename):
                logging.warning('skipped quarantined dataset file: %s', filename)
                continue
            active_files.append(filename)
        if not active_files:
            return [], []

        try:
            return active_files, self.loader.load_gz_log_files(active_files)
        except RuntimeError:
            # 批量异常只说明至少一个文件失败。逐个重放后，健康文件仍然进入本轮训练，
            # 可确认的数据错误会隔离，无法确认根因的异常则立即向上传播。
            loaded_files = []
            loaded_data = []
            for filename in active_files:
                try:
                    # 返回的是单元素列表，所以下面要取file_data[0]
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

    def iter_file_samples(self, filename, file_games): # 重写
        """把单个牌谱的玩家视角转换为带来源定位信息的合法训练样本。

        每次产出 ``(entry, player_id, sample_idx)``。``entry`` 是训练器使用的六元样本；
        后两个值不参与普通 DataLoader 训练，但预计算缓存用它们记录样本来自哪个玩家和
        哪个决策位置。Rust ``take_*`` API 会转移内部数组所有权，同一 ``Gameplay`` 的
        相应字段在本次调用后不可再次提取。
        """
        shared = None
        shared_feature = None

        for game in file_games:
            # 以下逐动作字段长度都为 M。at_kyoku 把第 i 个动作映射到 0..K-1 的局编号；
            # dones 标记局末样本，apply_gamma 决定该动作是否增加折扣距离。
            obs = game.take_obs()
            if self.oracle:
                invisible_obs = game.take_invisible_obs()
            actions = game.take_actions()
            masks = game.take_masks()
            at_kyoku = game.take_at_kyoku()
            dones = game.take_dones()
            apply_gamma = game.take_apply_gamma()

            # 以下字段描述整个玩家视角。正常四人牌谱的不同视角应携带完全相同的 GRP
            # 特征、最终顺位和终局分数，只有 player_id 不同。
            grp = game.take_grp()
            player_id = game.take_player_id()
            grp_feature = grp.take_feature()
            rank_by_player = grp.take_rank_by_player()
            final_scores = grp.take_final_scores()

            if shared is None:
                # 第一个可用视角承担唯一一次 GRP 前向，并缓存四家全部目标。
                shared_feature = grp_feature.copy()
                shared = self._build_shared_grp_targets(
                    grp_feature,
                    rank_by_player,
                    final_scores,
                )
            else:
                # 共享计算成立的前提是 Rust 为各视角提取了相同的整场信息。这里显式验证，
                # 避免悄悄把第一个玩家视角的奖励套用到不一致数据。
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
            # 从共享的 (K, 4) 和 (K+1, 4) 矩阵中选择当前玩家一列。
            kyoku_rewards = kyoku_rewards_by_player[:, player_id]
            player_ranks = rank_by_player_seq[:, player_id]
            game_size = len(obs)
            # 玩家可能没有在最后几局产生决策样本，因此奖励序列可以比实际引用范围更长；
            # 反过来则意味着 Rust 样本与 GRP 局序列不一致，不能继续训练。
            if len(kyoku_rewards) < at_kyoku[-1] + 1:
                raise RuntimeError('GRP 奖励序列短于训练样本所需的局数')

            # 从局末向前递推有效折扣距离。Rust 保证每局最后一个样本 done=True，因此
            # 该位置保持 0，也不会在最后一个元素访问 i+1。apply_gamma 当前只对动作标签
            # 0..37 为真，其他响应类动作不会增加 gamma 的指数。
            steps_to_done = np.zeros(game_size, dtype=np.int64)
            for i in reversed(range(game_size)):
                if not dones[i]:
                    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])

            for i in range(game_size):
                action = int(actions[i])
                mask = masks[i]
                # 数据动作必须同时位于动作空间内且被当时的合法动作 mask 允许。过滤而不是
                # 修补标签可以避免用未知规则错误污染 DQN/CQL 目标。
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
                # 同一局的所有动作共享 kyoku_reward；训练器稍后使用
                # gamma ** steps_to_done 将局末奖励折扣回当前决策。player_rank 取 k+1，
                # 表示该局结束后（下一局开始前）的顺位，而不是当前局初顺位。
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

    def populate_buffer(self, file_list): # 重写
        """解析一批日志并把逐动作样本追加到基类管理的局部缓冲区。

        外层 ``loaded_files`` 与 ``data`` 一一对应，内层 ``file_games`` 是通过玩家筛选的
        若干视角。来源定位字段只供预计算器使用，普通训练缓冲区仅保存 ``entry``；随后
        基类负责打乱缓冲区并逐条 yield，PyTorch DataLoader 再自动堆叠成 batch。
        """
        loaded_files, data = self._load_files_with_quarantine(file_list)
        for filename, file_games in zip(loaded_files, data, strict=True):
            for entry, _, _ in self.iter_file_samples(filename, file_games):
                self.buffer.append(entry)


# 明确限制 ``from dataloader_with_grp_transformer import *`` 的公共表面；worker_init_fn
# 直接复用基类实现，其余以下名称是离线训练器进行数据清单和隔离管理所需的 API。
__all__ = [
    'FileDatasetsIter',
    'dataset_file_stat',
    'dataset_quarantine_path',
    'is_quarantined_dataset_file',
    'quarantine_dataset_file',
    'worker_init_fn',
]
