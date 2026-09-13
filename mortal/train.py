"""Mortal 主策略的离线与在线训练入口。

训练样本由 ``FileDatasetsIter`` 从 mjai 对局日志中提取，依次经过 ``Brain``
编码、``DQN`` 动作价值预测和 ``AuxNet`` 顺位辅助预测。离线模式使用固定日志并
增加 CQL 约束；在线模式则从参数服务器取得自对战 replay，并定期回传新参数。

在线评测结束后训练进程存在已知的卡住问题，因此 ``main`` 以父进程持续拉起训练
子进程；离线模式不需要这层管理，直接完成配置指定的一轮数据训练后退出。

配置由同目录的 ``config.py`` 在首次导入时加载：若环境变量 ``MORTAL_CFG`` 已设置，
则读取它指定的 TOML；否则读取进程当前工作目录下的 ``config.toml``。相对路径不会
基于 ``train.py`` 或配置文件所在目录展开，因此建议从仓库根目录启动，并让
``MORTAL_CFG`` 使用绝对路径。TOML 中各配置项的相对路径同样由具体消费者按当前
工作目录解析。
"""


def train():
    """初始化训练资源，并按配置执行一轮离线训练或持续执行在线训练。

    本函数会读取配置、模型和 checkpoint，创建数据加载器，并写入 TensorBoard、
    checkpoint 及在线参数服务。它不接收显式参数，所有运行约束均来自 ``config``。
    """
    # prelude 负责统一日志格式、标准输入编码和第三方告警过滤，需先于其余训练导入。
    import prelude

    import logging
    import sys
    import os
    import gc
    import gzip
    import json
    import shutil
    import random
    import torch
    from os import path
    from glob import glob
    from datetime import datetime
    from itertools import chain
    from torch import optim, nn
    from torch.amp import GradScaler
    from torch.nn.utils import clip_grad_norm_
    from torch.utils.data import DataLoader
    from torch.utils.tensorboard import SummaryWriter
    from common import submit_param, parameter_count, drain, filtered_trimmed_lines, tqdm
    from player import TestPlayer
    from dataloader import FileDatasetsIter, worker_init_fn
    from lr_scheduler import LinearWarmUpCosineAnnealingLR
    from model import Brain, DQN, AuxNet
    from libriichi.consts import obs_shape
    # 导入 config 时会立即打开并解析 TOML，且 Python 的模块缓存使本进程后续导入
    # 复用同一份 config。优先级为 MORTAL_CFG 指定文件 > 当前工作目录/config.toml；
    # 不会合并两份配置，也不会自动相对 train.py 定位默认文件。

    # 注：目前config.toml当中其实缺失了很多配置项，这些配置项更完整的版本目前位于
    # /home/tml104/Mortal/artifacts/mortal_transformer_offline_bs1024_v1/config.toml，
    # （具体请参考train_cmd.md当中的开始训练命令）
    #
    from config import config

    # control 决定训练模式和所有以 batch 为单位的周期。这里的 steps 表示已经处理的
    # 完整训练 batch 数，不等同于样本数，也不一定等同于 optimizer.step() 次数。
    # control.version：同时选择 Rust 侧观测编码版本以及 Brain/DQN 的网络分支；当前
    # train.py 要求 Brain 返回单个 1024 维 phi，实际应配置为 2、3 或 4。
    version = config['control']['version']

    # control.online：false 使用固定日志并计算 CQL；true 从服务端 drain 在线 replay、
    # 关闭 CQL、定期提交参数，并由 main 使用可重启的子进程持续训练。
    online = config['control']['online']
    # control.batch_size：一次前向和反向使用的决策样本数；它直接影响显存占用，并决定
    # obs、动作、mask、Q 值统计缓冲的第一维批量规模。
    batch_size = config['control']['batch_size']
    # control.opt_step_every：梯度累积的 batch 数。loss 会除以该值，累计到边界才调用
    # optimizer.step()；近似有效 batch size 为 batch_size * opt_step_every。
    opt_step_every = config['control']['opt_step_every']
    # control.save_every：每处理多少个完整 batch 写一次 TensorBoard 并覆盖普通
    # checkpoint；该值还决定内存中 Q 预测/目标统计窗口的长度。
    save_every = config['control']['save_every']
    # control.test_every：每处理多少个完整 batch 与固定 baseline 做一次一对三评测；
    # 评测发生在保存之后，因此必须是 save_every 的整数倍。
    test_every = config['control']['test_every']
    # control.submit_every：仅在线模式使用，表示训练期间每隔多少个 batch 向参数服务
    # 发布一次 Brain/DQN；保存点和一轮 replay 结束时还可能额外发布。
    submit_every = config['control']['submit_every']
    # test_play.games：每次定期评测计划运行的总对局数。test_play() 用 games // 4
    # 作为 seed 数，每个 seed 轮换四个座次；不是 4 的倍数时余数会被舍弃。
    test_games = config['test_play']['games']
    # cql.min_q_weight：离线 CQL 正则项在总 loss 中的乘数，用于抑制日志分布外动作的
    # 过高 Q 估计；在线模式不计算 CQL，此值也不会参与在线 loss。
    min_q_weight = config['cql']['min_q_weight']
    # aux.next_rank_weight：下一局顺位四分类交叉熵在总 loss 中的乘数；该辅助任务只
    # 训练共享表征，不参与普通出牌推理。
    next_rank_weight = config['aux']['next_rank_weight']
    # 保存点必须位于梯度累积边界，评测点必须位于保存点，才能得到完整且可恢复的状态。
    assert save_every % opt_step_every == 0
    assert test_every % save_every == 0

    # control.device：Brain、DQN、AuxNet、训练 batch 和损失所在的 torch 设备，例如
    # cuda:0 或 cpu；评测 baseline 可以通过 baseline.test.device 使用另一设备。
    device = torch.device(config['control']['device'])
    # control.enable_cudnn_benchmark：允许 cuDNN 针对稳定输入形状搜索较快算法；可能
    # 增加启动开销并降低严格可复现性，评测期间会临时关闭，结束后恢复该值。
    torch.backends.cudnn.benchmark = config['control']['enable_cudnn_benchmark']
    # control.enable_amp：控制 autocast 和 GradScaler，减少支持设备上的计算与显存成本；
    # false 时 GradScaler 保留相同调用接口但不缩放梯度。
    enable_amp = config['control']['enable_amp']
    # control.enable_compile：true 时对三个训练模型调用 torch.compile()；可提升稳定形状
    # 下的吞吐，但会产生首次编译开销，并要求当前 PyTorch/设备后端支持。
    enable_compile = config['control']['enable_compile']

    # env.pts：长度为 4 的一至四位效用值。RewardCalculator 用 GRP 的顺位概率计算
    # 相邻局面期望效用之差，并把该差值作为每局奖励；它不是原始点棒差。
    pts = config['env']['pts']
    # env.gamma：把局奖励折扣回当前动作的系数，target = gamma ** steps_to_done * reward；
    # 设为 1 时同一局内不随动作距离衰减。
    gamma = config['env']['gamma']

    # dataset.file_batch_size：GameplayLoader 每次集中解析的牌谱文件数，控制 CPU 解析
    # 缓冲规模和局部洗牌范围；它与神经网络的 control.batch_size 无关。
    file_batch_size = config['dataset']['file_batch_size']
    # dataset.reserve_ratio：每批新解析样本中暂留到下一文件批次继续混洗的比例；0 表示
    # 当前文件批次的样本立即全部产出，较大值会增强相邻文件批次间的混合。
    reserve_ratio = config['dataset']['reserve_ratio']
    # dataset.num_workers：DataLoader 子进程数。大于 1 时文件列表会预先打乱并按 worker
    # 切分；每个 worker 都会在 CPU 上各自加载一份 GRP 奖励模型。
    num_workers = config['dataset']['num_workers']
    # dataset.num_epochs：FileDatasetsIter 在一次 train_epoch() 内重复遍历文件列表的
    # 次数；外层离线训练仍只调用一次 train_epoch()。
    num_epochs = config['dataset']['num_epochs']
    # dataset.enable_augmentation：每个内部 epoch 是否在首轮之后再执行一轮相反状态的
    # 数据增强，使同一文件列表同时产生增强与非增强样本。
    enable_augmentation = config['dataset']['enable_augmentation']
    # dataset.augmented_first：决定每个内部 epoch 首轮是否启用增强；若 augmentation
    # 开启，第二轮使用相反值；若关闭，它仍决定唯一一轮是否增强。
    augmented_first = config['dataset']['augmented_first']
    # optim.eps：AdamW 分母的数值稳定项，避免二阶矩估计接近零时发生除零或过大更新。
    eps = config['optim']['eps']
    # optim.betas：AdamW 一阶动量和二阶矩指数移动平均的两个衰减系数。
    betas = config['optim']['betas']
    # optim.weight_decay：只施加到 Linear/Conv1d 的 weight 参数组；bias、BatchNorm 和
    # 其他参数被放入无权重衰减组。
    weight_decay = config['optim']['weight_decay']
    # optim.max_grad_norm：正数时在 optimizer.step() 前按全体参数总范数裁剪梯度；
    # 设为 0 或负数时完全禁用梯度裁剪。
    max_grad_norm = config['optim']['max_grad_norm']

    # Brain 把观测编码为共享表征 phi，DQN 产生各动作 Q 值，AuxNet 从同一 phi 预测
    # 当前动作所在局结束后的四档顺位。现有循环要求 Brain 返回单个 1024 维张量，
    # 因此标准训练配置使用 v2-v4；v1 返回 (mu, logsig)，不能直接走此训练路径。
    # resnet.conv_channels：Brain 残差主干的通道宽度，越大通常容量、显存和计算量越高。
    # resnet.num_blocks：残差块数量，决定主干深度；两项共同决定 checkpoint 参数形状，
    # 恢复旧权重时必须与其保存配置一致。
    mortal = Brain(version=version, **config['resnet']).to(device)
    dqn = DQN(version=version).to(device)
    aux_net = AuxNet((4,)).to(device)
    all_models = (mortal, dqn, aux_net)
    if enable_compile:
        for m in all_models:
            m.compile()

    logging.info(f'version: {version}')
    logging.info(f'obs shape: {obs_shape(version)}')
    logging.info(f'mortal params: {parameter_count(mortal):,}')
    logging.info(f'dqn params: {parameter_count(dqn):,}')
    logging.info(f'aux params: {parameter_count(aux_net):,}')

    # freeze_bn.mortal：true 时即使 Brain 处于 train 模式，BatchNorm 仍使用已有运行
    # 均值/方差，适合在线数据分布小批量波动场景；它不会冻结 BN 的 weight 和 bias。
    mortal.freeze_bn(config['freeze_bn']['mortal'])

    # 只对 Linear/Conv1d 的 weight 应用 AdamW 权重衰减；bias、归一化参数及其他参数
    # 放入无衰减组。优化器基础 lr 设为 1，使 scheduler 返回的倍率直接对应实际 lr。
    decay_params = []
    no_decay_params = []
    for model in all_models:
        params_dict = {}
        to_decay = set()
        for mod_name, mod in model.named_modules():
            for name, param in mod.named_parameters(prefix=mod_name, recurse=False):
                params_dict[name] = param
                if isinstance(mod, (nn.Linear, nn.Conv1d)) and name.endswith('weight'):
                    to_decay.add(name)
        decay_params.extend(params_dict[name] for name in sorted(to_decay))
        no_decay_params.extend(params_dict[name] for name in sorted(params_dict.keys() - to_decay))
    param_groups = [
        {'params': decay_params, 'weight_decay': weight_decay},
        {'params': no_decay_params},
    ]
    optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
    # optim.scheduler 整节传入自定义调度器：peak/final 是峰值与最终学习率，
    # warm_up_steps 控制从 init 线性升至 peak 的 batch 数，max_steps 控制随后余弦下降
    # 的终点。可选 init 是初始学习率，offset 平移当前步数，epoch_size>0 时让曲线按
    # 该周期重启。scheduler 每个 batch 前进一步，而不是每次 optimizer.step() 才前进。
    scheduler = LinearWarmUpCosineAnnealingLR(optimizer, **config['optim']['scheduler'])
    # GradScaler 在 enable_amp=False 时退化为直通逻辑，后续无需维护两套反向传播代码。
    scaler = GradScaler(device.type, enabled=enable_amp)
    # TestPlayer 构造时会间接读取 baseline.test.state_file（固定对手 checkpoint）、
    # baseline.test.device（对手推理设备）、baseline.test.enable_compile（是否编译对手）
    # 和 test_play.log_dir（评测牌谱输出目录）。即使尚未到首次评测，这些配置和权重
    # 也必须有效；test_play 会在评测前删除并重建该专用日志目录。
    test_player = TestPlayer() # 用于训练到一半进行定期衡量当前模型的实力，可以先不看
    best_perf = {
        'avg_rank': 4.,
        'avg_pt': -135.,
    } # 衡量出来的最佳结果表现

    steps = 0
    # control.state_file：普通训练 checkpoint 路径。文件存在时从中恢复模型及适用的
    # 优化状态；每到 save_every 又覆盖该文件，因此不可指向需保留的只读基线权重。
    state_file = config['control']['state_file']
    # control.best_state_file：评测同时改善 avg_pt 和 avg_rank 时，从普通 checkpoint
    # 复制出的最佳版本路径；它与 state_file 应配置为不同文件。
    best_state_file = config['control']['best_state_file']
    if path.exists(state_file):
        # weights_only=True 限制反序列化范围；map_location 使模型与优化器状态落到训练设备。
        state = torch.load(state_file, weights_only=True, map_location=device)
        timestamp = datetime.fromtimestamp(state['timestamp']).strftime('%Y-%m-%d %H:%M:%S')
        logging.info(f'loaded: {timestamp}')
        mortal.load_state_dict(state['mortal'])
        dqn.load_state_dict(state['current_dqn'])
        aux_net.load_state_dict(state['aux_net'])
        # checkpoint 内 state['config']['control']['online'] 是保存时的旧模式快照，不是
        # 当前 TOML 的第二份配置。离线 checkpoint 切换到在线训练时保留模型和步数，
        # 但重新初始化 optimizer/scheduler；在线续训在线或离线续训则恢复优化状态。
        if not online or state['config']['control']['online']:
            optimizer.load_state_dict(state['optimizer'])
            scheduler.load_state_dict(state['scheduler'])
        scaler.load_state_dict(state['scaler'])
        best_perf = state['best_perf']
        steps = state['steps']

    optimizer.zero_grad(set_to_none=True)
    mse = nn.MSELoss()
    ce = nn.CrossEntropyLoss()

    if device.type == 'cuda':
        logging.info(f'device: {device} ({torch.cuda.get_device_name(device)})')
    else:
        logging.info(f'device: {device}')

    if online:
        # submit_param 与 drain 会间接读取 online.remote.host 和 online.remote.port，二者
        # 分别指定参数/replay 服务的主机名与 TCP 端口。先发布初始参数并标记训练端
        # 空闲，客户端即可基于该版本开始生成 replay；离线模式不会使用这两个键。
        submit_param(mortal, dqn, is_idle=True)
        logging.info('param has been submitted')

    # control.tensorboard_dir：SummaryWriter 的 event 输出目录。stats 和 Q 值缓冲覆盖
    # 一个 save_every 窗口，loss、学习率、Q 分布及评测指标都写入此目录。
    writer = SummaryWriter(config['control']['tensorboard_dir'])
    stats = {
        'dqn_loss': 0,
        'cql_loss': 0,
        'next_rank_loss': 0,
    }
    all_q = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    all_q_target = torch.zeros((save_every, batch_size), device=device, dtype=torch.float32)
    idx = 0

    def train_epoch():
        """构造一次数据迭代并消费其中所有可组成完整 batch 的样本。

        离线模式从文件索引读取固定数据集；在线模式会阻塞等待服务器 drain 出新的
        replay。函数会更新外层的 ``steps``、统计缓冲、模型参数和输出文件。
        """
        nonlocal steps
        nonlocal idx

        player_names = []
        if online:
            # 在线牌谱含 trainee 与 baseline 等视角，只训练名为 trainee 的玩家样本。
            player_names = ['trainee']
            # drain() 连接 online.remote 指定的服务并阻塞轮询，直到服务端返回一个非空
            # drain_dir；目录本身由 server 配置和管理，不在 train.py 中直接指定。
            dirname = drain()
            file_list = list(map(lambda p: path.join(dirname, p), os.listdir(dirname)))
        else:
            # 空玩家集合表示不过滤；非空时仅保留至少包含一个目标玩家的整场牌谱。
            player_names_set = set()
            # dataset.player_names_files：若干纯文本白名单文件，每行一个玩家名，空行会
            # 忽略；列表为空或文件内容均为空时不按玩家过滤离线牌谱。
            for filename in config['dataset']['player_names_files']:
                with open(filename) as f:
                    player_names_set.update(filtered_trimmed_lines(f))
            player_names = list(player_names_set)
            logging.info(f'loaded {len(player_names):,} players')

            # dataset.file_index：缓存筛选后牌谱路径列表的 torch 文件。存在时直接复用，
            # 不会重新扫描 globs，也不会自动发现新增、删除或改动的源牌谱。
            file_index = config['dataset']['file_index'] #file_index 是一个由 torch.save() 写出的牌谱路径缓存文件。它保存的是牌谱路径列表

            '''
            文件内容例如:
            {
                "file_list": [
                    "/data/logs/game003.json.gz",
                    "/data/logs/game002.json.gz",
                    "/data/logs/game001.json.gz",
                ]
            }
            '''

            # 若牌谱文件索引存在时直接加载
            if path.exists(file_index):
                # 索引只缓存文件列表，不会自动感知新增日志或日志内容变化。
                index = torch.load(file_index, weights_only=True)
                file_list = index['file_list']
            else:
                logging.info('building file index...')
                file_list = []
                # dataset.globs：仅在 file_index 不存在时递归展开的离线 .json.gz 路径
                # 模式列表；相对 glob 以启动 train.py 时的当前工作目录为基准。
                for pat in config['dataset']['globs']:
                    file_list.extend(glob(pat, recursive=True))
                if len(player_names_set) > 0:
                    filtered = []
                    for filename in tqdm(file_list, unit='file'):
                        with gzip.open(filename, 'rt') as f:
                            start = json.loads(next(f))
                            if not set(start['names']).isdisjoint(player_names_set): # 若不是“交集为空”，则将此牌谱加入filtered
                                filtered.append(filename)
                    file_list = filtered
                file_list.sort(reverse=True)
                torch.save({'file_list': file_list}, file_index)
        logging.info(f'file list size: {len(file_list):,}')

        # 此时file_list为过滤后的文件路径列表

        # 显示距离下一次基线评测还需处理多少个 batch；整除时结果为 0。
        before_next_test_play = (test_every - steps % test_every) % test_every
        logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

        if num_workers > 1:
            # worker_init_fn 随后按连续区间切分列表，预先打乱可避免各 worker 数据偏置。
            random.shuffle(file_list)

        # 每个样本对应一次决策，字段依次为观测、实际动作、合法动作 mask、距本局结束
        # 的折扣步数、本局奖励，以及该局结束后的玩家顺位标签。
        # FileDatasetsIter 还会间接读取 grp.network.hidden_size/num_layers 来构造 CPU GRP，
        # 并从 grp.state_file 安全加载顺位预测权重。每个 DataLoader worker 各加载一份；
        # GRP 只计算奖励、不进入主优化器，其结构必须与 checkpoint 匹配。
        file_data = FileDatasetsIter(
            version = version,
            file_list = file_list,
            pts = pts,
            file_batch_size = file_batch_size,
            reserve_ratio = reserve_ratio,
            player_names = player_names,
            num_epochs = num_epochs,
            enable_augmentation = enable_augmentation,
            augmented_first = augmented_first,
        )
        data_loader = iter(DataLoader(
            dataset = file_data,
            batch_size = batch_size,
            drop_last = False,
            num_workers = num_workers,
            pin_memory = True,
            worker_init_fn = worker_init_fn,
        ))

        # IterableDataset 的文件/worker 边界可能产生短 batch。先缓存这些 batch，数据
        # 迭代完毕后再拼接成完整 batch；最终仍不足 batch_size 的少量样本会被舍弃。
        remaining_obs = []
        remaining_actions = []
        remaining_masks = []
        remaining_steps_to_done = []
        remaining_kyoku_rewards = []
        remaining_player_ranks = []
        remaining_bs = 0
        pb = tqdm(total=save_every, desc='TRAIN', initial=steps % save_every)

        def train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks):
            """对一个完整 batch 前向、反向，并处理更新、保存和评测周期。

            张量形状分别为 ``obs=(B, *obs_shape(version))``、``actions=(B,)``、
            ``masks=(B, ACTION_SPACE)``，其余监督量均为 ``(B,)``；这里
            ``B == batch_size``。函数会修改模型梯度、外层计数器和训练输出。
            """
            nonlocal steps
            nonlocal idx
            nonlocal pb

            obs = obs.to(dtype=torch.float32, device=device)
            actions = actions.to(dtype=torch.int64, device=device)
            masks = masks.to(dtype=torch.bool, device=device)
            steps_to_done = steps_to_done.to(dtype=torch.int64, device=device)
            kyoku_rewards = kyoku_rewards.to(dtype=torch.float64, device=device)
            player_ranks = player_ranks.to(dtype=torch.int64, device=device)
            # 数据生产端应保证记录动作在该时刻合法，尽早检查可防止错误标签静默训练。
            assert masks[range(batch_size), actions].all()

            # 使用从当前动作到本局结束的有效步数折扣局奖励，得到 Monte Carlo Q 目标。
            # 奖励先以 float64 完成幂和乘法，再转为网络损失使用的 float32。
            q_target_mc = gamma ** steps_to_done * kyoku_rewards
            q_target_mc = q_target_mc.to(torch.float32)

            with torch.autocast(device.type, enabled=enable_amp): # enable_amp 默认为 True，也即开启了混合精度训练
                phi = mortal(obs)
                # DQN 会把非法动作位置屏蔽为 -inf；q 只抽取日志中实际执行动作的价值。
                q_out = dqn(phi, masks)
                q = q_out[range(batch_size), actions]
                # 0.5 系数使平方误差的导数形式更简洁，不改变最优解位置。
                dqn_loss = 0.5 * mse(q, q_target_mc) # 计算：对所选的动作，预测的Q与实际Q进行mse
                cql_loss = 0
                if not online:
                    # CQL 压低数据集未选择动作的 Q 值，缓解固定离线数据上的分布外高估。
                    # 在线数据由当前策略持续生成，因此该模式不加入此惩罚。
                    cql_loss = q_out.logsumexp(-1).mean() - q.mean()

                # 辅助任务预测动作所在局结束后的 0..3 顺位，为共享表征提供结果监督。
                next_rank_logits, = aux_net(phi)
                next_rank_loss = ce(next_rank_logits, player_ranks) # 4分类交叉熵损失

                loss = sum((
                    dqn_loss,
                    cql_loss * min_q_weight,
                    next_rank_loss * next_rank_weight,
                ))
            # 每个 batch 的 loss 除以累积次数，保证累积后的梯度尺度近似普通大 batch。
            scaler.scale(loss / opt_step_every).backward()

            # 统计张量不参与梯度；保存窗口内每个样本的 Q 预测和训练目标供直方图使用。
            with torch.inference_mode():
                stats['dqn_loss'] += dqn_loss
                if not online:
                    stats['cql_loss'] += cql_loss
                stats['next_rank_loss'] += next_rank_loss
                all_q[idx] = q
                all_q_target[idx] = q_target_mc

            steps += 1
            idx += 1
            if idx % opt_step_every == 0:
                if max_grad_norm > 0: # 梯度裁剪（配置中默认没开）
                    # AMP 梯度（自动混合精度训练）必须先还原尺度，裁剪阈值才对应真实梯度范数。
                    scaler.unscale_(optimizer)
                    params = chain.from_iterable(g['params'] for g in optimizer.param_groups)
                    clip_grad_norm_(params, max_grad_norm)
                scaler.step(optimizer) #
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            # scheduler 按训练 batch 前进，即使当前 batch 只累积梯度、没有更新参数。
            scheduler.step()
            pb.update(1)

            if online and steps % submit_every == 0:
                # is_idle=False 表示训练仍在消费 replay，服务端只更新可供客户端拉取的参数。
                submit_param(mortal, dqn, is_idle=False)
                logging.info('param has been submitted')

            if steps % save_every == 0:
                pb.close()

                # 固定间隔抽样，避免完整 Q 张量让 TensorBoard event 文件过度膨胀。
                all_q_1d = all_q.cpu().numpy().flatten()[::128]
                all_q_target_1d = all_q_target.cpu().numpy().flatten()[::128]

                writer.add_scalar('loss/dqn_loss', stats['dqn_loss'] / save_every, steps)
                if not online:
                    writer.add_scalar('loss/cql_loss', stats['cql_loss'] / save_every, steps)
                writer.add_scalar('loss/next_rank_loss', stats['next_rank_loss'] / save_every, steps)
                writer.add_scalar('hparam/lr', scheduler.get_last_lr()[0], steps)
                writer.add_histogram('q_predicted', all_q_1d, steps)
                writer.add_histogram('q_target', all_q_target_1d, steps)
                writer.flush()

                for k in stats:
                    stats[k] = 0
                idx = 0

                before_next_test_play = (test_every - steps % test_every) % test_every
                logging.info(f'total steps: {steps:,} (~{before_next_test_play:,})')

                # 普通 checkpoint 保存继续训练所需的模型、优化状态、统计与配置快照。
                # GRP 和 baseline 都是外部固定模型，不包含在该 checkpoint 中。
                state = {
                    'mortal': mortal.state_dict(),
                    'current_dqn': dqn.state_dict(),
                    'aux_net': aux_net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'scaler': scaler.state_dict(),
                    'steps': steps,
                    'timestamp': datetime.now().timestamp(),
                    'best_perf': best_perf,
                    # 保存当前解析后的完整配置，供恢复逻辑判断旧训练模式并追溯实验；
                    # 其中的路径仍只是字符串，不会把 GRP、baseline 或数据文件嵌入 checkpoint。
                    'config': config,
                }
                torch.save(state, state_file)

                if online and steps % submit_every != 0:
                    # 保存点也强制发布参数，避免 submit_every 较大时客户端长期拿不到新版本。
                    submit_param(mortal, dqn, is_idle=False)
                    logging.info('param has been submitted')

                if steps % test_every == 0:
                    # 每个 seed 生成四场轮换座次的对局，所以配置局数在此除以 4。
                    stat = test_player.test_play(test_games // 4, mortal, dqn, device)
                    # 评测引擎会切换模型状态；返回训练前显式恢复 train 模式。
                    mortal.train()
                    dqn.train()

                    # 该顺位点只用于展示和 best checkpoint 判断，绝不进入训练 target。
                    avg_pt = stat.avg_pt([90, 45, 0, -135])
                    # 只有平均顺位不变差且平均顺位点不降低时，才认为刷新最佳记录。
                    better = avg_pt >= best_perf['avg_pt'] and stat.avg_rank <= best_perf['avg_rank']
                    if better:
                        past_best = best_perf.copy()
                        best_perf['avg_pt'] = avg_pt
                        best_perf['avg_rank'] = stat.avg_rank

                    logging.info(f'avg rank: {stat.avg_rank:.6}')
                    logging.info(f'avg pt: {avg_pt:.6}')
                    writer.add_scalar('test_play/avg_ranking', stat.avg_rank, steps)
                    writer.add_scalar('test_play/avg_pt', avg_pt, steps)
                    writer.add_scalars('test_play/ranking', {
                        '1st': stat.rank_1_rate,
                        '2nd': stat.rank_2_rate,
                        '3rd': stat.rank_3_rate,
                        '4th': stat.rank_4_rate,
                    }, steps)
                    writer.add_scalars('test_play/behavior', {
                        'agari': stat.agari_rate,
                        'houjuu': stat.houjuu_rate,
                        'fuuro': stat.fuuro_rate,
                        'riichi': stat.riichi_rate,
                    }, steps)
                    writer.add_scalars('test_play/agari_point', {
                        'overall': stat.avg_point_per_agari,
                        'riichi': stat.avg_point_per_riichi_agari,
                        'fuuro': stat.avg_point_per_fuuro_agari,
                        'dama': stat.avg_point_per_dama_agari,
                    }, steps)
                    writer.add_scalar('test_play/houjuu_point', stat.avg_point_per_houjuu, steps)
                    writer.add_scalar('test_play/point_per_round', stat.avg_point_per_round, steps)
                    writer.add_scalars('test_play/key_step', {
                        'agari_jun': stat.avg_agari_jun,
                        'houjuu_jun': stat.avg_houjuu_jun,
                        'riichi_jun': stat.avg_riichi_jun,
                    }, steps)
                    writer.add_scalars('test_play/riichi', {
                        'agari_after_riichi': stat.agari_rate_after_riichi,
                        'houjuu_after_riichi': stat.houjuu_rate_after_riichi,
                        'chasing_riichi': stat.chasing_riichi_rate,
                        'riichi_chased': stat.riichi_chased_rate,
                    }, steps)
                    writer.add_scalar('test_play/riichi_point', stat.avg_riichi_point, steps)
                    writer.add_scalars('test_play/fuuro', {
                        'agari_after_fuuro': stat.agari_rate_after_fuuro,
                        'houjuu_after_fuuro': stat.houjuu_rate_after_fuuro,
                    }, steps)
                    writer.add_scalar('test_play/fuuro_num', stat.avg_fuuro_num, steps)
                    writer.add_scalar('test_play/fuuro_point', stat.avg_fuuro_point, steps)
                    writer.flush()

                    if better:
                        torch.save(state, state_file)
                        logging.info(
                            'a new record has been made, '
                            f'pt: {past_best["avg_pt"]:.4} -> {best_perf["avg_pt"]:.4}, '
                            f'rank: {past_best["avg_rank"]:.4} -> {best_perf["avg_rank"]:.4}, '
                            f'saving to {best_state_file}'
                        )
                        shutil.copy(state_file, best_state_file)
                    if online:
                        # 已知问题：在线训练在评测返回后继续执行会无原因卡住。正常退出当前
                        # 子进程，让 main 中的父进程重新拉起，是现有的进程级规避措施。
                        sys.exit(0)
                pb = tqdm(total=save_every, desc='TRAIN')

        # 满 batch 可直接训练；各 worker 产生的短 batch 留待数据迭代完成后统一重组。
        for obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks in data_loader:
            bs = obs.shape[0]
            if bs != batch_size: # 最后一个batch的长度可能不足batch_size
                remaining_obs.append(obs)
                remaining_actions.append(actions)
                remaining_masks.append(masks)
                remaining_steps_to_done.append(steps_to_done)
                remaining_kyoku_rewards.append(kyoku_rewards)
                remaining_player_ranks.append(player_ranks)
                remaining_bs += bs
                continue
            train_batch(obs, actions, masks, steps_to_done, kyoku_rewards, player_ranks)

        # 只切出完整 batch，末尾不足 batch_size 的样本不会跨下一次 train_epoch 保留。
        remaining_batches = remaining_bs // batch_size
        if remaining_batches > 0:
            obs = torch.cat(remaining_obs, dim=0)
            actions = torch.cat(remaining_actions, dim=0)
            masks = torch.cat(remaining_masks, dim=0)
            steps_to_done = torch.cat(remaining_steps_to_done, dim=0)
            kyoku_rewards = torch.cat(remaining_kyoku_rewards, dim=0)
            player_ranks = torch.cat(remaining_player_ranks, dim=0)
            start = 0
            end = batch_size
            while end <= remaining_bs:
                train_batch(
                    obs[start:end],
                    actions[start:end],
                    masks[start:end],
                    steps_to_done[start:end],
                    kyoku_rewards[start:end],
                    player_ranks[start:end],
                )
                start = end
                end += batch_size
        pb.close()

        if online:
            # 当前 drain 目录已消费完，通知服务端训练端空闲并发布最终参数。
            submit_param(mortal, dqn, is_idle=True)
            logging.info('param has been submitted')

    while True:
        train_epoch()
        # 一轮数据完成后主动回收 Python 对象；不强制清空 CUDA allocator 的缓存。
        gc.collect()
        # torch.cuda.empty_cache()
        # torch.cuda.synchronize()
        if not online:
            # 离线模式一次进程只运行一轮，便于由外部命令控制 epoch 和数据版本。
            break

def main():
    """直接运行离线训练，或作为父进程监管在线训练子进程。

    ``MORTAL_IS_SUB_PROC`` 仅供本模块区分在线父子进程。在线子进程正常退出时，
    父进程等待三秒后重新拉起；非零退出码则向上传递，避免隐藏真实错误。
    """
    import os
    import sys
    import time
    from subprocess import Popen
    # 在线父进程需要先读取 control.online 决定是否创建子进程。子进程是独立 Python
    # 进程，会继承环境变量和当前工作目录，并按文件顶部说明重新加载同一目标 TOML。
    from config import config

    # 此环境变量是内部进程协议，用户手动设置会绕过在线父进程监管。
    is_sub_proc_key = 'MORTAL_IS_SUB_PROC'
    # 与 train() 中的 control.online 是同一配置键；父进程必须在创建模型和 DataLoader
    # 之前读取它，以决定直接训练还是进入在线子进程监管循环。
    online = config['control']['online']
    if not online or os.environ.get(is_sub_proc_key, '0') == '1':
        train()
        return

    cmd = (sys.executable, __file__)
    env = {
        is_sub_proc_key: '1',
        **os.environ.copy(),
    }
    # 在线评测后的 sys.exit(0) 会回到这里；异常退出则保留退出码并停止重启。
    while True:
        child = Popen(
            cmd,
            stdin = sys.stdin,
            stdout = sys.stdout,
            stderr = sys.stderr,
            env = env,
        )
        if (code := child.wait()) != 0:
            sys.exit(code)
        time.sleep(3)

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        # 用户中断属于正常停止操作，不输出额外 traceback。
        pass
