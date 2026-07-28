# Mortal 训练指南

本文说明仓库内正式入口的训练依赖、执行顺序和配置边界。模型结构见 [network.md](network.md)，GRP 样本见 [train_grp_1.md](train_grp_1.md)，奖励公式见 [rewards.md](rewards.md)。

## 先选择训练任务

| 目标 | 入口 | 配置节 | 主要产物 |
| --- | --- | --- | --- |
| 训练 GRU 顺位模型 | [`train_grp.py`](../mortal/train_grp.py) | `[grp]` | GRP checkpoint |
| 训练 Transformer 顺位模型 | [`train_grp_transformer.py`](../mortal/train_grp_transformer.py) | `[grp_transformer]` | 独立 checkpoint |
| 离线训练主策略 | [`train.py`](../mortal/train.py)，`online=false` | 完整主配置 | `Brain + DQN + AuxNet` checkpoint |
| 在线训练主策略 | `server.py + client.py + train.py` | 完整主配置 | 持续更新的主策略 checkpoint |

标准主策略数据加载器固定使用 GRU 版 `GRP` 计算奖励。训练好的 `GRP_Transformer` 不能仅通过修改配置路径替代它；需要同时更改数据加载器的模型类和 checkpoint 配置。

## 配置加载规则

[`mortal/config.py`](../mortal/config.py) 读取：

1. `MORTAL_CFG` 指定的 TOML；
2. 否则读取当前工作目录的 `config.toml`。

建议始终从仓库根目录运行，并显式给出绝对配置路径：

```bash
MORTAL_CFG=/absolute/path/config.toml python mortal/train_grp.py
MORTAL_CFG=/absolute/path/config.toml python mortal/train.py
```

注意：

- [`mortal/config.example.toml`](../mortal/config.example.toml) 是完整键参考，路径和部分数值只是占位示例；
- 根目录 [`config.toml`](../config.toml) 当前只有 `grp` 与 `grp_transformer`，不能直接运行 `train.py`、`mortal.py`、`server.py` 或 `client.py`；
- 相对路径按进程的当前工作目录解析，不按配置文件所在目录解析；
- checkpoint 一律使用 `torch.load(..., weights_only=True)`。

## 前置条件

### 共通环境

- 已安装与硬件匹配的 PyTorch 和 NumPy；
- 已构建并复制 `libriichi` Python 扩展；
- 训练数据是 `libriichi` 可解析的逐行 mjai `.json.gz` 日志；
- 所有输出目录的父目录可写，配置中的数据和权重路径真实存在。

### 各任务依赖

| 任务 | 必需输入 |
| --- | --- |
| GRP / GRP Transformer | 训练日志、验证日志 |
| 主策略离线训练 | 离线日志、GRU 版 GRP checkpoint、测试 baseline checkpoint |
| 主策略在线训练 | GRU 版 GRP checkpoint、训练 baseline、测试 baseline、参数服务和工作目录 |

`train.py` 启动时立即构造 `TestPlayer` 并加载测试 baseline，即使距离第一次 `test_every` 评测还很远。因此 baseline 不是可延后准备的依赖。

当前主训练循环把 `Brain(obs)` 当作单个 1024 维 `phi`，`AuxNet` 也固定接收 1024 维输入；返回 `(mu, logsig)` 的 v1 不能直接由现有 `train.py` 训练。v1 推理仍有专门分支，但要恢复 v1 训练需另行适配训练循环。标准主训练配置应使用 v2–v4。

## 顺位模型训练

两种顺位模型共用同一种监督任务：把每场对局按 `start_kyoku` 提取为 7 维序列，再将所有历史前缀作为样本，预测最终四家顺位的 24 种排列。

### 必需配置

GRU 版最小结构如下：

```toml
[grp]
state_file = '/path/to/grp.pth'

[grp.network]
hidden_size = 64
num_layers = 2

[grp.control]
device = 'cuda:0'
enable_cudnn_benchmark = false
tensorboard_dir = '/path/to/tensorboard/grp'
batch_size = 512
save_every = 2000
val_steps = 400

[grp.dataset]
train_globs = ['/data/train/**/*.json.gz']
val_globs = ['/data/val/**/*.json.gz']
file_index = '/path/to/grp_file_index.pth'
file_batch_size = 50

[grp.optim]
lr = 1e-5
```

Transformer 版使用相同的 `control`、`dataset` 和 `optim` 形状，但顶层改为 `[grp_transformer]`，网络参数是 `d_model`、`nhead`、`num_layers`、`max_seq_len`。未配置的参数使用 [`GRP_Transformer`](../mortal/model.py) 构造函数默认值。

### 执行和恢复

```bash
MORTAL_CFG=/absolute/path/config.toml python mortal/train_grp.py
MORTAL_CFG=/absolute/path/config.toml python mortal/train_grp_transformer.py
```

两个脚本都会：

1. 构建或读取训练/验证文件索引；
2. 创建模型和 `AdamW`；
3. 若 `state_file` 已存在，恢复模型、优化器和 `steps`；
4. 无限循环训练数据，并每 `save_every` 步做固定 `val_steps` 次验证；
5. 写入 loss/accuracy 的 TensorBoard 数据并覆盖各自 checkpoint。

如果索引中记录的文件已经缺失，脚本会重建索引；它不会检测所有新增文件或其他数据内容变化。想重新扫描完整数据集时，应明确处理目标索引文件，不要误删其他实验索引。

GRU 在 CUDA 不可用时会回退到 CPU；Transformer 训练脚本采用相同回退逻辑。两者都使用 `float64`，CPU 训练可能很慢。

## 主策略离线训练

### 依赖关系

```text
已训练 GRU GRP
       |
离线日志 -> GameplayLoader -> RewardCalculator -> 动作样本
                                            |
                              Brain + DQN + AuxNet
                                            |
                        checkpoint + 定期 baseline 评测
```

`train.py` 更新 `Brain`、`DQN` 和 `AuxNet`；GRP 与 baseline 只加载和推理，不进入优化器。

### 数据准备

主配置至少要完整覆盖以下路径：

| 配置键 | 内容 |
| --- | --- |
| `dataset.globs` | 离线 `.json.gz` 日志 |
| `dataset.file_index` | 主训练文件列表缓存 |
| `dataset.player_names_files` | 可选玩家名白名单文件 |
| `grp.state_file` | GRU 版 GRP checkpoint |
| `baseline.test.state_file` | 定期评测对手 checkpoint |
| `control.state_file` | 当前训练状态 |
| `control.best_state_file` | 最佳评测状态副本 |
| `control.tensorboard_dir` | 训练指标目录 |
| `test_play.log_dir` | 定期评测日志目录 |

当 `dataset.file_index` 不存在时，脚本扫描 `dataset.globs` 并保存索引。如果配置了玩家白名单，还会打开每个 gzip 日志的首行，根据 `start_game.names` 过滤文件。已有索引会直接复用，不会自动发现新增日志。

### 训练顺序

运行：

```bash
MORTAL_CFG=/absolute/path/full-config.toml python mortal/train.py
```

`control.online=false` 时，流程如下：

1. 创建 `Brain`、`DQN`、`AuxNet`，按需调用 `compile()`；
2. 设置 BatchNorm 行为，创建 AdamW、学习率调度器和 AMP `GradScaler`；
3. 若 `control.state_file` 存在，恢复 checkpoint；
4. 加载测试 baseline；
5. 构建或读取日志文件索引；
6. 每个 DataLoader worker 加载一份 GRP 和 `RewardCalculator`；
7. `GameplayLoader` 解析日志并按动作组装样本；
8. 执行前向、反向、梯度累积/裁剪和优化；
9. 每 `save_every` 步保存并写 TensorBoard；
10. 每 `test_every` 步对 baseline 做一对三评测；
11. `train_epoch()` 完成后退出。

外层离线模式只调用一次 `train_epoch()`，但 `dataset.num_epochs` 决定数据迭代器内部重复文件列表多少次；开启 augmentation 还会追加一轮增强/非增强数据。因此“外层一个 epoch”不一定等于“每个文件只读取一次”。

### 单个 batch

每条样本包含：

```text
obs, action, mask, steps_to_done, kyoku_reward, player_rank
```

当前入口没有启用 oracle，所以不会产生 `invisible_obs`。训练计算：

```text
phi = Brain(obs)
q_all = DQN(phi, mask)
q = q_all[actual_action]
q_target = gamma ** steps_to_done * kyoku_reward

dqn_loss = 0.5 * MSE(q, q_target)
cql_loss = logsumexp(q_all) - mean(q)      # 仅离线
next_rank_loss = CrossEntropy(AuxNet(phi), player_rank)
```

总损失是三项的加权和。`opt_step_every` 用于梯度累积；loss 会先除以该值再反向。scheduler 每个 batch 都前进一步，不是每次 `optimizer.step()` 才前进。

### 保存、恢复和 best

普通 checkpoint 包含：

- `mortal`、`current_dqn`、`aux_net`；
- `optimizer`、`scheduler`、`scaler`；
- `steps`、`timestamp`、`best_perf`；
- 本次运行的完整 `config` 快照。

从离线 checkpoint 切到在线训练时，优化器和 scheduler 不一定恢复：只有当前是离线模式，或旧 checkpoint 的 `control.online` 为真时才恢复它们。模型、scaler、步数和 best 指标仍会加载。

每次评测计算展示用的 `[90, 45, 0, -135]` pt。只有同时满足 `avg_pt` 不下降且 `avg_rank` 不上升时，才把当前状态复制到 `best_state_file`。

## 主策略在线训练

在线训练至少需要三个长期进程，共用一份完整配置：

```text
server.py：保存参数，接收和转移 replay
client.py：拉参数，与 baseline 自对战，上传日志
train.py：drain 日志，训练并提交新参数
```

推荐启动顺序是：

1. 核对所有工作目录；
2. 启动 `server.py`；
3. 启动 `train.py`，让服务端拿到第一版参数；
4. 启动一个或多个 `client.py`。

对应命令为：

```bash
MORTAL_CFG=/absolute/path/online.toml python mortal/server.py
MORTAL_CFG=/absolute/path/online.toml python mortal/train.py
MORTAL_CFG=/absolute/path/online.toml TRAIN_PLAY_PROFILE=default python mortal/client.py
```

`client.py` 使用 `TRAIN_PLAY_PROFILE` 选择 `[train_play.<profile>]`，默认为 `default`。每轮拉取参数，调用 `TrainPlayer` 生成日志，再整体上传。

`train.py` 在线模式会：

- 启动时提交当前参数；
- 阻塞等待 `drain()` 返回新 replay；
- 只选择日志中名为 `trainee` 的玩家视角；
- 不计算 CQL loss；
- 每 `submit_every` 步提交新参数；
- 在一次定期评测后主动退出子进程，由父进程重新启动，以规避源码注释记录的卡住问题。

### 在线目录风险

这些入口会删除配置目录中的已有内容：

- `server.py` 启动时递归删除并重建 `online.server.buffer_dir` 和 `drain_dir`；
- `TrainPlayer.train_play()` 每轮删除并重建当前 profile 的 `train_play.*.log_dir`；
- `TestPlayer.test_play()` 在评测前删除 `test_play.log_dir`。

这些路径必须是专用临时目录，不能指向数据集、checkpoint、仓库或共享日志目录。没有核对路径前不要运行 server、client 或会触发对战的训练流程。

## 配置速查

| 配置节 | 关键键 | 作用 |
| --- | --- | --- |
| `[control]` | `version`、`online`、设备、batch、保存/测试/提交间隔 | 训练总控 |
| `[dataset]` | `globs`、`file_index`、worker、epoch、augmentation | 主训练数据 |
| `[env]` | `gamma`、`pts` | 回报折扣和顺位价值 |
| `[resnet]` | `conv_channels`、`num_blocks` | `Brain` 规模 |
| `[cql]` | `min_q_weight` | 离线保守项权重 |
| `[aux]` | `next_rank_weight` | 辅助分类权重 |
| `[freeze_bn]` | `mortal` | 是否固定 BN 运行统计 |
| `[optim]` | AdamW、梯度裁剪参数 | 优化器 |
| `[optim.scheduler]` | `peak`、`final`、warmup、最大步数 | 学习率曲线 |
| `[baseline.train/test]` | 设备、编译、checkpoint | 在线对手和评测对手 |
| `[train_play.*]` | 对局数、日志目录、探索参数 | 在线 replay 生成 |
| `[test_play]` | 对局数、日志目录 | 定期评测 |
| `[online.remote]` | host、port | 参数服务地址 |
| `[online.server]` | buffer/drain、容量、顺序模式 | replay 缓冲 |

配置键的完整类型和默认示例以 [`config.example.toml`](../mortal/config.example.toml) 为准。

训练入口还要求 `save_every` 能被 `opt_step_every` 整除，且 `test_every` 能被 `save_every` 整除；不满足时会在启动阶段触发断言。

## 监控与运行检查

TensorBoard 可指向具体实验目录：

```bash
tensorboard --logdir /path/to/tensorboard --port 6006
```

开始长时间训练前至少检查：

1. `MORTAL_CFG` 指向预期文件，且从预期工作目录执行；
2. 数据 glob 非空，训练集和验证集没有误重叠；
3. GRP、baseline 和恢复 checkpoint 的结构与配置匹配；
4. checkpoint、TensorBoard、文件索引和对战日志路径互不覆盖；
5. 在线模式的三个可清理目录都是专用目录；
6. GPU、CUDA、AMP 和 `torch.compile` 组合在当前环境可用；
7. 先做只读配置/索引核对和小规模目标验证，再启动数天训练。

完整训练、自对战和服务器启动都不是普通冒烟测试：它们会大量占用计算资源并写入或删除配置指定的目录。
