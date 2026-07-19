# Mortal 训练流程说明

这份文档总结的是当前仓库里 **主干网络 `Brain + DQN + AuxNet` 的实际训练流程**，重点回答下面几个问题：

- Mortal 在训练前需要先准备什么
- 训练的具体顺序是怎样的
- 每一步到底做了什么
- 各种超参数从哪里来、控制什么
- `oracle` 在当前训练代码里是否真的启用

对应代码主要在：

- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)
- [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py)
- [mortal/player.py](/home/tml104/Mortal/mortal/player.py)
- [mortal/client.py](/home/tml104/Mortal/mortal/client.py)
- [mortal/server.py](/home/tml104/Mortal/mortal/server.py)
- [mortal/config.example.toml](/home/tml104/Mortal/mortal/config.example.toml)

conda run -n mortal tensorboard --logdir ./artifacts/tensorboard/grp --port 6006
conda run -n mortal tensorboard --logdir ./artifacts/tensorboard/grp_transformer/ --port 6006

---

## 1. 先说结论：默认训练顺序是什么

按当前仓库实现，主干网络训练的默认顺序是：

1. 先准备好一个可用的 `GRP`
2. 再准备训练日志数据集或在线自博弈数据
3. 然后运行 `train.py` 训练 `Brain + DQN + AuxNet`
4. 训练过程中定期保存 checkpoint
5. 定期和 baseline 打 `test_play`
6. 如果表现更好，则拷贝为 `best_state_file`

如果是离线训练：

- 数据来自本地日志文件
- 默认只跑一个 epoch

如果是在线训练：

- 需要额外启动 `server.py` 和若干 `client.py`
- `client` 负责拿当前参数去打自博弈
- `train.py` 从服务端 drain 到新日志后继续训练

所以最简化地理解：

> Mortal 的训练不是“直接端到端喂环境”，而是“先把日志变成监督样本，再按 batch 训练一个 Q 网络 + 辅助头”。

---

## 2. 主干训练前需要准备什么

主干训练开始前，至少需要准备下面几类东西。

### 2.1 一个已训练好的 GRP

`dataloader.py` 在构建数据迭代器时会直接加载：

```python
grp_state = torch.load(config['grp']['state_file'], ...)
```

然后用它构造 `RewardCalculator`。

所以如果没有 `config['grp']['state_file']` 对应的模型文件，主干训练无法按当前实现正常进行。

这也是为什么要先有 `GRP`。

### 2.2 一份配置文件

训练配置通过：

- 环境变量 `MORTAL_CFG`
- 或默认的 `config.toml`

来读取。

代码入口在 [mortal/config.py](/home/tml104/Mortal/mortal/config.py)。

### 2.3 训练数据来源

分两种：

- 离线训练：本地日志文件
- 在线训练：由 `client.py` 自博弈生成并上传给 `server.py`

### 2.4 baseline 模型

`train.py` 中会创建 `TestPlayer()`，它会加载：

- `config['baseline']['test']['state_file']`

如果启用了在线训练，`TrainPlayer()` 还会加载：

- `config['baseline']['train']['state_file']`

因此训练流程默认还依赖一个 baseline 模型，用于：

- 离线训练中的定期评测
- 在线训练中的对手模型

### 2.5 三种训练方式分别要准备什么、放在哪里、怎么执行

这一小节只讲实操层面最关心的三件事：

- 需要准备什么数据
- 这些数据在配置里应该挂到哪里
- 最后运行哪个脚本

#### A. `GRP` 模型训练

`GRP` 训练对应脚本是：

- [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py)

需要准备的数据内容：

- 一批 `.json.gz` 对局日志
- 这些日志需要能被 `libriichi.dataset.Grp` 读取
- 也就是说，本质上还是项目使用的 mjai / gameplay 日志数据

这些数据在配置里放在哪里：

- 训练集路径放到：
  - `config['grp']['dataset']['train_globs']`
- 验证集路径放到：
  - `config['grp']['dataset']['val_globs']`
- 索引缓存路径放到：
  - `config['grp']['dataset']['file_index']`

最小配置关注项是：

```toml
[grp]
state_file = '/path/to/grp.pth'

[grp.dataset]
train_globs = ['/path/to/dataset/train/**/*.json.gz']
val_globs = ['/path/to/dataset/val/**/*.json.gz']
file_index = '/path/to/grp_file_index.pth'
file_batch_size = 50
```

训练产物会放在哪里：

- `GRP` 模型权重保存到 `config['grp']['state_file']`
- TensorBoard 日志写到 `config['grp']['control']['tensorboard_dir']`

怎么执行：

如果你在仓库根目录下，一般直接运行：

```bash
python mortal/train_grp.py
```

如果你使用自定义配置文件，则可以：

```bash
MORTAL_CFG=/path/to/config.toml python mortal/train_grp.py
```

#### B. 主干离线训练

主干离线训练对应脚本是：

- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)

需要准备的数据内容：

1. 一份已经训练好的 `GRP` 权重
2. 一批离线 `.json.gz` 日志
3. 一个 baseline 权重

这些数据在配置里放在哪里：

- `GRP` 权重：
  - `config['grp']['state_file']`
- 离线日志路径：
  - `config['dataset']['globs']`
- 文件索引缓存：
  - `config['dataset']['file_index']`
- 可选玩家过滤文件：
  - `config['dataset']['player_names_files']`
- 测试 baseline：
  - `config['baseline']['test']['state_file']`
- 当前训练 checkpoint：
  - `config['control']['state_file']`
- 最优 checkpoint：
  - `config['control']['best_state_file']`

最小配置关注项是：

```toml
[control]
online = false
state_file = '/path/to/mortal.pth'
best_state_file = '/path/to/best.pth'

[dataset]
globs = ['/path/to/dataset/**/*.json.gz']
file_index = '/path/to/file_index.pth'

[grp]
state_file = '/path/to/grp.pth'

[baseline.test]
state_file = '/path/to/baseline.pth'
```

训练时数据实际放在哪里：

- 日志文件放在任意目录都可以
- 关键不是固定目录名，而是 `dataset.globs` 能匹配到它们

怎么执行：

```bash
python mortal/train.py
```

如果使用自定义配置文件：

```bash
MORTAL_CFG=/path/to/config.toml python mortal/train.py
```

#### C. 主干在线训练

在线训练涉及三个脚本：

- [mortal/server.py](/home/tml104/Mortal/mortal/server.py)
- [mortal/client.py](/home/tml104/Mortal/mortal/client.py)
- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)

需要准备的数据内容：

1. 一份已经训练好的 `GRP` 权重
2. 一个训练用 baseline 权重
3. 一个测试用 baseline 权重
4. 不需要预先准备离线训练日志
   因为在线模式的数据由 `client.py` 自博弈生成

这些路径在配置里放在哪里：

- `GRP` 权重：
  - `config['grp']['state_file']`
- 训练 baseline：
  - `config['baseline']['train']['state_file']`
- 测试 baseline：
  - `config['baseline']['test']['state_file']`
- server 缓冲目录：
  - `config['online']['server']['buffer_dir']`
- server drain 目录：
  - `config['online']['server']['drain_dir']`
- 自博弈日志目录：
  - `config['train_play'][profile]['log_dir']`

最小配置关注项是：

```toml
[control]
online = true

[grp]
state_file = '/path/to/grp.pth'

[baseline.train]
state_file = '/path/to/baseline_train.pth'

[baseline.test]
state_file = '/path/to/baseline_test.pth'

[online.server]
buffer_dir = '/path/to/buffer'
drain_dir = '/path/to/drain'

[train_play.default]
log_dir = '/path/to/train_play'
games = 800
```

这些数据和目录实际怎么放：

- baseline / GRP 权重放在任意你配置的路径即可
- `buffer_dir` 和 `drain_dir` 是 server 自己维护的工作目录
- `train_play.*.log_dir` 是 client 每轮自博弈临时写日志的目录

在线训练通常按下面顺序启动：

1. 先启动参数服务器

```bash
python mortal/server.py
```

2. 再启动一个或多个自博弈 worker

```bash
python mortal/client.py
```

如果你要切换训练 profile：

```bash
TRAIN_PLAY_PROFILE=default python mortal/client.py
```

3. 最后启动训练进程

```bash
python mortal/train.py
```

如果三者都使用同一份自定义配置文件，可以分别这样跑：

```bash
MORTAL_CFG=/path/to/config.toml python mortal/server.py
MORTAL_CFG=/path/to/config.toml python mortal/client.py
MORTAL_CFG=/path/to/config.toml python mortal/train.py
```

这里需要特别注意：

- 在线训练不是只跑一个命令
- 至少要有 `server + train + client`
- 可以有多个 `client` 并行产生日志

---

## 3. 训练主干时到底训练哪些网络

在 [mortal/train.py](/home/tml104/Mortal/mortal/train.py) 里，真正被创建并优化的是：

- `mortal = Brain(...)`
- `dqn = DQN(...)`
- `aux_net = AuxNet((4,))`

也就是说，主干训练更新的是：

- `Brain`
- `DQN`
- `AuxNet`

不会在 `train.py` 中更新的包括：

- `GRP`
- baseline 模型

其中各模块的角色是：

- `Brain`
  - 把麻将状态编码成高层表示
- `DQN`
  - 给动作打 Q 值
- `AuxNet`
  - 预测 next rank，提供辅助监督

---

## 4. 离线训练的完整顺序

这一节讲默认离线模式，也就是：

```toml
[control]
online = false
```

时，`train.py` 实际按什么顺序工作。

### 第 1 步：读取配置

训练开始后先从配置中取出关键超参数，例如：

- `version`
- `batch_size`
- `opt_step_every`
- `save_every`
- `test_every`
- `device`
- `enable_amp`
- `enable_compile`
- `pts`
- `gamma`
- `file_batch_size`
- `num_workers`
- `weight_decay`
- `max_grad_norm`

也就是说，训练行为基本是配置驱动的。

### 第 2 步：构造模型

代码会实例化：

- `Brain(version=version, **config['resnet'])`
- `DQN(version=version)`
- `AuxNet((4,))`

如果 `enable_compile=true`，还会对这些模型执行：

```python
m.compile()
```

### 第 3 步：设置 BatchNorm、优化器、学习率调度器、AMP

训练里接着会：

- 根据 `freeze_bn.mortal` 决定是否冻结 `Brain` 中 BatchNorm 的运行方式
- 把参数拆成 `decay_params` 和 `no_decay_params`
- 创建 `AdamW`
- 创建 `LinearWarmUpCosineAnnealingLR`
- 创建 `GradScaler`

参数分组规则是：

- `Linear` 和 `Conv1d` 的 `weight` 进入 weight decay
- 其他参数不做 decay

### 第 4 步：尝试从 checkpoint 恢复

如果 `control.state_file` 已存在，就会恢复：

- `mortal`
- `dqn`
- `aux_net`
- `optimizer`
- `scheduler`
- `scaler`
- `best_perf`
- `steps`

这说明训练默认支持断点续训。

### 第 5 步：准备评测器

训练开始时会创建：

- `TestPlayer()`

它会加载测试用 baseline，用来在 `test_every` 时做定期评测。

### 第 6 步：构建文件列表

离线训练时，`train_epoch()` 会：

1. 从 `dataset.player_names_files` 加载玩家名过滤集合
2. 根据 `dataset.globs` 遍历日志路径
3. 如果存在 `dataset.file_index`，直接读取缓存好的文件列表
4. 否则重新扫描并建立索引

如果设置了玩家名过滤，就只保留包含目标玩家的日志文件。

### 第 7 步：构建数据集迭代器

接着训练会创建：

- `FileDatasetsIter(...)`
- `torch.utils.data.DataLoader(...)`

这里 `FileDatasetsIter` 是真正的数据核心。它会：

1. 加载一个已训练好的 `GRP`
2. 创建 `RewardCalculator`
3. 用 `GameplayLoader` 读取日志
4. 把每一步动作转换成训练样本

每条样本里包含：

- `obs`
- `actions`
- `masks`
- `steps_to_done`
- `kyoku_rewards`
- `player_ranks`

### 第 8 步：开始按 batch 训练

训练循环里，核心逻辑在 `train_batch(...)`。

每个 batch 会：

1. 把数据搬到目标设备
2. 计算：
   - `phi = mortal(obs)`
   - `q_out = dqn(phi, masks)`
   - `q = q_out[range(batch_size), actions]`
3. 计算 Monte Carlo 风格目标：
   - `q_target_mc = gamma ** steps_to_done * kyoku_rewards`
4. 计算损失：
   - `dqn_loss`
   - `cql_loss`
   - `next_rank_loss`
5. 反向传播
6. 按 `opt_step_every` 决定是否真正执行一次 `optimizer.step()`
7. 每步都执行 `scheduler.step()`

### 第 9 步：定期保存与记录

每当 `steps % save_every == 0`，会做这些事情：

- 写 TensorBoard：
  - `dqn_loss`
  - `cql_loss`
  - `next_rank_loss`
  - 学习率
  - Q 值分布
- 把当前状态保存到 `state_file`

保存内容包括：

- 三个模型的参数
- 优化器状态
- scheduler 状态
- scaler 状态
- 当前 step
- 配置快照

### 第 10 步：定期评测

每当 `steps % test_every == 0`，会调用：

```python
test_player.test_play(...)
```

它会让当前模型和 baseline 在 `OneVsThree` 环境里打若干局。

记录指标包括：

- 平均排名
- 平均 pt
- 各名次率
- 和牌率、放铳率、副露率、立直率
- 各类点数与时机统计

如果新模型满足：

- `avg_pt >= best_perf['avg_pt']`
- 且 `avg_rank <= best_perf['avg_rank']`

则会把当前 `state_file` 复制成 `best_state_file`。

### 第 11 步：离线模式默认只跑一个 epoch

`train.py` 最后有一段很关键：

```python
while True:
    train_epoch()
    ...
    if not online:
        break
```

所以离线模式下默认行为是：

- 跑一个 `train_epoch()`
- 然后退出

这里的“一个 epoch”到底覆盖多少数据，还要结合：

- `dataset.num_epochs`
- `FileDatasetsIter`

来看。

它并不一定等于“全量文件只过一次”，而是“按 `FileDatasetsIter` 当前设定跑完一次”。

---

## 5. 在线训练的完整顺序

在线模式是：

```toml
[control]
online = true
```

这时训练不是只读本地静态日志，而是和 `server.py` / `client.py` 配合工作。

### 5.1 参与者

在线训练至少有三个角色：

1. `server.py`
   - 参数服务器
   - 保存当前参数版本
   - 接收 replay 日志
2. `client.py`
   - 自博弈 worker
   - 拉取最新参数
   - 对战生成日志
   - 上传日志
3. `train.py`
   - 主训练进程
   - 从 server drain 数据后继续训练

### 5.2 在线训练顺序

整体流程是：

```text
train.py 启动
-> 先把当前参数提交给 server
-> client.py 从 server 拉参数
-> client 自博弈生成日志
-> client 把日志提交回 server
-> train.py 调用 drain() 取回日志
-> train.py 用这些日志继续训练
-> train.py 再把新参数提交给 server
-> 进入下一轮
```

### 5.3 `drain()` 的作用

[mortal/common.py](/home/tml104/Mortal/mortal/common.py) 中：

- `drain()` 会不断向 server 请求可训练日志
- 直到 server 返回非空结果

因此在线训练下，`train.py` 实际是在等新的 replay 日志到来。

### 5.4 `client.py` 具体做什么

`client.py` 的流程是：

1. 连接 server，请求最新参数
2. 把参数加载进本地 `Brain` 和 `DQN`
3. 调用 `TrainPlayer.train_play(...)`
4. 用当前模型与 baseline 打若干局
5. 收集生成的日志文件
6. 再提交回 server

也就是说，在线训练里的数据并不是人工给定日志，而是由当前模型不断和 baseline 对战自己生成出来的。

### 5.5 在线训练中的一个特殊实现

`train.py` 的 `main()` 在 online 模式下不会直接裸跑训练，而是：

- 启动一个子进程去执行训练
- 子进程退出后再重启

代码注释说明原因是：

- 在线训练在某处会卡住
- 所以作者用子进程重启的方式规避这个问题

这说明当前在线训练实现是“能用，但带有工程性 workaround”的状态。

---

## 6. 数据是怎么从日志变成训练样本的

这部分在 [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py) 中。

### 6.1 `GameplayLoader` 先解析日志

每局日志会被解析出：

- `obs`
- `actions`
- `masks`
- `at_kyoku`
- `dones`
- `apply_gamma`

这些都是“按动作”的数据。

### 6.2 奖励由 `GRP` + `RewardCalculator` 生成

接着还会提取：

- `grp_feature`
- `rank_by_player`
- `final_scores`

然后通过：

- `GRP`
- `RewardCalculator`

计算出：

- `kyoku_rewards`

这部分的细节已经在 [rewards.md](/home/tml104/Mortal/docs2/rewards.md) 里展开。

### 6.3 每条训练样本的组成

最终每条 entry 是：

- `obs[i]`
- `actions[i]`
- `masks[i]`
- `steps_to_done[i]`
- `kyoku_rewards[at_kyoku[i]]`
- `player_ranks[at_kyoku[i] + 1]`

如果启用了 oracle，还会额外插入：

- `invisible_obs[i]`

但请注意，当前 `train.py` 默认并没有启用 oracle。

---

## 7. `oracle` 在当前训练代码里到底有没有启用

代码层面，系统是支持 oracle 训练的：

- `Brain` 支持 `is_oracle=True`
- `dataloader.py` 支持 `oracle=True`
- `GameplayLoader` 也支持 `oracle=True`

如果启用 oracle，数据集会额外提供：

- `invisible_obs`

主干输入会变成：

- 普通观测 `obs`
- 加上不可见信息 `invisible_obs`

但是，**当前仓库的主训练入口 `train.py` 并没有把 oracle 打开**。

关键证据是：

- `FileDatasetsIter(...)` 在 `train.py` 里创建时，没有传 `oracle=True`
- `Brain(...)` 在 `train.py` 里创建时，也没有传 `is_oracle=True`
- `player.py`、`one_vs_three.py`、`mortal.py` 这些常用入口也都写死了 `is_oracle=False`

因此对当前仓库默认训练流程，最准确的说法是：

- 代码支持 oracle
- 但默认主干训练流程并未启用 oracle

换句话说，oracle 更像一个“框架能力”，而不是这个仓库默认训练脚本正在使用的主路径。

---

## 8. 训练时主要超参数都从哪里来

下面按模块总结。

### 8.1 `[control]`

控制训练总流程：

- `version`
  - 模型版本号
- `online`
  - 是否在线训练
- `state_file`
  - 当前 checkpoint 路径
- `best_state_file`
  - 最优 checkpoint 路径
- `tensorboard_dir`
  - TensorBoard 日志目录
- `device`
  - 训练设备
- `enable_cudnn_benchmark`
  - 是否开启 cudnn benchmark
- `enable_amp`
  - 是否开启混合精度
- `enable_compile`
  - 是否 `torch.compile()`
- `batch_size`
  - batch 大小
- `opt_step_every`
  - 梯度累积步数
- `save_every`
  - 每多少 step 保存一次
- `test_every`
  - 每多少 step 测试一次
- `submit_every`
  - 在线训练每多少 step 向 server 提交一次参数

### 8.2 `[dataset]`

控制数据集构造：

- `globs`
  - 离线日志文件路径模式
- `file_index`
  - 缓存的文件索引
- `file_batch_size`
  - 每次载入多少日志文件做解析
- `reserve_ratio`
  - buffer 中保留多少比例暂不立即吐出
- `num_workers`
  - DataLoader worker 数
- `player_names_files`
  - 玩家名过滤列表
- `num_epochs`
  - `FileDatasetsIter` 内部循环次数
- `enable_augmentation`
  - 是否启用日志增强
- `augmented_first`
  - 是否先跑增强版数据

### 8.3 `[env]`

控制回报定义：

- `gamma`
  - 折扣因子
- `pts`
  - 排名价值映射

### 8.4 `[resnet]`

控制 `Brain` 主干规模：

- `conv_channels`
- `num_blocks`

### 8.5 `[cql]`

控制离线训练中的 CQL 正则：

- `min_q_weight`

### 8.6 `[aux]`

控制辅助头损失权重：

- `next_rank_weight`

### 8.7 `[freeze_bn]`

控制 BatchNorm 行为：

- `mortal`

注释里明确写了：

- 在线训练时常设为 `true`

### 8.8 `[optim]`

控制优化器：

- `eps`
- `betas`
- `weight_decay`
- `max_grad_norm`

### 8.9 `[optim.scheduler]`

控制学习率调度：

- `peak`
- `final`
- `warm_up_steps`
- `max_steps`

调度器实现是：

- 先线性 warmup
- 再 cosine annealing

### 8.10 `[baseline]`

控制对手模型：

- `baseline.train`
  - 在线自博弈时的 baseline
- `baseline.test`
  - 定期测试时的 baseline

### 8.11 `[train_play.*]`

控制在线训练时 `client.py` 自博弈参数：

- `games`
- `log_dir`
- `boltzmann_epsilon`
- `boltzmann_temp`
- `top_p`
- `repeats`

其中 `TRAIN_PLAY_PROFILE` 环境变量决定用哪个 profile。

### 8.12 `[online]`

控制在线训练基础设施：

- `history_window`
- `online.remote.host`
- `online.remote.port`
- `online.server.buffer_dir`
- `online.server.drain_dir`
- `online.server.capacity`
- `online.server.force_sequential`

---

## 9. 训练时每个损失项是怎么用的

在 `train_batch(...)` 中，总损失是：

```text
loss = dqn_loss
     + min_q_weight * cql_loss
     + next_rank_weight * next_rank_loss
```

具体地：

- `dqn_loss`
  - 主损失
  - `0.5 * MSE(q, q_target_mc)`
- `cql_loss`
  - 只在离线训练时启用
  - 在线训练时直接设为 0
- `next_rank_loss`
  - 辅助分类损失

这意味着：

- 离线训练比在线训练多一个保守 Q 正则
- 在线训练主要依赖主损失和辅助损失

---

## 10. 训练时如何保存、恢复和选 best

### 10.1 恢复

如果 `state_file` 已存在，就从里面恢复：

- 模型参数
- 优化器状态
- scheduler 状态
- scaler 状态
- 已训练 step
- 历史 best 指标

### 10.2 普通保存

每次到 `save_every`，就把当前状态写回 `state_file`。

### 10.3 best 保存

每次到 `test_every` 后，若当前模型同时满足：

- 平均 pt 不差于历史 best
- 平均 rank 不差于历史 best

就把当前 checkpoint 复制到 `best_state_file`。

这是一个双条件筛选，而不是只看单一指标。

---

## 11. 一张总流程图

### 11.1 离线训练

```text
先训练好 GRP
-> 准备 config.toml
-> 准备离线日志
-> train.py 读取配置
-> 创建 Brain / DQN / AuxNet
-> 可选恢复 checkpoint
-> 构造 FileDatasetsIter
-> dataloader 加载 GRP 并计算奖励
-> 按 batch 训练
-> 每 save_every 保存
-> 每 test_every 和 baseline 对打测试
-> 满足条件则更新 best_state_file
-> 离线模式默认跑完一个 train_epoch 后退出
```

### 11.2 在线训练

```text
先训练好 GRP
-> 启动 server.py
-> 启动若干 client.py
-> train.py 启动并提交初始参数
-> client 拉取参数并和 baseline 自博弈
-> client 把日志上传给 server
-> train.py drain 日志
-> dataloader 构造样本并训练
-> train.py 提交新参数
-> client 拉新参数继续自博弈
-> 循环
```

---

## 12. 一句话总结

如果只记一句话，可以记成：

> Mortal 的主干训练本质上是一个“配置驱动的、基于日志样本的 Q 学习流程”：先用 `GRP` 把对局日志变成带奖励的监督样本，再按 batch 训练 `Brain + DQN + AuxNet`，并定期与 baseline 对战评估；默认训练路径不启用 oracle，但代码框架本身支持 oracle。 
