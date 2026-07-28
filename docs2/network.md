# Mortal 网络结构

模型定义集中在 [`mortal/model.py`](../mortal/model.py)。先记住各模块的边界：

| 模块 | 输入 | 输出 | 是否参与普通决策 |
| --- | --- | --- | --- |
| `Brain` | 当前状态观测 | 状态表示 | 是 |
| `DQN` | 状态表示、合法动作 mask | 46 个动作的 Q 值 | 是 |
| `AuxNet` | `Brain` 的状态表示 | 下一局面顺位分类 logits | 否，仅训练 |
| `GRP` | 按局推进的 7 维序列 | 最终顺位排列 logits | 否，用于奖励和复盘 |
| `GRP_Transformer` | 同上 | 最终顺位排列 logits | 否，实验实现 |

普通决策只有一条主线：

```text
obs + mask -> Brain -> DQN -> 合法动作 Q 值 -> MortalEngine -> action
```

## 输入和动作契约

### 观测

`libriichi.state.PlayerState.encode_obs()` 把麻将状态编码为 `(channels, 34)`。34 是基本牌种轴，channels 是手牌、牌河、副露、分数和局况等特征通道。

| 版本 | 普通观测 | Oracle 额外观测 |
| --- | ---: | ---: |
| v1 | `(938, 34)` | `(211, 34)` |
| v2 | `(942, 34)` | `(217, 34)` |
| v3 | `(934, 34)` | `(217, 34)` |
| v4 | `(1012, 34)` | `(217, 34)` |

维度由 [`libriichi/src/consts.rs`](../libriichi/src/consts.rs) 定义。批量前向时再在最前面增加 `batch_size` 维。

Oracle 会把不可见信息沿通道维与 `obs` 拼接。模型类支持这一模式，但当前 `train.py`、`mortal.py` 和常用评测入口都没有启用它。

### 动作空间

`ACTION_SPACE = 46`，编号是 Rust 生产端与 Python 网络之间的协议：

| 编号 | 动作 |
| --- | --- |
| `0..36` | 打牌；在杠牌二次选择状态中表示选择牌种 |
| `37` | 立直 |
| `38..40` | 吃低、吃中、吃高 |
| `41` | 碰 |
| `42` | 杠的决定；必要时再用一次 0..36 选择牌种 |
| `43` | 和牌 |
| `44` | 九种九牌等主动流局 |
| `45` | 跳过 |

合法动作由 bool mask 给出。改变观测版本、动作编号或动作总数时，必须同步检查 Rust 的编码/解码和 Python 的模型/引擎。

## 主决策网络

### `Brain`：状态编码

`Brain` 的编码器是面向 34 种牌轴的一维残差网络：

```text
(B, in_channels, 34)
-> Conv1d(in_channels, conv_channels, kernel=3)
-> N x ResBlock
-> Conv1d(conv_channels, 32, kernel=3)
-> Flatten
-> Linear(32 * 34, 1024)
-> 版本相关输出
```

`conv_channels` 和 `num_blocks` 来自 `[resnet]`。示例配置是 192 个通道、40 个残差块，但结构本身不写死这两个值。

一维卷积沿牌种轴工作，适合建模同一花色的相邻数牌和跨通道的同牌种信息。最后的全连接层把 34 个位置聚合为全局状态表示。

### `ResBlock` 和通道注意力

每个残差块保留输入形状 `(B, conv_channels, 34)`：

```text
x -> 两层 Conv1d -> ChannelAttention -> + x -> activation（非 pre-act 版本）
```

`ChannelAttention` 分别沿牌轴取平均值和最大值，经共享两层 MLP 后相加并取 sigmoid，得到每个通道的缩放权重。它只重标定通道，不改变张量形状。

### `Brain` 版本差异

| 版本 | 激活/残差布局 | BatchNorm | 输出 |
| --- | --- | --- | --- |
| v1 | ReLU，post-activation | `eps` 默认值 | `mu` 和 `logsig`，各 512 维 |
| v2 | Mish，pre-activation | `eps` 默认值 | 1024 维 `phi` |
| v3 | Mish，pre-activation | `eps=1e-3` | 1024 维 `phi` |
| v4 | 同 v3 | `eps=1e-3` | 1024 维 `phi` |

v1 先执行 `Linear(1024, 512) + ReLU`，再生成高斯分布参数。推理时可使用 `mu`，也可在 `stochastic_latent=true` 时从 `Normal(mu, exp(logsig) + 1e-6)` 采样。v2–v4 使用确定性的 `phi`。

这是模型类支持的版本范围，不等于所有入口都支持所有版本。当前 `train.py` 和 1024 维 `AuxNet` 没有处理 v1 的二元输出，标准主训练实际应使用 v2–v4；`MortalEngine` 的推理路径则显式支持 v1。

`Brain.freeze_bn(True)` 让 BatchNorm 在模型处于训练模式时仍使用累计统计量；它没有关闭 BatchNorm 参数的梯度。

### `DQN`：合法动作价值

`DQN` 使用 dueling 形式，将状态价值 `V(s)` 和动作优势 `A(s,a)` 合成 Q 值：

```text
Q(s,a) = V(s) + A(s,a) - mean_legal(A(s,*))
```

平均值只统计当前合法动作，随后所有非法位置被设为 `-inf`。这样 `argmax` 和概率采样都不会选择非法动作。

| 版本 | 输入 | Value/Advantage 头 |
| --- | ---: | --- |
| v1 | 512 | 两个线性层：`512 -> 1/46` |
| v2 | 1024 | 两个 MLP：`1024 -> 512 -> 1/46` |
| v3 | 1024 | 两个 MLP：`1024 -> 256 -> 1/46` |
| v4 | 1024 | 单个 `Linear(1024, 47)` 后拆成 1 和 46 |

v4 仍按相同公式组合 `V` 与 `A`，只是共享一个输出层。

### `AuxNet`：辅助分类头

当前主干训练构造 `AuxNet((4,))`：

```text
phi: (B, 1024) -> Linear(1024, 4, bias=False) -> next-rank logits
```

标签是当前动作所在局结束后的玩家顺位。它用交叉熵为 `Brain` 增加结果相关监督，只参与训练；普通推理 checkpoint 虽包含其参数，选动作时不会调用它。

## 顺位预测网络

### 共同任务

Rust 的 `libriichi.dataset.Grp` 在每个 `start_kyoku` 提取一行 7 维特征：

```text
[grand_kyoku, honba, kyotaku, score_0/10000, ..., score_3/10000]
```

模型接收从对局开始到当前时刻的前缀序列，预测四家最终顺位。四人顺位共有 `4! = 24` 种合法排列，所以输出是 `(B, 24)` logits，而不是四个彼此独立的分类器。

`calc_matrix()` 将 24 类概率边缘化为 `(B, 4 players, 4 ranks)`；每个玩家的四个顺位概率和为 1。`get_label()` 则把 `[rank_by_player]` 映射回 0..23 的训练标签。

### `GRP`：GRU 实现

默认结构为：

```text
变长序列，每步 7 维
-> GRU(hidden_size, num_layers, batch_first=True)
-> 拼接每层最终隐藏状态
-> Linear(hidden_size * num_layers, 同维度)
-> ReLU
-> Linear(..., 24)
```

输入通过 `pad_sequence` 和 `pack_padded_sequence` 组成 `PackedSequence`，因此 GRU 不处理尾部补零。模块显式使用 `float64`。详细样本构造见 [train_grp_1.md](train_grp_1.md)。

### `GRP_Transformer`：实验实现

结构为：

```text
7 维序列
-> Linear(7, d_model)
-> 加可学习位置编码
-> 在序列头拼接 [CLS]
-> TransformerEncoder
-> 取 [CLS] 输出
-> MLP -> 24 logits
```

变长 batch 使用 `padding_mask`，`True` 表示注意力应忽略的位置；`[CLS]` 永远有效。模型同样使用 `float64`，最大序列长度受 `max_seq_len` 的位置 embedding 限制。

两种实现不能只替换类名就视为完全兼容：

- 配置节分别是 `[grp]` 和 `[grp_transformer]`；
- checkpoint 参数结构不同；
- 训练输入分别是 `PackedSequence` 和 padded tensor + mask；
- 当前 [`dataloader.py`](../mortal/dataloader.py) 和 [`mortal.py`](../mortal/mortal.py) 明确实例化 `GRP`，不会自动使用 Transformer。

## 模块关系

```text
普通决策
obs -> Brain -> DQN -> action

主干训练
obs -> Brain -> DQN -------> dqn_loss / cql_loss
              \-> AuxNet -> next_rank_loss

奖励构造（主干前向图之外）
对局前缀 -> 已训练 GRP -> 排名概率 -> kyoku_reward
```

因此 `GRP`：

- 不读取 `Brain` 的 `phi`；
- 不参与 `DQN` 选动作；
- 不与主干共享参数；
- 不被 `train.py` 的优化器更新；
- 但会改变主干的奖励 target，因而间接影响策略。

复盘模式也使用 GRP 输出额外的 `phi_matrix`，但该结果不会反向改变已经输出的动作。

## 主干训练目标

每个动作样本包含 `obs`、实际动作、合法 mask、`steps_to_done`、按局奖励和下一时刻顺位。总损失为：

```text
loss = dqn_loss
     + min_q_weight * cql_loss
     + next_rank_weight * next_rank_loss
```

| 损失 | 定义 | 作用 |
| --- | --- | --- |
| `dqn_loss` | `0.5 * MSE(Q(s,a), gamma^steps * reward)` | 拟合实际动作的折扣回报 |
| `cql_loss` | `logsumexp(Q_legal) - Q(s,a)` | 离线模式抑制未观测动作的过高估值 |
| `next_rank_loss` | 四分类交叉熵 | 让状态表示保留局末顺位信息 |

在线模式把 `cql_loss` 设为 0。奖励的完整推导见 [rewards.md](rewards.md)，训练循环见 [train.md](train.md)。

## 推理选动作

[`MortalEngine`](../mortal/engine.py) 默认对 Q 值执行 `argmax`。当 `boltzmann_epsilon > 0` 时，每个样本以 `1 - epsilon` 的概率贪心，否则对 `Q / temperature` 做 top-p 采样：

- `top_p >= 1`：从全部合法动作采样；
- `0 < top_p < 1`：保留按概率降序构成 nucleus 的高概率动作（包含跨过阈值的那个动作）；
- `top_p <= 0`：退化为 argmax。

这部分随机性属于引擎策略，不改变网络结构。完整推理路径仍是：

```text
libriichi 编码状态 -> Brain 理解局面 -> DQN 评价合法动作 -> Engine 决定贪心或采样
```
