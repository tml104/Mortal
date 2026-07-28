# 主策略奖励计算

主策略训练不直接使用每手点数差，也不只在整场结束时给一个最终名次奖励。当前实现把“最终顺位的期望价值变化”作为每局奖励，再将它分配给该局内的动作。

核心代码：

- [`mortal/reward_calculator.py`](../mortal/reward_calculator.py)
- [`mortal/dataloader.py`](../mortal/dataloader.py)
- [`mortal/train.py`](../mortal/train.py)
- [`libriichi/src/dataset/gameplay.rs`](../libriichi/src/dataset/gameplay.rs)

## 一张图看完整链路

```text
mjai 日志
-> 每个 start_kyoku 的 7 维 GRP 特征
-> 已训练的 GRU 版 GRP
-> 各时刻的最终顺位概率
-> 用 env.pts 转为期望 pt
-> 相邻时刻作差，得到 kyoku_reward
-> 按 at_kyoku 分给局内动作
-> 用 gamma ** steps_to_done 折扣
-> DQN 的 q_target
```

三个关键结论：

1. 奖励单位由 `env.pts` 定义，不是麻将原始点数。
2. 奖励先按局计算，同一局的动作共享同一个未折扣奖励。
3. GRP 只参与数据预处理，不在主干反向传播图中。

## 第一步：构造时序特征

`libriichi.dataset.Grp` 在每个 `start_kyoku` 记录：

```text
[grand_kyoku, honba, kyotaku, score_0/10000, ..., score_3/10000]
```

若一场对局有 `K` 个局开始时刻，就得到 `(K, 7)` 的 `grp_feature`。`RewardCalculator.calc_grp()` 构造所有前缀：

```text
feature[:1], feature[:2], ..., feature[:K]
```

已训练的 GRU 版 `GRP` 对每个前缀输出最终顺位分布。`calc_matrix()` 将 24 种完整顺位排列转换为：

```text
matrix.shape = (K, 4 players, 4 ranks)
```

其中 `matrix[t, p]` 是时刻 `t` 对玩家 `p` 最终获得第 1–4 名的概率。

## 第二步：追加真实终局

对目标玩家 `p`，`calc_rank_prob()` 取出：

```text
matrix[:, p]  # K 个 GRP 预测
```

并在末尾追加真实最终名次的 one-hot 向量。因此最终序列长度是 `K + 1`：

```text
p_0, p_1, ..., p_(K-1), p_K
```

- `p_0..p_(K-1)` 来自 GRP；
- `p_K` 是真实终局顺位。

默认 `uniform_init=False`，所以第一个时刻仍采用 GRP 预测。`RewardCalculator` 支持将第一个分布改为均匀分布，但标准 [`dataloader.py`](../mortal/dataloader.py) 没有开启它。

## 第三步：顺位概率转为期望 pt

配置给出四个顺位的价值：

```toml
[env]
pts = [6.0, 4.0, 2.0, 0.0]
```

记该向量为 `v`，时刻 `t` 的顺位概率为 `p_t`，则期望价值是：

```text
E_t = p_t · v
```

例如：

```text
p_t = [0.20, 0.40, 0.30, 0.10]
v   = [6,    4,    2,    0]

E_t = 0.20*6 + 0.40*4 + 0.30*2 + 0.10*0 = 3.4
```

`pts` 决定优化偏好。修改它不会改变 GRP 的顺位概率，却会改变相同概率变化对应的奖励大小。

不要混淆两种 pt：

- `env.pts` 用于训练奖励；
- 评测中的 `[90, 45, 0, -135]` 只用于显示和 best checkpoint 判断，不进入训练 target。

## 第四步：相邻期望作差

每局奖励定义为：

```text
r_k = E_(k+1) - E_k,  k = 0..K-1
```

代码对应：

```python
exp_pts = rank_prob @ self.pts
reward = exp_pts[1:] - exp_pts[:-1]
```

例如某局前后的期望 pt 分别是 `3.4` 和 `4.1`，则该局奖励为 `+0.7`；若从 `4.1` 降到 `3.6`，奖励为 `-0.5`。

最后一项使用真实终局 one-hot，因此最后一局的奖励把最后一次 GRP 估计衔接到真实比赛结果。

相邻差分还有一个性质：整场未折扣奖励会望远镜式相消，和为：

```text
sum(r_k) = E_K - E_0
```

也就是最终真实顺位价值减去开局时的 GRP 期望价值。

## 第五步：分配给逐动作样本

`GameplayLoader` 产生逐动作数据，并用 `at_kyoku[i]` 标记动作属于哪一局。组装第 `i` 条样本时：

```python
kyoku_rewards[at_kyoku[i]]
```

所以同一局内所有动作获得同一个 `kyoku_reward`。这是一种信用分配近似：数据并不能证明每个动作对结果的独立因果贡献，网络通过大量状态–动作样本学习哪些选择通常与更好的局结果相关。

辅助标签 `player_ranks[at_kyoku[i] + 1]` 与奖励来源不同：它根据相邻局开始分数（最后再接真实终局分数）稳定排序得到，用于 `AuxNet` 的下一时刻顺位分类。

## 第六步：折扣到当前动作

训练 target 是：

```python
q_target_mc = gamma ** steps_to_done * kyoku_rewards
```

其中：

- `gamma` 来自 `[env]`；
- `steps_to_done` 是从当前样本到该局最后一个动作之间的折扣步数；
- 局内最后一个动作的 `steps_to_done` 为 0。

`dones` 根据相邻样本的 `at_kyoku` 是否增加生成，最后一条样本也强制为 done。数据加载器从后向前累计：

```python
if not dones[i]:
    steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
```

### 哪些动作增加折扣步数

Rust 当前实际条件是：

```rust
self.apply_gamma.push(label <= 37);
```

结合动作编号，这会对以下样本记一个折扣步：

- `0..36`：打牌或杠牌的牌种选择；
- `37`：立直。

`38..45` 的吃、碰、杠决定、和牌、主动流局和跳过不会增加步数。源码旁的英文注释写着“only discard and kan will discount”，但判断条件也包含立直；运行行为应以 `label <= 37` 为准。

当 `gamma = 1` 时，所有动作的 target 都不因距离衰减；`steps_to_done` 仍会计算，但数值不影响 target。

## 训练损失中的位置

主干只取日志实际动作的 Q 值：

```text
q = DQN(Brain(obs), mask)[actual_action]
dqn_loss = 0.5 * MSE(q, q_target_mc)
```

GRP 在 DataLoader worker 的 CPU 上以 inference mode 运行，输出转成 NumPy 奖励后才进入训练 batch。因此：

- 主干梯度不会流入 GRP；
- 主干优化器不包含 GRP 参数；
- 更换 GRP 权重会让同一批日志得到不同 target；
- 主干 checkpoint 不保存所用 GRP 的参数副本，只在保存的配置快照中保留其路径。

为了保持实验可复现，训练过程中不应悄悄覆盖 `grp.state_file`。

## `calc_delta_points()` 的地位

`RewardCalculator.calc_delta_points()` 计算当前玩家相邻时刻的原始点数差：

```text
下一时刻 score - 当前时刻 score
```

标准主策略 [`dataloader.py`](../mortal/dataloader.py) 调用的是 `calc_delta_pt()`，没有调用 `calc_delta_points()`。后者目前是备用方法，不是默认奖励。

## 最简记忆

```text
GRP 估计“现在最终各名次的概率”
-> pts 定义“各名次值多少”
-> 相邻期望之差定义“这一局值多少”
-> at_kyoku 把局奖励交给动作
-> gamma 和 steps_to_done 决定动作距离折扣
```
