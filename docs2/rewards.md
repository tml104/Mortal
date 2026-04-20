# Mortal 的奖励是如何计算的

这份文档总结的是当前仓库里 **主干网络训练时实际使用的奖励定义**，对应代码主要在：

- [mortal/reward_calculator.py](/home/tml104/Mortal/mortal/reward_calculator.py)
- [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py)
- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)
- [libriichi/src/dataset/gameplay.rs](/home/tml104/Mortal/libriichi/src/dataset/gameplay.rs)

如果先给一句最重要的结论，可以概括为：

> Mortal 训练时学的不是“这一手立即涨了多少点”，而是“这一手最终会带来多大的期望 rank-pt 收益”，这个收益先按局（kyoku）计算，再分配给该局中的动作。

---

## 1. 总结版结论

Mortal 的主干训练奖励可以拆成 4 步：

1. 先用 `GRP` 预测每一局结束后，当前玩家拿到第 1/2/3/4 名的概率。
2. 再用配置里的 `pts` 向量，把这个“排名概率”转换成“期望 pt”。
3. 然后对相邻两个局末时刻的期望 pt 做差，得到每一局带来的奖励增量。
4. 最后把这个“按局计算的奖励”分配给该局中的每一个动作，并结合 `gamma` 做折扣，形成训练 target。

所以它更像是：

- 以“最终排名价值”为核心的奖励

而不是：

- 直接用当前动作后的点数涨跌做奖励
- 直接用最终名次做单一终局奖励

---

## 2. 奖励不是直接用分数差

从代码看，主干训练真正使用的是：

- `kyoku_rewards`

它来自：

```python
kyoku_rewards = self.reward_calc.calc_delta_pt(player_id, grp_feature, rank_by_player)
```

这里调用的是 [RewardCalculator.calc_delta_pt](/home/tml104/Mortal/mortal/reward_calculator.py)。

这意味着主干训练默认不是用：

- `calc_delta_points()` 里的 raw score 差值

而是用：

- `calc_delta_pt()` 里的期望 pt 差值

也就是说，**训练奖励的核心单位不是“点数变化”，而是“排名价值变化”**。

---

## 3. 第一步：GRP 先预测排名概率

奖励计算的第一步发生在 [RewardCalculator.calc_grp](/home/tml104/Mortal/mortal/reward_calculator.py)：

```python
seq = list(map(
    lambda idx: torch.as_tensor(grp_feature[:idx+1], device=self.device),
    range(len(grp_feature)),
))

with torch.inference_mode():
    logits = self.grp(seq)
matrix = self.grp.calc_matrix(logits)
```

这里的含义是：

- `grp_feature` 描述一局比赛随着局数推进的过程特征
- 代码把前 `1` 局、前 `2` 局、前 `3` 局……分别组成一个前缀序列
- 把这些前缀序列送进已经训练好的 `GRP`
- `GRP` 输出每个时间点的排名概率矩阵

这个 `matrix` 的含义可以理解为：

- 在当前时刻
- 当前玩家拿第 1/2/3/4 名的概率分别是多少

注意，这里不是直接看真实最终名次，而是先看 **在该时刻看来，最终排名的概率分布**。

---

## 4. 第二步：把排名概率转换成期望 pt

接下来 [RewardCalculator.calc_rank_prob](/home/tml104/Mortal/mortal/reward_calculator.py) 会取出当前玩家自己的排名概率：

```python
rank_prob = torch.cat((matrix[:, player_id], final_ranking))
```

这里做了两件事：

1. `matrix[:, player_id]`
   取出当前玩家在每个时刻的排名概率向量。
2. `final_ranking`
   再额外拼上一个“最终真实结果”的 one-hot 向量。

也就是说，得到的 `rank_prob` 是一个时间序列：

- 前面几项：`GRP` 预测的排名概率
- 最后一项：真实最终排名

然后 [RewardCalculator.calc_delta_pt](/home/tml104/Mortal/mortal/reward_calculator.py) 会执行：

```python
exp_pts = rank_prob @ self.pts
```

其中 `self.pts` 来自配置：

```toml
[env]
pts = [6.0, 4.0, 2.0, 0.0]
```

这一步的意思是：

- 如果第 1 名值 `6`
- 第 2 名值 `4`
- 第 3 名值 `2`
- 第 4 名值 `0`

那么对某个时刻的排名概率分布：

```text
[P(1st), P(2nd), P(3rd), P(4th)]
```

它对应的期望 pt 就是：

```text
P(1st) * 6 + P(2nd) * 4 + P(3rd) * 2 + P(4th) * 0
```

所以这里算出来的是：

- 当前时刻，从最终排名价值角度看，这名玩家“期望能拿到多少 pt”

---

## 5. 第三步：对相邻时刻做差，得到每一局的奖励

在 [RewardCalculator.calc_delta_pt](/home/tml104/Mortal/mortal/reward_calculator.py) 中，真正形成奖励的是：

```python
reward = exp_pts[1:] - exp_pts[:-1]
```

也就是：

- 下一时刻的期望 pt
- 减去当前时刻的期望 pt

这个差值就表示：

> 打完这一局之后，当前玩家的“最终排名价值预期”提高了多少，或者下降了多少。

因此这里得到的 `reward` 本质上是：

- **每一局（kyoku）结束后带来的排名价值增量**

而不是：

- 每一步动作的即时奖励

举个直观例子：

- 某时刻模型认为你最终大概值 `3.8 pt`
- 打完这一局后变成大概值 `4.5 pt`
- 那么这一局对应的奖励就是 `+0.7`

反过来如果局面对最终排名预期变差了，奖励就是负数。

---

## 6. 第四步：把“按局奖励”分配给该局中的动作

在 [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py) 里，训练样本是“按动作”构造的，但奖励本身是“按局”算出来的。

核心代码是：

```python
entry = [
    obs[i],
    actions[i],
    masks[i],
    steps_to_done[i],
    kyoku_rewards[at_kyoku[i]],
    player_ranks[at_kyoku[i] + 1],
]
```

这里最关键的是：

- `at_kyoku[i]` 表示第 `i` 个动作属于哪一局
- `kyoku_rewards[at_kyoku[i]]` 就把“这一局的奖励”分给这局里的每个动作

因此：

- 同一局里的多个动作，会共享同一个 `kyoku_reward`

这说明主干训练不是在学习“每步即时 reward”，而是在学习：

> “这一步动作，最终对本局结束后的排名价值增量贡献有多大。”

---

## 7. 第五步：再用 `steps_to_done` 和 `gamma` 形成训练目标

在 [mortal/train.py](/home/tml104/Mortal/mortal/train.py) 中，Q 学习目标是：

```python
q_target_mc = gamma ** steps_to_done * kyoku_rewards
```

所以最终训练 target 由三部分决定：

1. `kyoku_rewards`
   这一步所在局的排名价值增量
2. `steps_to_done`
   从当前动作到这一局结束，还要经过多少个“计入折扣的动作”
3. `gamma`
   折扣因子

这说明 Mortal 使用的是一种 **Monte Carlo 风格的回报目标**：

- 不是标准的一步 TD bootstrap
- 而是把一局结束时的奖励，折回到这局内的各个动作上

---

## 8. `steps_to_done` 是怎么计算的

`steps_to_done` 也是在 [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py) 里算的：

```python
steps_to_done = np.zeros(game_size, dtype=np.int64)
for i in reversed(range(game_size)):
    if not dones[i]:
        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
```

它的含义是：

- 从当前位置开始
- 到本局 reward 生效前
- 还需要经过多少个会被折扣计数的动作

这里真正决定“某一步算不算一个折扣步长”的是 `apply_gamma[i]`。

---

## 9. 哪些动作会被计入折扣

在 [libriichi/src/dataset/gameplay.rs](/home/tml104/Mortal/libriichi/src/dataset/gameplay.rs) 中有一行很重要的代码：

```rust
// only discard and kan will discount
self.apply_gamma.push(label <= 37);
```

作者的注释已经写得很直接：

- 只有 `discard` 和 `kan` 会计入折扣

这意味着：

- 不是所有动作类型都会让 `steps_to_done` 增加 1
- 某些响应动作不会额外拉长折扣距离

从设计意图上，可以理解成：

- 真正推进局面主时间轴的，主要是出牌和杠相关动作
- 其他动作不一定被视为一个完整的“时间推进步”

因此 `gamma` 的作用并不是对日志里的每一条动作记录都统一衰减，而是只对作者认为应当计入时间推进的动作衰减。

---

## 10. `pts` 决定了奖励偏好

配置里的：

```toml
[env]
pts = [6.0, 4.0, 2.0, 0.0]
```

对奖励定义非常关键。

因为 `GRP` 输出的是排名概率，真正把它变成标量奖励的是这个 `pts` 向量。

例如：

- 如果你把 `pts` 改成 `[3, 1, -1, -3]`
- 那么同样的排名概率变化，会变成另一种奖励形状

所以：

- `GRP` 决定“当前时刻最终排名的概率怎么看”
- `pts` 决定“不同排名在奖励上值多少钱”

这两个一起决定了主干网络最终追求的目标。

---

## 11. 训练时真正学到的是什么

把前面的几步串起来，主干网络学到的其实是：

> 对于当前状态下的某个动作，这个动作与“本局结束时我最终排名价值的提升或下降”之间的关系。

更具体地说，它不是直接学：

- 这一手立刻加了多少点

而是在学：

- 这一手最终会让自己在“第几名值多少 pt”这个目标上变得更好还是更坏

这也是为什么 `GRP` 会成为主干训练的前置依赖：

- 没有 `GRP`，就没有这个“排名概率 -> 期望 pt -> 局奖励”的计算链

---

## 12. `calc_delta_points()` 是什么

[RewardCalculator.calc_delta_points](/home/tml104/Mortal/mortal/reward_calculator.py) 里还有另一个函数：

```python
def calc_delta_points(self, player_id, grp_feature, final_scores):
    seq = np.concatenate((grp_feature[:, 3 + player_id] * 1e4, [final_scores[player_id]]))
    delta_points = seq[1:] - seq[:-1]
    return delta_points
```

它计算的是：

- 当前玩家 raw score 的相邻时刻差值

也就是更接近“点数变化”的东西。

但从当前主干训练代码看：

- `dataloader.py` 实际调用的是 `calc_delta_pt()`
- 没有调用 `calc_delta_points()`

所以在当前实现里：

- `calc_delta_points()` 只是备用工具
- 主干训练默认并不用它作为奖励

---

## 13. 整个奖励链路的文本流程图

可以把整个奖励计算过程总结成下面这张图：

```text
原始对局日志
-> GameplayLoader
-> 提取每步的 obs / action / mask / at_kyoku / done / apply_gamma
-> 同时提取每局的 grp_feature / rank_by_player
-> 已训练好的 GRP
-> 预测每个时刻的排名概率
-> 用 pts 向量换算成期望 pt
-> 对相邻时刻做差
-> 得到每一局的 kyoku_rewards
-> 用 at_kyoku 分配给该局中的每一个动作
-> 用 steps_to_done 和 gamma 做折扣
-> q_target_mc
-> 用来训练主干网络
```

---

## 14. 一句话总结

如果只记一句话，可以记成：

> Mortal 的奖励本质上是“每一局结束后，自己最终排名价值的期望增量”，它由 `GRP` 的排名预测和 `pts` 的排名价值映射共同决定，再折回该局中的动作用于训练。
