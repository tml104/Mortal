# `train.py` 解读

本文汇总对 [`mortal/train.py`](../mortal/train.py) 中主训练流程的解读，包括 `train_batch`、CQL 权重、AMP 与梯度更新链路、数据增强、AdamW 参数分组、玩家白名单和牌谱文件索引。标题中的行号以当前源码为准；若逻辑分散在多处，则同时列出配置读取与实际使用位置。

主训练使用的 `FileDatasetsIter`、局奖励映射、样本缓冲以及多 worker 文件切分，集中见
[`dataloader.py` 解读](dataloader解读.md)。

## 第 337～445、573～609 行：`train_batch` 的完整训练流程

### 函数职责与输入契约

`train_batch` 是 `train_epoch()` 内部的一个嵌套函数，每次处理一个长度恰好为 `batch_size` 的完整 batch。它会执行前向传播、构造损失、反向传播、参数更新、学习率推进和周期性保存。函数没有返回值，它会更新闭包中的 `steps`、`idx`、进度条、模型梯度和训练输出。

| 参数 | batch 形状 | 用途 |
| --- | --- | --- |
| `obs` | `(B, *obs_shape(version))` | 当前玩家的可见状态观测 |
| `actions` | `(B,)` | 牌谱中实际执行的动作编号 |
| `masks` | `(B, ACTION_SPACE)` | 每个动作在当前状态下是否合法 |
| `steps_to_done` | `(B,)` | 当前决策到本局结束的有效折扣步数 |
| `kyoku_rewards` | `(B,)` | 冻结 GRP 模型计算的本局顺位效用变化 |
| `player_ranks` | `(B,)` | 本局结束后的玩家顺位，取值为 `0..3` |

这里 `B == batch_size`，`ACTION_SPACE == 46`。观测的 `(C, 34)` 形状在 v1～v4 中依次为 `(938, 34)`、`(942, 34)`、`(934, 34)` 和 `(1012, 34)`。`DataLoader` 产生的短 batch 会先存入 `remaining_*` 列表，数据迭代完毕后再拼成完整 batch；尾部仍不足 `batch_size` 的样本会被舍弃。函数内部还会检查每个日志动作都位于对应的合法动作集：

```python
assert masks[range(batch_size), actions].all()
```

### Q 目标与前向传播

数据加载器先用 GRP 预测每局前后的最终顺位概率 $p_k$，再与配置中的 `pts` 做点积得到 $V_k=p_k\cdot\mathrm{pts}$，局奖励为 $r_k=V_{k+1}-V_k$。`train_batch` 按有效步数将这个局奖励折扣回当前动作：

$$
y_i=\gamma^{n_i}r_i
$$

其中 $n_i$ 对应 `steps_to_done[i]`，$r_i$ 对应 `kyoku_rewards[i]`。`gamma=1` 时，同一局内的动作共享相同目标。该目标直接回归经 GRP 塑形的局奖励；计算过程不读取下一状态 Q 值，也没有 target network。

前向传播依次执行：

```python
phi = mortal(obs)
q_out = dqn(phi, masks)
q = q_out[range(batch_size), actions]
next_rank_logits, = aux_net(phi)
```

`Brain` 将观测编码为共享表征 `phi`。当前路径要求 `phi` 是单个 1024 维张量，因此常规主训练使用 v2～v4；v1 的 `Brain` 会返回 `(mu, logsig)` 元组，无法直接传给这里的 `DQN`。

`DQN` 使用 dueling 结构为 46 个动作计算 Q 值：$Q(s,a)=V(s)+A(s,a)-\operatorname{mean}_{a'\in\mathcal A_{\mathrm{legal}}}A(s,a')$。非法动作随后被设为 `-inf`；`q` 只取每个样本中日志动作的 Q 值。`AuxNet` 基于同一个 `phi` 输出四个顺位 logits。

### 损失函数与梯度流向

总损失为：

$$
L=L_{\mathrm{DQN}}+\alpha L_{\mathrm{CQL}}+\beta L_{\mathrm{rank}}
$$

| 损失 | 定义与作用 | 接收梯度的模型 |
| --- | --- | --- |
| `dqn_loss` | $\frac{1}{2}\operatorname{MSE}(Q(s,a),y)$，回归日志动作的 Q 目标 | `Brain` + `DQN` |
| `cql_loss` | 约束合法动作的 Q 分布，仅在离线模式启用 | `Brain` + `DQN` |
| `next_rank_loss` | 本局结束顺位的四分类交叉熵 | `Brain` + `AuxNet` |

系数 $\alpha$ 和 $\beta$ 分别由 `cql.min_q_weight` 和 `aux.next_rank_weight` 控制。CQL 的详细公式见下一节。

### 梯度计算与参数更新位置

损失计算完成后，以下代码负责反向传播和参数更新：

```python
scaler.scale(loss / opt_step_every).backward()

if idx % opt_step_every == 0:
    if max_grad_norm > 0:
        scaler.unscale_(optimizer)
        clip_grad_norm_(params, max_grad_norm)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

scheduler.step()
```

`backward()` 计算并累积梯度，`scaler.step(optimizer)` 在梯度有限时最终调用 `optimizer.step()` 修改模型参数。当 `opt_step_every=K` 时，每 K 个 batch 执行一次参数更新，近似有效 batch size 为 `batch_size × K`。`save_every % opt_step_every == 0` 保证 checkpoint 位于完整的梯度累积边界。各调用的内部行为见后文 AMP 小节。

### 计数、统计与周期任务

`steps` 记录已处理的完整 batch 数，`idx` 记录当前 `save_every` 窗口中的 batch 下标。每个 batch 都会在 `torch.inference_mode()` 中累加 DQN 和顺位 loss，离线模式还会累加 CQL loss；同时保存每个样本的预测 Q 值与目标 Q 值。这些统计操作不会延长训练计算图的生命周期。

后续副作用由 `steps` 触发：

| 条件 | 行为 |
| --- | --- |
| 在线模式且 `steps % submit_every == 0` | 发布最新的 `Brain` 和 `DQN` 参数 |
| `steps % save_every == 0` | 写入 TensorBoard、重置统计窗口并保存普通 checkpoint |
| `steps % test_every == 0` | 与固定 baseline 评测，记录对局指标，并在 `avg_pt` 和 `avg_rank` 同时满足条件时更新最佳 checkpoint |

TensorBoard 中的 loss 是最近 `save_every` 个 batch 的平均值。Q 值直方图会将窗口张量展平后每隔 128 个元素取样，以限制 TensorBoard event 文件体积。

在线模式完成定期评测后，子进程会执行 `sys.exit(0)`，再由 `main()` 中的父进程重新拉起训练。这是项目对“在线评测返回后进程可能卡住”问题的现有规避措施。

## 第 88～90、402～416 行：`cql.min_q_weight` 与 CQL 损失

### CQL 是什么

CQL 是 **Conservative Q-Learning（保守 Q 学习）**，主要用于离线强化学习。

离线训练只能使用已有的固定日志，不能让当前策略在环境中自由尝试动作。对于日志中很少出现或从未出现的动作，模型没有足够的监督信号，却可能错误地估计出很高的 Q 值。如果直接选择 Q 值最高的动作，策略就可能偏向这些缺少数据支持的动作。

CQL 通过向训练目标加入保守惩罚，降低这种分布外动作被高估的风险。

### Mortal 中的 CQL 损失

[`train.py`](../mortal/train.py) 中的实现是：

```python
cql_loss = q_out.logsumexp(-1).mean() - q.mean()
```

其中：

- `q_out` 是每个样本所有合法动作的 Q 值；
- `q` 是牌谱日志中实际执行动作的 Q 值；
- 非法动作已经在 [`DQN.forward()`](../mortal/model.py) 中被设为 `-inf`，不会参与 `logsumexp` 的有效计算。

对单个样本来说，这一项可以写为：

```text
logsumexp(Q(s, 所有合法动作)) - Q(s, 日志动作)
```

它也等价于把 Q 值当作 logits，对日志动作计算交叉熵：

```text
-log softmax(Q(s, ·))[日志动作]
```

最小化该损失会提高日志动作相对于其他合法动作的优势。同时，DQN 回归损失会把日志动作的 Q 值锚定到 Monte Carlo 目标，因此 CQL 的整体效果主要是压低其他缺少数据支持的动作。

### `min_q_weight` 的作用

主训练的总损失为：

```text
总损失 = DQN 损失
         + min_q_weight × CQL 损失
         + next_rank_weight × 顺位预测损失
```

`cql.min_q_weight` 就是 CQL 损失在总损失中的乘数：

- 设为 `0` 相当于关闭 CQL；
- 设得较大会让策略更保守，更偏向牌谱中实际出现的行为；
- 设得过大可能使模型过度模仿日志策略，压制日志中少见但实际较好的动作；
- 合适的数值取决于牌谱质量、动作覆盖度和实际评测结果。

当 `control.online=false` 时，训练使用固定日志并计算 CQL。在线训练中，数据由当前策略持续生成，实现会把 `cql_loss` 保持为 `0`，因此 `min_q_weight` 不参与在线损失。

`min_q_weight` 是仅参与训练损失加权的超参数。它通过梯度间接影响最终权重，推理阶段不会直接读取该值或修改 Q 值。

## 第 104～106、195～196、398～444 行：AMP 与梯度更新链路

### AMP 是什么

AMP 是 **Automatic Mixed Precision（自动混合精度）**。它让适合低精度的算子使用 FP16 或 BF16，同时让数值敏感算子依照 PyTorch 规则保留较高精度。模型参数通常继续以 FP32 保存并由优化器更新。PyTorch 会按算子自动选择运算精度：

- 卷积和全连接等适合低精度的运算可使用低精度；
- 对数、归约和损失等数值敏感运算会按 PyTorch 的 autocast 规则选择精度；
- 动作编号、布尔 mask 等整数数据不受影响。

当前代码没有显式指定 autocast 的 `dtype`。CUDA 上通常使用 FP16，CPU 上通常使用 BF16，实际行为还取决于 PyTorch 版本和硬件后端。

### `autocast` 与 `GradScaler` 的分工

`autocast` 控制前向传播和损失计算的算子精度：

```python
with torch.autocast(device.type, enabled=enable_amp):
    phi = mortal(obs)
    q_out = dqn(phi, masks)
    # 计算 DQN、CQL 和顺位辅助损失
```

开启时，`Brain`、`DQN` 和 `AuxNet` 中适合低精度的运算会由 PyTorch 自动降精度执行。局奖励目标先在 autocast 外使用 `float64` 计算，然后转成 `float32`，因此该计算不受 autocast 影响。

`GradScaler` 负责梯度的动态缩放与非有限值检查：

```python
scaler = GradScaler(device.type, enabled=enable_amp)
```

FP16 的数值范围较小，反向传播中的微小梯度可能下溢为 `0`。`GradScaler` 把 loss 与反向传播链上的梯度同比放大，提高这些中间值在 FP16 中保留的概率。

完整行为由多个调用共同完成：

| 调用 | 内部职责 |
| --- | --- |
| `scaler.scale(value)` | 返回 `value × scale`，并在首次使用时初始化缩放系数 |
| `.backward()` | 由 PyTorch autograd 沿计算图计算缩放后的梯度 |
| `scaler.unscale_(optimizer)` | 将 optimizer 持有的梯度除以 `scale`，同时检查 `inf` 和 `NaN` |
| `scaler.step(optimizer)` | 自动反缩放（尚未显式执行时），并按检查结果决定是否调用 `optimizer.step()` |
| `scaler.update()` | 根据本轮的非有限值检查结果调整下一轮 `scale` |

`.backward()` 属于 Tensor/autograd 接口。`GradScaler` 提供缩放系数，并管理反缩放、检查结果和动态调节状态。

### `scale(loss / opt_step_every)` 与梯度累积

设当前缩放系数为 $S$，`opt_step_every=K`，则传入反向传播的损失为：

$$
L_{\mathrm{scaled}}=S\frac{L}{K}
$$

对参数 $\theta$ 产生的暂存梯度为：

$$
g_{\mathrm{scaled}}=S\frac{1}{K}\frac{\partial L}{\partial\theta}
$$

乘以 $S$ 可减少低精度反向传播中的梯度下溢。除以 $K$ 使 K 个 batch 累加后的梯度与这 K 个 batch 的平均梯度尺度一致。

例如 `K=4` 时，四次 `backward()` 在参数的 `.grad` 中累加：

$$
S\frac{g_1}{4}+S\frac{g_2}{4}+S\frac{g_3}{4}+S\frac{g_4}{4}
=S\frac{g_1+g_2+g_3+g_4}{4}
$$

累积结束后统一除以 $S$，即可得到四个 batch 的平均梯度。`scaler.update()` 只在累积边界执行，所以同一个累积窗口使用相同的 $S$。

### 从 `scaler.step()` 到 `scheduler.step()`

1. **`scaler.unscale_(optimizer)`**：开启梯度裁剪时，项目先显式反缩放，让 `clip_grad_norm_` 使用真实梯度范数。关闭裁剪时，`scaler.step()` 内部会自动完成反缩放和检查。
2. **`scaler.step(optimizer)`**：`optimizer` 持有 `Brain`、`DQN` 和 `AuxNet` 的参数引用、参数分组、学习率、AdamW 动量与二阶矩状态。GradScaler 通过它定位待反缩放的梯度，并在梯度全部有限时调用 `optimizer.step()`。梯度出现 `inf` 或 `NaN` 时，当次 AdamW 参数更新会被跳过。
3. **`scaler.update()`**：汇总 `step()` 记录的检查结果。当前环境的 PyTorch 2.11.0 默认从 65536 开始；遇到非有限梯度时乘以 `backoff_factor=0.5`；连续 `growth_interval=2000` 次成功更新后乘以 `growth_factor=2.0`。本项目沿用这些默认值。稳定计数以 `update()` 次数为单位，在梯度累积场景中对应优化器更新尝试次数。
4. **`optimizer.zero_grad(set_to_none=True)`**：将参数的 `.grad` 设为 `None`，为下一个累积窗口做准备。它保留已更新的模型参数以及 AdamW 的动量和二阶矩状态。当次更新被跳过时，这一步仍会清除异常梯度。首个 batch 前还会在第 232 行执行一次初始清理。
5. **`scheduler.step()`**：自定义 `LinearWarmUpCosineAnnealingLR` 会更新 optimizer 参数组中的学习率，供后续 AdamW 更新使用。该调用位于梯度累积条件之外，因此每个 batch 都会推进；当前 batch 只累积梯度或 AMP 跳过参数更新时，学习率曲线也会前进一步。

`scaler.step()` 在梯度有限时将更新工作交给 AdamW。具体的一阶矩、二阶矩、参数更新和权重衰减公式见后文“AdamW 参数分组”一节。

### 动态缩放的完整时序

```text
autocast 前向传播
  -> scale(loss / K)
  -> backward() 累积缩放梯度
  -> 到达 K 个 batch 的累积边界
  -> unscale + 检查 inf/NaN
       -> 梯度有限：optimizer.step()
       -> 发现非有限值：跳过 optimizer.step()
  -> update() 调整 scale
  -> zero_grad() 清理梯度
  -> scheduler.step() 推进学习率
```

### 开启和关闭时的差异

开启 AMP 后，在支持低精度加速的现代 NVIDIA GPU 上通常可以：

- 降低激活张量的显存占用；
- 提高卷积和矩阵乘法的吞吐量；
- 容纳更大的 `batch_size`；
- 缩短受 GPU 计算限制的训练时间。

如果训练主要受牌谱解析、CPU 数据加载或 I/O 限制，AMP 的提速可能不明显。AMP 也会改变部分运算的舍入误差，所以与纯 FP32 训练不会逐位一致。`GradScaler` 可以缓解反向梯度下溢；对于前向计算已经产生的 `NaN` 或 `inf`，应另行定位根因。

关闭 AMP 时：

- `torch.autocast(..., enabled=False)` 不会自动降精度；
- 模型前向和损失通常使用 FP32；
- `GradScaler(enabled=False)` 会原样返回 loss，`step()` 直接调用 `optimizer.step()`，`update()` 不调整缩放状态；
- 训练代码仍可以保留 `scale()`、`unscale_()` 和 `step()` 调用，无需另外维护一套 FP32 训练分支。

`control.enable_amp` 主要控制主训练循环。评测和推理使用各自传给 [`MortalEngine`](../mortal/engine.py) 的 `enable_amp`，不一定跟随这个配置。

普通 checkpoint 会保存 `GradScaler` 状态，并在恢复训练时加载它。如果旧 checkpoint 在 AMP 关闭时保存，其 scaler 状态通常为空；之后改为开启 AMP 并直接恢复时，部分 PyTorch 版本可能拒绝加载空的 scaler 状态。

## 第 130～135、346～348 行：数据增强的开关与首轮顺序

### 项目中的数据增强是什么

这里的数据增强会将整份 mjai 对局日志中的万子和筒子系统性互换，它不向观测张量添加噪声：

```text
1m ↔ 1p
2m ↔ 2p
...
9m ↔ 9p

5mr ↔ 5pr
```

索子、赤五索子、字牌和未知牌保持不变。例如：

```text
原始：1m  5pr  7s  E
增强：1p  5mr  7s  E
```

转换会统一应用到起手牌、摸切、吃碰杠、宝牌指示牌和里宝牌指示牌等所有相关事件。转换完成后，Rust `GameplayLoader` 会从头推进牌局状态，重新生成：

- 模型观测 `obs`；
- 日志动作标签；
- 合法动作 `mask`；
- 吃碰杠的相关选择。

例如，原本“切 `1m`”的动作会同步变为“切 `1p`”。分数、玩家顺位和奖励结果保持不变。

万子和筒子的麻将规则是对称的，所以万筒互换后仍是一份规则等价的有效牌谱。这种增强可以降低模型对某个花色位置的偶然偏好，促使它学习万筒之间应有的规则对称性。

### 两个参数的真实分工

[`FileDatasetsIter.build_iter()`](../mortal/dataloader.py) 的核心逻辑是：

```python
for _ in range(self.num_epochs):
    yield from self.load_files(self.augmented_first)
    if self.enable_augmentation:
        yield from self.load_files(not self.augmented_first)
```

`augmented_first` 决定每个内部 epoch 第一遍文件是否增强：

- `false`：第一遍使用原始牌谱；
- `true`：第一遍使用万筒互换后的牌谱。

`enable_augmentation` 决定第一遍结束后，是否追加一遍相反形态的数据。

四种组合的实际行为如下：

| `enable_augmentation` | `augmented_first` | 每个内部 epoch 的读取方式 |
| --- | --- | --- |
| `false` | `false` | 只读取原始牌谱 |
| `false` | `true` | 只读取增强牌谱 |
| `true` | `false` | 先原始，后增强 |
| `true` | `true` | 先增强，后原始 |

因此，`enable_augmentation` 不能单独理解为“是否增强”。更准确的理解是：

- `augmented_first` 决定第一遍的数据形态；
- `enable_augmentation` 决定是否追加一遍相反形态的数据。

即使 `enable_augmentation=false`，只要 `augmented_first=true`，唯一的一遍数据仍然会使用增强形式。

### 对数据量、顺序和训练计划的影响

开启 `enable_augmentation` 后，同一批文件在每个内部 epoch 会遍历两遍，所以通常会：

- 使训练样本数和 batch 数接近翻倍；
- 使牌谱解析量和训练时间接近翻倍；
- 更快到达以 batch 计数的 `save_every`、`test_every` 和学习率调度节点。

若有 `N` 个文件且 `dataset.num_epochs=3`，则不开启增强时总文件遍历量约为 `3N`，开启后约为 `6N`。实际完整 batch 数还会受玩家过滤、worker 切分和尾部不足一个 batch 的样本影响。

每次 `load_files()` 都会重新打乱文件列表，所以原始样本和对应的增强样本通常不会紧邻出现。当 `num_workers>1` 时，每个 worker 会独立执行先后顺序，全局输出不一定是严格分隔的两大段。

增强数据是原对局的规则等价变换，独立牌局数量保持不变。当前增强只交换万子和筒子，生成范围不包含三种数牌花色的所有排列。

一般同时训练原始和增强数据时可使用：

```toml
[dataset]
enable_augmentation = true
augmented_first = false
```

只训练原始数据时使用：

```toml
[dataset]
enable_augmentation = false
augmented_first = false
```

`enable_augmentation=false` 且 `augmented_first=true` 表示只训练万筒互换版本，通常更适合检查增强实现或进行对照实验。

这两个参数不改变模型结构或 checkpoint 参数形状，但会改变数据量、训练步数和样本分布。`train.py` 无论离线还是在线模式都会把它们传给数据加载器，不会根据 `control.online` 自动禁用。

## 第 173～188 行：AdamW 参数分组

[`train.py` 第 173～188 行](../mortal/train.py) 的核心作用是：

> 把 `Brain`、`DQN` 和 `AuxNet` 的参数分成“应用权重衰减”和“不应用权重衰减”两组，随后交给 AdamW 优化器。

这段代码本身还没有更新参数，只是构造优化器所需的参数分组。

### 1. Adam 如何更新参数

先约定下面的符号：

- $\theta_{t-1}$：第 $t$ 次更新前的模型参数；
- $g_t=\nabla_\theta L(\theta_{t-1})$：当前 batch 对参数的梯度；
- $\eta_t$：第 $t$ 次更新使用的学习率；
- $\beta_1,\beta_2$：一阶矩和二阶矩的衰减系数，对应配置中的 `betas`；
- $\epsilon$：防止分母为零的极小正数，对应配置中的 `eps`。

普通 SGD 直接沿当前梯度的反方向更新参数：

$$
\theta_t=\theta_{t-1}-\eta_t g_t
$$

它对所有参数使用同一个全局学习率，没有根据不同参数的梯度历史分别调整步长。

Adam 则为每个参数维护两组历史统计量。第一组是一阶矩 $m_t$，即梯度的指数移动平均：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$m_t$ 类似动量：如果某个参数连续多次朝相同方向产生梯度，这些方向会累积；如果梯度方向频繁反转，正负梯度会部分抵消。

第二组是二阶矩 $v_t$，即梯度平方的指数移动平均：

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

其中 $g_t^2$ 是逐元素平方。$v_t$ 用于估计每个参数近期梯度的尺度和波动：梯度长期较大的参数具有较大的 $v_t$，其后续更新会被更强地缩小。

Adam 初始化时通常令：

$$
m_0=0,\qquad v_0=0
$$

因此训练刚开始时，$m_t$ 和 $v_t$ 会因为从零开始而系统性偏小。Adam 使用下面的偏差修正：

$$
\hat m_t=\frac{m_t}{1-\beta_1^t}
$$

$$
\hat v_t=\frac{v_t}{1-\beta_2^t}
$$

完成偏差修正后，Adam 的参数更新为：

$$
\boxed{
\theta_t=
\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
$$

所有乘法、除法、平方和开方都是逐元素进行的。对第 $j$ 个参数单独看，它的更新量为：

$$
\Delta\theta_{t,j}
=-\eta_t\frac{\hat m_{t,j}}{\sqrt{\hat v_{t,j}}+\epsilon}
$$

所以，$\eta_t$ 是全局学习率，而分母 $\sqrt{\hat v_{t,j}}+\epsilon$ 会为每个参数产生不同的有效缩放。这就是 Adam 被称为“自适应优化器”的主要原因。

`betas` 决定两组移动平均保留多少历史信息：$\beta_1$ 越大，方向变化越平滑；$\beta_2$ 越大，梯度尺度估计变化越慢。$\epsilon$ 主要保证数值稳定，但当 $\sqrt{\hat v_t}$ 本身很小时，它也会实际影响更新步长。

### 2. AdamW 增加的解耦权重衰减

AdamW 在 Adam 自适应更新之外，单独对指定参数执行权重衰减。令 $\lambda$ 表示 `weight_decay`，衰减步骤为：

$$
\theta'_{t-1}
=\theta_{t-1}-\eta_t\lambda\theta_{t-1}
=(1-\eta_t\lambda)\theta_{t-1}
$$

随后再应用 Adam 根据任务梯度计算出的自适应更新。将两个步骤合起来，可以写成：

$$
\boxed{
\theta_t=
(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
}
$$

这里的 $m_t$ 和 $v_t$ 仍然只由任务损失的原始梯度 $g_t$ 计算：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t
$$

$$
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2
$$

“解耦”表示 $\lambda\theta_{t-1}$ 不进入 Adam 的 $m_t$ 和 $v_t$，也不接受 $\sqrt{\hat v_t}+\epsilon$ 的逐参数缩放。权重衰减仍受学习率影响，单步的实际收缩比例为：

$$
\eta_t\lambda
$$

例如，若当前学习率 $\eta_t=0.001$，`weight_decay=0.01`，则一次更新的衰减因子为：

$$
1-0.001\times0.01=0.99999
$$

这表示该步把需要衰减的参数缩小约 $0.001\%$，与 $1\%$ 有三个数量级的差距。如果暂时忽略梯度更新，并假设学习率保持不变，那么经过 $T$ 次实际参数更新后：

$$
\theta_T=(1-\eta\lambda)^T\theta_0
$$

当 $\eta\lambda$ 很小时，还可以近似为：

$$
\theta_T\approx e^{-\eta\lambda T}\theta_0
$$

若学习率随训练变化，则累计衰减为：

$$
\theta_T=\theta_0\prod_{t=1}^{T}(1-\eta_t\lambda)
\approx\theta_0\exp\left(-\lambda\sum_{t=1}^{T}\eta_t\right)
$$

因此，`weight_decay` 的实际效果还与学习率曲线和参数更新次数有关。相同的 `weight_decay` 配合不同的峰值学习率、训练步数或 scheduler，最终收缩强度可能明显不同。

本项目创建优化器时写的是：

```python
optimizer = optim.AdamW(param_groups, lr=1, weight_decay=0, betas=betas, eps=eps)
```

这里的 `lr=1` 是给 `LambdaLR` 使用的基础学习率。自定义 `LinearWarmUpCosineAnnealingLR` 返回 `init`、`peak`、`final` 所描述的学习率曲线；因为基础值是 1，`1 × scheduler 返回值` 就是参数组当时的实际学习率。AdamW 的梯度更新和权重衰减都使用这个实际学习率。学习率与梯度累积的调用时序见前文“从 `scaler.step()` 到 `scheduler.step()`”。

权重衰减可以限制线性层和卷积层权重持续增大，通常用于缓解过拟合；但梯度更新仍可能把参数推大，因此它不保证权重范数单调下降。

### 3. AdamW 与“Adam + L2 正则化”的区别

传统 L2 正则化是在任务损失中加入参数平方和。令任务损失为 $L_{\text{task}}$，则加入 L2 后的总损失为：

$$
L_{\text{total}}(\theta)
=L_{\text{task}}(\theta)
+\frac{\lambda}{2}\lVert\theta\rVert_2^2
$$

因为：

$$
\nabla_\theta\frac{\lambda}{2}\lVert\theta\rVert_2^2
=\lambda\theta
$$

所以 Adam 接收到的总梯度变成：

$$
g_t^{\text{L2}}=g_t+\lambda\theta_{t-1}
$$

如果把这个梯度直接交给 Adam，那么一阶矩和二阶矩会更新为：

$$
m_t=\beta_1m_{t-1}
+(1-\beta_1)(g_t+\lambda\theta_{t-1})
$$

$$
v_t=\beta_2v_{t-1}
+(1-\beta_2)(g_t+\lambda\theta_{t-1})^2
$$

最终更新为：

$$
\theta_t=\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}
$$

此时 L2 项不仅影响当前更新方向，还会进入动量历史 $m_t$ 和梯度平方历史 $v_t$，随后被 Adam 的逐参数自适应分母缩放。也就是说，同样大小的参数，在不同的梯度历史下可能受到不同强度的有效收缩。

AdamW 的做法则是：

1. 用纯任务梯度 $g_t$ 更新 $m_t$ 和 $v_t$；
2. 用 $(1-\eta_t\lambda)\theta_{t-1}$ 直接衰减参数；
3. 再应用 Adam 自适应梯度项。

因此，在 Adam 这类自适应优化器中，两种方法通常不等价：

| 方法 | 正则/衰减项是否进入 `m`、`v` | 是否被 Adam 自适应缩放 |
| --- | --- | --- |
| Adam + 损失中的 L2 项 | 是 | 是 |
| AdamW 的 `weight_decay` | 否 | 否 |

对于不带动量和自适应缩放的普通 SGD，两者可以得到相同的代数形式：

$$
\theta_t
=\theta_{t-1}-\eta_t(g_t+\lambda\theta_{t-1})
$$

整理后为：

$$
\theta_t
=(1-\eta_t\lambda)\theta_{t-1}-\eta_tg_t
$$

但 Adam 会对总梯度维护移动平均并按二阶矩缩放，所以这个等价关系不再成立。AdamW 的核心价值正是把参数收缩从 Adam 的自适应梯度计算中分离出来。

因此，已经使用 AdamW 的 `weight_decay` 时，通常不应再无意地向 loss 添加同一套 L2 正则项，否则会同时施加两种正则化。只有在明确希望同时使用两种机制时，才应该有意这样配置。

### 本项目为什么要分组

一般不希望所有参数都衰减：

| 参数类型 | 是否衰减 | 原因 |
| --- | --- | --- |
| `Linear.weight` | 是 | 主要的权重矩阵，适合正则化 |
| `Conv1d.weight` | 是 | 主要的卷积核参数，适合正则化 |
| `Linear.bias` | 否 | 偏置通常不需要通过收缩控制容量 |
| `Conv1d.bias` | 否 | 同上 |
| `BatchNorm.weight` | 否 | 它是归一化后的缩放系数，衰减可能干扰归一化 |
| `BatchNorm.bias` | 否 | 它是归一化后的平移系数 |
| 其他参数 | 否 | 当前规则采取保守策略，只明确衰减 `Linear`/`Conv1d` 权重 |

## 第 278～286 行：读取与清洗玩家白名单

这段逻辑只在离线模式中执行。`dataset.player_names_files` 可以指定多个纯文本文件，每行一个玩家名：

```python
player_names_set = set()
for filename in config['dataset']['player_names_files']:
    with open(filename) as f:
        player_names_set.update(filtered_trimmed_lines(f))
player_names = list(player_names_set)
```

`filtered_trimmed_lines` 定义在 [`common.py`](../mortal/common.py) 第 16～17 行：

```python
def filtered_trimmed_lines(lines):
    return filter(lambda l: l, map(lambda l: l.strip(), lines))
```

它先对每行执行 `strip()`，删除首尾空格、制表符和换行符；再过滤清理后的空字符串。返回值是惰性、只能按顺序消费的 `filter` 迭代器，`set.update()` 会在消费它的同时合并多个文件并去重。例如：

```text
原文件：             最终 player_names_set：
  Alice                {"Alice", "Bob"}

Bob
Alice
```

该函数不会过滤注释、进行大小写归一化或模糊匹配；玩家名中间的空格也会保留。因此 `# comment` 会被当成真实玩家名，`Alice` 和 `alice` 也是两个不同名称。

空的 `player_names_files` 或只含空白行的名单会产生空集合。在本项目中，空玩家集合表示“不限制玩家”，样本仍会正常生成。非空集合则有两层用途：

1. 建立文件索引时，排除完全不含目标玩家的牌谱；
2. 后续传给 `GameplayLoader`，在已保留的整场牌谱中只为白名单玩家生成决策样本。

例如一场牌谱含 `Alice` 和其他三人，白名单只有 `Alice`：整份牌谱会进入数据加载器，但只产生 `Alice` 视角的训练样本。

## 第 288～325 行：构建与复用牌谱文件索引

`dataset.file_index` 指向一个由 `torch.save()` 写出的路径缓存，其核心结构为：

```python
{
    "file_list": [
        "/data/logs/game003.json.gz",
        "/data/logs/game002.json.gz",
        "/data/logs/game001.json.gz",
    ]
}
```

它只保存牌谱文件路径，不保存观测、动作、奖励、文件哈希或白名单元数据。源码中紧随赋值的三引号内容只是未赋值的多行字符串示例，不会构造字典，也不影响训练。

代码按索引是否存在分为两条路径：

| 情况 | 行为 |
| --- | --- |
| `file_index` 已存在 | 使用 `torch.load(..., weights_only=True)` 安全加载，直接取出 `index['file_list']` |
| `file_index` 不存在 | 递归展开 `dataset.globs`，可选地按玩家名过滤，逆序排列路径后写入索引 |

建立索引时，`glob(pat, recursive=True)` 把每个模式的匹配结果追加到 `file_list`。不同模式若匹配到同一文件，当前代码不会去重。相对 glob 以启动 `train.py` 时的当前工作目录为基准。

当玩家白名单非空时，脚本只打开每份 gzip 牌谱的第一行，把 `start_game.names` 与白名单转成的集合比较：

```python
if not set(start['names']).isdisjoint(player_names_set):
    filtered.append(filename)
```

`isdisjoint()` 在两集合没有任何交集时返回 `True`，所以外层 `not` 表示“四名玩家中至少一人在白名单中”。这一层以整场牌谱为过滤单位，样本级过滤由后续 `GameplayLoader` 执行。白名单为空时跳过此步，保留所有 glob 结果。

最后，`file_list.sort(reverse=True)` 按路径字符串逆序排列，并用 `torch.save({'file_list': file_list}, file_index)` 缓存。此顺序与文件修改时间无关；后续 `FileDatasetsIter` 还会在每轮重新打乱路径，所以最终训练顺序会再次变化。第 325 行记录牌谱文件数，该数值不包含样本数、batch 数或训练步数。

索引不会自动失效。已有索引时，脚本不会重新扫描 glob，也不会重做文件级玩家过滤。因此修改 glob、白名单或源牌谱后，必须确保 `file_index` 与新数据范围一致：过宽的旧索引主要增加无效 I/O，过窄的旧索引则会使目标牌谱根本无法进入数据加载器。
