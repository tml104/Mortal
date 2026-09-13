# 深度学习基础概念与用到的网络介绍

本文先解释阅读 Mortal 网络代码时会遇到的基础名词，再介绍 [`mortal/model.py`](../mortal/model.py) 中 `Brain` 使用的具体网络组件。本文不讨论训练算法。

## 张量和形状

张量可以先理解为“有多个维度的数字表”。例如一批 Mortal v4 观测的形状是 `(B, 1012, 34)`：

- `B` 是一次同时处理的局面数，即 batch size（批大小）；
- `1012` 是特征通道数；
- `34` 是不区分赤牌的 34 种基本牌。

形状只说明数字如何排列，不说明数字的含义。同为一行 34 个数，有的通道表示“手中有哪些牌”，有的通道则把当前分数重复写满整行。Mortal 的具体输入见 [network.md 的“v4 的 1012 个观测通道”](network.md#v4-的-1012-个观测通道)。

## 神经网络层在做什么

神经网络由许多带参数的计算层组成。训练就是根据样本调整这些参数，使输出更接近训练目标；推理则固定参数，只做计算。

`Linear`（全连接层）会对所有输入做加权求和，再加上偏置。若输入是 1024 个数、输出是 47 个数，便可以写成：

```text
y = Wx + b
W 的形状为 (47, 1024)，b 的形状为 (47,)
```

仅把线性层反复叠加，表达能力仍等价于一个线性变换。因此网络会在线性层或卷积层之间加入 ReLU、Mish 等非线性激活函数。Mortal 使用的卷积、激活和残差结构见下文的 [Conv1d](#conv1d沿牌种轴观察相邻位置)、[BatchNorm](#batchnorm统一数值尺度)、[Mish](#mish加入非线性) 和 [残差块](#残差块学习对原表示的修正)。

## logits、softmax 和概率

模型经常先输出一组没有归一化的分数，通常称为 **logits**。softmax 再把它们转换成总和为 1 的概率：

```text
p_i = exp(logit_i) / sum(exp(logit_j))
```

例如三个 logits 为 `[2, 1, 0]`，softmax 后约为 `[0.665, 0.245, 0.090]`。最大项仍是第一项，但现在三个数都在 0 到 1 之间，且总和为 1。

把某个非法动作的 logit 设为负无穷后，`exp(-inf) = 0`，它的概率也就是 0；其余合法动作仍会被归一化到总和为 1。PyTorch 的 [`softmax`](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html) 和 [`Categorical`](https://docs.pytorch.org/docs/stable/distributions.html#categorical) 分别提供显式归一化和按 logits 采样。

## Q 值不是概率

强化学习中的 **Q 值**用一个数评价“在当前状态选择某动作，预期能得到多好的长期结果”。它可以是任意实数，各动作的 Q 值不需要大于 0，也不需要相加为 1。

[`mortal/model.py`](../mortal/model.py) 中的 `DQN` 一次输出 46 个动作的 Q 值。默认推理直接选择合法动作中 Q 值最大的一个，所以根本不需要概率。只有启用随机采样时，[`mortal/engine.py`](../mortal/engine.py) 中的 `MortalEngine` 才把 `Q / temperature` 当作 logits，经 softmax 转成概率。完整过程见 [network.md 的“Q 值何时会变成概率”](network.md#q-值何时会变成概率)。

## Conv1d：沿牌种轴观察相邻位置

本文提到的 `Brain`、`ResNet`、`ResBlock` 和 `ChannelAttention` 类都定义在 [`mortal/model.py`](../mortal/model.py)；`Conv1d`、`BatchNorm1d` 和 `Mish` 则是这些类调用的 PyTorch 标准网络层。

`Conv1d` 是一维卷积。Mortal 把观测排成 `(B, channels, 34)`，所以卷积沿最后的 34 种牌移动。`kernel_size=3` 表示每次同时查看当前位置及其左右相邻位置，并把所有输入通道的信息加权组合成新的通道。

以 v4 的第一层和示例配置为例：

```text
输入 (B, 1012, 34)
-> Conv1d(1012, 192, kernel_size=3, padding=1)
输出 (B, 192, 34)
```

这一层有 192 个卷积核，每个卷积核都查看全部 1012 个输入通道和连续 3 个牌种位置，因此权重张量的形状是 `(192, 1012, 3)`。第一层在源码中设置了 `bias=False`，所以单个输出数值可以写成：

```text
y[b, o, t] = sum(c=0..1011) sum(k=0..2) W[o, c, k] * x[b, c, t + k - 1]
```

其中 `b` 是局面编号，`o` 是 192 个输出通道之一，`t` 是 34 个牌种位置之一。超出牌种轴两端的输入按 0 处理。对固定的 `b`、`o`、`t`，程序会收集该局面的 1012 个通道在左侧、当前位置和右侧的值，共最多 `1012 * 3` 个数，分别乘以权重再求和，得到一个输出值。

先用“2 个输入通道、1 个输出通道、3 个位置”的缩小例子手算一次。假设输入和卷积核是：

```text
输入通道 0：x[0] = [1, 2, 0]
输入通道 1：x[1] = [0, 1, 3]

通道 0 的核：W[0] = [1,   0, -1]
通道 1 的核：W[1] = [0.5, 0, 0.5]
```

计算中间位置 `t=1` 时，卷积会查看每个通道的三个位置：

```text
通道 0 的贡献 = 1*1   + 2*0 + 0*(-1) = 1
通道 1 的贡献 = 0*0.5 + 1*0 + 3*0.5  = 1.5
输出             = 1 + 1.5 = 2.5
```

实际 v4 第一层做的是同一种计算，只是每次汇总 1012 个输入通道，并用 192 组不同权重各算一次，因此每个牌种位置从 1012 个输入特征变成 192 个学习特征。相同的 192 组卷积核会沿 34 个位置重复使用。

`padding=1` 会在整个牌种轴的左右两端各补一个 0，因此卷积前后的长度仍为 34。卷积可以学习万、筒、索中相邻数牌的关系，例如二万、三万、四万可能组成顺子；代码没有硬编码花色边界，边界处的模式也由训练学习。卷积始终在同一个 `b` 内计算，不会把 batch 中 A 局面的数值和 B 局面混在一起。

与全连接层相比，同一组卷积参数会在 34 个位置重复使用。这既减少参数，也让模型能以相似方式处理不同位置的局部组合。具体计算和形状规则见 PyTorch 的 [`Conv1d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)。

## BatchNorm：统一数值尺度

`BatchNorm1d`（批归一化）可以先理解为：把同一特征通道中的数值调整到较稳定的中心和尺度，再允许网络学习最合适的缩放与平移。这样后续层不必不断适应前面层数值范围的剧烈变化，深层网络通常会更容易训练。

对形状为 `(B, C, L)` 的输入，Mortal 的 `BatchNorm1d` 会分别处理 `C` 个通道。训练时，每个通道使用 batch 维和长度维上的数值计算均值 `mu` 和方差 `var`：

```text
x_hat = (x - mu) / sqrt(var + eps)
y     = gamma * x_hat + beta
```

`x_hat` 是标准化后的值；`eps` 是防止除零并改善数值稳定性的小常数；`gamma` 和 `beta` 是训练得到的缩放和偏移参数。因此 BatchNorm 不是把所有通道强行变成永久相同的分布，网络仍可通过 `gamma`、`beta` 恢复或改变需要的尺度。

用一个通道中的三个数 `[2, 4, 6]` 举例。它们的均值是 4，方差是 `8/3`。暂时忽略很小的 `eps`，标准化后约为：

```text
(2 - 4) / sqrt(8/3) = -1.225
(4 - 4) / sqrt(8/3) =  0
(6 - 4) / sqrt(8/3) =  1.225
```

原来的数值以 4 为中心且范围较大，标准化后以 0 为中心、尺度更统一。如果学到 `gamma=2`、`beta=0.5`，最终输出还会变成约 `[-1.95, 0.5, 2.95]`，说明网络仍能选择合适的范围。

训练时，每个 batch 的均值和方差还会用于更新累计统计量；[`MortalEngine`](../mortal/engine.py) 构造模型时调用 `.eval()`，普通推理因此使用训练期间保存的累计均值和方差，而不依赖当前送入几个局面。[`mortal/model.py`](../mortal/model.py) 为这些层设置 `momentum=0.01`，v3 和 v4 另设 `eps=1e-3`。详见 PyTorch 的 [`BatchNorm1d` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)。

## Mish：加入非线性

Mish 的数学表达式是：

```text
Mish(x) = x * tanh(softplus(x))
        = x * tanh(ln(1 + exp(x)))
```

它逐个处理数值，不改变张量形状。很大的正数经过 Mish 后接近原值；负数不会像 ReLU 那样全部直接变成 0，而是保留一部分平滑的负输出。平滑的非线性让多层卷积不再等价于一个简单线性变换，从而能够表示更复杂的局面规律。详见 PyTorch 的 [`Mish` 文档](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html)。

## 残差块：学习对原表示的修正

普通的深层网络把每层输出直接交给下一层。残差块（residual block）额外保留原输入 `x`，让卷积分支只需学习修正量 `F(x)`：

```text
输出 = x + F(x)
```

如果某些层暂时没有学到有用修正，原信息仍可沿加法路径继续传递。这有助于训练很深的网络。Mortal v4 使用 pre-activation（预激活）布局：

```text
x
|-> BatchNorm -> Mish -> Conv1d
|-> BatchNorm -> Mish -> Conv1d -> ChannelAttention
`-----------------------------------------------> 相加
```

两层卷积都保持 `(B, conv_channels, 34)`，所以结果才能与 `x` 逐项相加。Mortal 的 `num_blocks` 决定这样的残差块重复多少次。残差网络的原始思路见 [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)，预激活布局见 [Identity Mappings in Deep Residual Networks](https://arxiv.org/abs/1603.05027)。

## ChannelAttention：调整各通道的重要程度

### 它在 `ResBlock` 中的位置

[`mortal/model.py`](../mortal/model.py) 中的 `ResBlock.forward()` 依次执行：

```text
u = F(x)                    两层卷积产生残差修正
u_att = ChannelAttention(u) 按通道调整残差修正
y = x + u_att               与原输入的恒等分支相加
```

这里用 `x` 表示残差块的原输入，用 `F(x)` 或 `u` 表示经过 `BatchNorm -> Mish -> Conv1d -> BatchNorm -> Mish -> Conv1d` 后的结果。`ChannelAttention` 只处理 `u`，不会直接缩放恒等分支 `x`。因此更完整的残差公式不是简单的 `y = x + F(x)`，而是：

```text
y = x + ChannelAttention(F(x))
```

以 [`mortal/config.example.toml`](../mortal/config.example.toml) 的 `conv_channels=192` 为例，v4 的一个 `ResBlock` 中各步形状如下。`B` 是一次处理的局面数，34 是牌种轴长度：

| 步骤 | 对应计算 | 形状 |
| --- | --- | --- |
| 残差块输入 | `x` | `(B, 192, 34)` |
| 两层卷积后的残差修正 | `u = F(x)` | `(B, 192, 34)` |
| 沿牌种轴求平均值 | `avg = u.mean(-1)` | `(B, 192)` |
| 沿牌种轴求最大值 | `max = u.amax(-1)` | `(B, 192)` |
| 平均值路径的共享 MLP | `192 -> 12 -> 192` | `(B, 192)` |
| 最大值路径的共享 MLP | `192 -> 12 -> 192` | `(B, 192)` |
| 相加并取 sigmoid | `weight` | `(B, 192)` |
| 在末尾增加一维 | `weight.unsqueeze(-1)` | `(B, 192, 1)` |
| 乘回残差修正 | `u_att` | `(B, 192, 34)` |
| 与恒等分支相加 | `y = x + u_att` | `(B, 192, 34)` |

`ChannelAttention` 的默认压缩比例 `ratio=16`，所以共享 MLP 的中间维度是 `192 // 16 = 12`。在 v4 中，`Brain` 把 Mish 作为 `actv_builder` 传给 `ResBlock`，因此这里的共享 MLP 实际是：

```text
Linear(192, 12) -> Mish -> Linear(12, 192)
```

平均值路径和最大值路径调用的是**同一个** `shared_mlp`，即两条路径共享两层 Linear 的参数，并非各自训练一套 MLP。这个 MLP 的输入是一整行 192 个通道概要，因此某个通道最终得到多大权重，可以同时参考其他 191 个通道的活动情况，而不是只检查该通道自己的平均值或最大值。

### 逐步计算公式

令 `u[b,c,t]` 表示第 `b` 个局面中，残差分支第 `c` 个通道在第 `t` 个牌种位置的数值。先把每个通道的 34 个位置压缩为平均值和最大值：

```text
avg[b,c] = (1 / 34) * sum(t=0..33) u[b,c,t]
max[b,c] = max(t=0..33) u[b,c,t]
```

再令 `g` 表示共享 MLP：

```text
g(z) = Linear_2(Mish(Linear_1(z)))
```

两条路径的结果相加后经过 sigmoid，产生每个通道的权重：

```text
weight[b,:] = sigmoid(g(avg[b,:]) + g(max[b,:]))
```

`weight` 原本只有 `(B, 192)`。`unsqueeze(-1)` 在末尾增加一个长度为 1 的维度，使其成为 `(B, 192, 1)`；与 `(B, 192, 34)` 的 `u` 相乘时，PyTorch 会把同一个通道权重广播到该通道的全部 34 个牌种位置：

```text
u_att[b,c,t] = weight[b,c] * u[b,c,t]
y[b,c,t]     = x[b,c,t] + u_att[b,c,t]
```

形状在整个过程中最终仍是 `(B, 192, 34)`，所以注意力处理后的残差修正可以和原输入 `x` 逐项相加。

### 一个可以手算的例子

实际网络有 192 个通道。为了避免列出 `192 * 34` 个数，下面只观察其中两个代表通道，其他 190 个通道仍然存在并照常参与共享 MLP 的计算。假设某个局面的两层卷积输出 `u = F(x)` 中：

```text
通道 c0：34 个位置中只有两个位置是 1，其余 32 个位置是 0
通道 c1：34 个位置全部是 0.4
```

那么两种池化得到：

```text
avg[c0] = 2 / 34 = 0.0588          max[c0] = 1.0
avg[c1] = 0.4                      max[c1] = 0.4
```

平均值揭示了两者的分布差异：`c0` 只在少数位置强烈响应，所以平均值很小；`c1` 在所有位置持续响应，所以平均值较大。最大值则保留了 `c0` 曾出现强响应这一事实，不会让它在求平均时被大量 0 完全掩盖。

接下来，共享 MLP 会同时查看全部 192 个通道的 `avg` 和 `max`。为了演示后续计算，假设训练后的 MLP 在考虑所有通道后，为 `c0` 产生的两路结果之和是 `1.0`，为 `c1` 产生的两路结果之和是 `-1.0`：

```text
score[c0] = g(avg)[c0] + g(max)[c0] =  1.0
score[c1] = g(avg)[c1] + g(max)[c1] = -1.0

weight[c0] = sigmoid( 1.0) = 0.731
weight[c1] = sigmoid(-1.0) = 0.269
```

这里的 `1.0` 和 `-1.0` 是为了便于手算而假设的 MLP 输出，不是源码中的固定规则。真实数值由模型参数和当前局面共同决定；网络也完全可能在另一个局面中给予 `c1` 更高权重。

权重乘回残差分支后：

```text
c0 原来为 1 的两个位置：1 * 0.731 = 0.731
c0 原来为 0 的其他位置：0 * 0.731 = 0
c1 的全部位置：          0.4 * 0.269 = 0.1076，约为 0.108
```

最后再与未经缩放的恒等分支 `x` 相加。假设某个强响应位置的 `x[c0,t] = 0.5`，另一个位置的 `x[c1,t] = 0.3`，则：

```text
y[c0,t] = 0.5 + 0.731 * 1.0 = 1.231
y[c1,t] = 0.3 + 0.269 * 0.4 = 0.4076，约为 0.408
```

这说明 ChannelAttention 调整的是“残差分支想对原表示做多大修正”。即使某个残差通道被明显抑制，原始的 `x` 仍能沿恒等分支传到下一层。

### 它的功能效果

- **同时概括普遍响应和突出响应。** 平均池化反映一个特征是否在许多牌种位置普遍出现，最大池化保留某个位置是否出现过特别强的信号。两者结合比只使用其中一种提供更多信息。
- **学习通道之间的关系。** `192 -> 12 -> 192` 的 MLP 会混合全部通道。例如模型可以根据若干“顺子结构”“可见牌”“进张”类学习特征是否同时活跃，决定另一通道的残差修正是否值得保留。这里的通道是网络学习出的内部特征，不保证能逐项对应人工命名的麻将概念。
- **随局面动态变化。** 权重由当前 `u` 计算，而不是训练完成后固定为一组常数。同一通道在不同局面中可以得到不同权重；40 个 `ResBlock` 也各自拥有独立的 ChannelAttention 参数，会在不同深度重新判断残差特征的重要程度。
- **控制修正量而不切断原信息。** 注意力位于残差分支、恒等相加之前。权重较小会减少该通道对 `x` 的修改，但不会删除恒等分支携带的状态，这保留了残差网络易于传递信息和梯度的特点。
- **保持接口形状不变。** 输入和输出都是 `(B, 192, 34)`，后续残差块不需要知道注意力内部如何计算。

训练时，最终损失的梯度会同时更新卷积层和共享 MLP。模型由此学习“在哪类局面中保留哪些残差通道，更有利于最终的价值估计和动作选择”，而不需要人工为注意力权重提供标签。

### 它不能直接做到什么

- 192 个权重不是动作概率，也不要求总和为 1；sigmoid 只是独立地把每个权重限制到 0 和 1 之间。
- 一个通道只有一个权重，这个权重会同时作用于该通道的 34 个位置。池化在生成权重时已经丢掉具体位置信息，因此它是**通道注意力**，不是判断“一万位置比二万位置更重要”的逐牌位置注意力。
- sigmoid 权重不会把残差分支的绝对值直接放大到原来的 1 倍以上。这里所说的“增强重要通道”，主要指少抑制重要通道、更多抑制其他通道，从而提高前者的相对影响。
- ChannelAttention 不读取合法动作 mask，也不直接输出或选择动作。它只调整 `Brain` 内部的状态特征，随后再由 `DQN` 计算动作 Q 值。
- 这种结构提供了学习动态通道权重的能力，但不能仅凭结构断言它一定提高麻将强度。若要确认实际收益，需要比较保留与移除 ChannelAttention 的消融实验。

该实现取自 CBAM 的 channel attention 思路，论文见 [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)。

## `Brain` 如何把局面压缩成 1024 个数

以 v4 和示例配置为例，完整形状变化如下：

```text
(B, 1012, 34)                       原始观测
-> (B, 192, 34)                     第一层 Conv1d
-> (B, 192, 34)                     40 个残差块，形状不变
-> (B, 32, 34)                      最后一层 Conv1d
-> (B, 1088)                        Flatten：32 * 34
-> (B, 1024)                        Linear + Mish
```

这里的“压缩”是训练得到的特征变换，不是文件压缩，也不是简单删除输入。第一层卷积把原始通道组合为局部特征，残差块反复修正这些特征，最后一层卷积将通道数降到 32；`Flatten` 再把 32 个通道的 34 个位置按固定顺序摊成 1088 个数，全连接层让所有位置互相组合并生成 1024 个输出。

最后的 1024 维向量叫 `phi`，是 `Brain` 对整个局面的学习表示。这里的 **1024 是输出向量长度，不是 v4 的输入通道数**。每个 `phi` 位置的含义由训练共同决定，通常不能逐项解释为某个固定麻将概念，而且这个过程不保证能从 `phi` 还原原始观测。随后 [`mortal/model.py`](../mortal/model.py) 中的 `DQN` 根据 `phi` 计算动作价值。配置中的 `192` 和 `40` 可以改变；`32`、`34` 和 `1024` 则由当前模型代码固定。

## 参考资料

- [PyTorch：Tensors](https://pytorch.org/tutorials/beginner/basics/tensorqs_tutorial.html)
- [PyTorch：Linear](https://docs.pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- [PyTorch：Softmax](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.softmax.html)
- [PyTorch：Conv1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.Conv1d.html)
- [PyTorch：BatchNorm1d](https://docs.pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html)
- [PyTorch：Mish](https://docs.pytorch.org/docs/stable/generated/torch.nn.Mish.html)
- [Human-level control through deep reinforcement learning（DQN 论文）](https://doi.org/10.1038/nature14236)
- [Deep Residual Learning for Image Recognition（ResNet 论文）](https://arxiv.org/abs/1512.03385)
- [Identity Mappings in Deep Residual Networks（预激活 ResNet 论文）](https://arxiv.org/abs/1603.05027)
- [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521)
