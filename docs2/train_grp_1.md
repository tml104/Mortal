# 顺位预测训练：样本与变长序列

本文解释 [`train_grp.py`](../mortal/train_grp.py) 和 [`train_grp_transformer.py`](../mortal/transformer_pipeline/train_grp_transformer.py) 如何把一场对局变成训练样本，以及 GRU/Transformer 如何处理不同长度的前缀。

两种模型的共同目标是：

> 只观察对局进行到当前局开始时的信息，预测四位玩家的最终顺位。

## 数据从哪里来

[`libriichi.dataset.Grp`](../libriichi/src/dataset/grp.rs) 逐行解析 mjai 日志。每遇到 `start_kyoku`，记录一行 7 维特征：

```text
[grand_kyoku, honba, kyotaku, score_0/10000, score_1/10000, score_2/10000, score_3/10000]
```

| 字段 | 含义 |
| --- | --- |
| `grand_kyoku` | 跨场风的局编号，东一为 0、南四为 7、西四为 11 |
| `honba` | 本场数 |
| `kyotaku` | 立直棒数 |
| `score_0..3` | 按绝对玩家 ID 排列的分数，除以 10000 |

若一场对局产生 `T` 行，`feature.shape == (T, 7)`。标签 `rank_by_player` 是长度 4 的零基顺位数组，例如：

```text
[0, 2, 3, 1]
```

表示玩家 0/1/2/3 最终分别为第 1/3/4/2 名。

## 一场对局如何展开为样本

训练脚本不会把整场日志只当成一个样本，而是生成所有非空前缀：

```python
for i in range(feature.shape[0]):
    inputs_seq = feature[:i + 1]
    sample = (inputs_seq, rank_by_player)
```

假设特征序列是：

```text
t1 = [0, 0, 0, 2.5, 2.5, 2.5, 2.5]
t2 = [1, 0, 0, 2.8, 2.2, 2.5, 2.5]
t3 = [2, 1, 0, 3.1, 2.0, 2.4, 2.5]
```

则生成：

| 样本 | 输入 | 形状 | 标签 |
| --- | --- | --- | --- |
| 1 | `[t1]` | `(1, 7)` | 最终 `rank_by_player` |
| 2 | `[t1, t2]` | `(2, 7)` | 同上 |
| 3 | `[t1, t2, t3]` | `(3, 7)` | 同上 |

因此长度为 `T` 的对局贡献 `T` 个样本。前缀越短，模型可用的信息越少；共享终局标签迫使模型学习顺位判断如何随比赛推进而更新。

## 为什么标签是 24 类

四位玩家的完整顺位共有：

```text
4! = 24
```

种排列。`GRP.get_label()` 把 `rank_by_player` 映射为 0..23 的类别，模型用交叉熵训练：

```python
labels = grp.get_label(rank_by_players)
loss = cross_entropy(logits, labels)
```

预测完整排列能天然避免“两位玩家同时第一”之类的不合法组合。需要单个玩家的顺位概率时，再由 `calc_matrix()` 对 24 类概率求边缘分布。

## 为什么需要自定义 `collate`

一个 batch 可能同时包含 `(1, 7)`、`(3, 7)`、`(8, 7)` 等变长张量。PyTorch 默认 `collate` 无法直接 `stack` 它们，所以两个训练脚本都实现了自己的组批函数。

共同的第一步是：

```python
padded = pad_sequence(inputs, batch_first=True)
```

输出形状为：

```text
(batch_size, max_seq_len, 7)
```

短序列在尾部补 `[0, 0, 0, 0, 0, 0, 0]`。补零只用于对齐形状；模型还必须知道哪些位置无效。

## GRU：pad 后再 pack

GRU 版记录每条序列的真实长度，并执行：

```python
packed_inputs = pack_padded_sequence(
    padded,
    lengths,
    batch_first=True,
    enforce_sorted=False,
)
```

`PackedSequence` 保存两类信息：

- 所有真实时间步组成的紧凑数据；
- 每个时间步仍有多少序列有效。

例如三条序列长度为 `[1, 2, 4]`，内部按长度处理后，有效 batch 大小依次是：

```text
batch_sizes = [3, 2, 1, 1]
```

GRU 在短序列结束后不再更新它的隐藏状态，因此尾部补零既不参与计算，也不会污染最终隐藏状态。

`enforce_sorted=False` 允许 DataLoader 保持当前随机顺序，由 PyTorch 内部处理长度排序。调用者不必手动排序和恢复标签顺序。

模型前向为：

```text
PackedSequence，每步 7 维
-> GRU
-> state: (num_layers, batch, hidden_size)
-> 转置并拼接所有层
-> MLP
-> (batch, 24) logits
```

注意 `GRP.forward(list_of_tensors)` 会在模型内部完成 pad/pack；训练脚本的 collate 已经生成 `PackedSequence`，所以直接调用 `forward_packed()`，避免重复整理。

## Transformer：pad 后构造 mask

Transformer 不接受 `PackedSequence`。其 collate 返回：

```text
padded_inputs: (B, L, 7)
padding_mask:  (B, L), bool
labels:        (B, 4)
```

mask 的规则是：

```text
False = 真实时间步
True  = padding，需要被注意力忽略
```

模型执行：

```text
padded_inputs
-> 7 维投影到 d_model
-> 加位置 embedding
-> 头部拼接 [CLS]
-> padding_mask 头部补一列 False
-> TransformerEncoder
-> 取 [CLS] 输出
-> MLP -> 24 logits
```

`[CLS]` 用于聚合整条序列，永远不被 mask。位置 embedding 只有 `max_seq_len` 个位置；输入前缀超过该值会越界，因此配置必须覆盖数据可能出现的最长对局序列。

训练脚本的 collate 已产生 padded tensor 和 mask，所以调用 `forward_padded()`。模型的普通 `forward(list_of_tensors)` 适用于其他调用方，会自行完成 padding 和 mask。

## 两种实现对照

| 项目 | `GRP` | `GRP_Transformer` |
| --- | --- | --- |
| 序列核心 | GRU | TransformerEncoder |
| 组批结果 | `PackedSequence` | padded tensor + bool mask |
| 全局表示 | 所有 GRU 层的最终隐藏状态 | `[CLS]` 输出 |
| 时间位置信息 | 递归顺序天然携带 | 可学习位置 embedding |
| 长度限制 | 无显式固定上限 | 受 `max_seq_len` 限制 |
| 配置节 | `[grp]` | `[grp_transformer]` |
| checkpoint | 结构独立 | 结构独立 |

两者都：

- 使用相同的 7 维特征和 24 类标签；
- 将模型参数转为 `float64`；
- 使用 AdamW 和交叉熵；
- 每 `save_every` 步执行 `val_steps` 次验证；
- 保存 `model`、`optimizer`、`steps`、`timestamp`。

## 文件加载和训练循环

`GrpFileDatasetsIter` 以文件批次工作：

1. 打乱日志文件列表；
2. 每次用 Rust `Grp.load_gz_log_files()` 读取 `file_batch_size` 个文件；
3. 把各对局展开成所有前缀并放入内存 buffer；
4. 随机遍历 buffer，随后清空；
5. `cycle=True` 时重新打乱并无限循环。

训练集和验证集都使用无限迭代器，训练由人工中断；`save_every` 控制验证与落盘频率，不代表完整 epoch。日志中显示的训练百分比是基于“每文件约 10 个前缀”的粗略估计，不是精确进度。

## 常见误解

- **补零会不会被 GRU 当成真实数据？** 不会，GRU 收到的是带真实长度的 `PackedSequence`。
- **Transformer 能否复用 PackedSequence？** 不能，它依赖 dense tensor 和 attention padding mask。
- **每场对局只有一个样本吗？** 不是，长度 `T` 会展开成 `T` 个前缀样本。
- **输出是每位玩家独立四分类吗？** 不是，先预测一个合法的 24 类完整排列。
- **Transformer checkpoint 能否直接放进 `grp.state_file`？** 不能，网络参数结构和标准主干加载类都不匹配。
- **补零向量会与真实东一局特征混淆吗？** mask/pack 明确标识有效长度，模型不会靠数值本身判断 padding。
