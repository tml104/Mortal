# `train_grp.py` 阅读笔记

这份文档回答两个问题：

1. 前缀序列是怎么构造的？一局游戏的前缀序列可能长什么样？
2. `GRU` 的输入有 padding，padding 是怎么做的？

对应代码主要在：

- [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py)
- [libriichi/src/dataset/grp.rs](/home/tml104/Mortal/libriichi/src/dataset/grp.rs)
- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)

## 1. 先说结论

`train_grp.py` 训练的不是“整局日志 -> 一个样本”。

它做的是：

- 先从一局游戏里提取一个按局推进的特征序列 `feature`
- 然后把这条序列切成所有可能的前缀：
  - 第 1 局前缀
  - 前 2 局前缀
  - 前 3 局前缀
  - ...
- 每个前缀都作为一个训练样本
- 标签始终是这整场游戏最终的 `rank_by_player`

所以它训练的是：

> “只看到对局进行到当前这个时刻时的信息，去预测最终四家排名。”

## 2. 前缀序列是怎么构造的

关键代码在 [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py)：

```python
for game in data:
    feature = game.take_feature()
    rank_by_player = game.take_rank_by_player()

    for i in range(feature.shape[0]):
        inputs_seq = torch.as_tensor(feature[:i + 1], dtype=torch.float64)
        self.buffer.append((
            inputs_seq,
            rank_by_player,
        ))
```

这几行的意思是：

- `feature` 是一整场游戏的时间序列特征
- 如果这场游戏一共经历了 `T` 个时间步
- 那么训练集里会从这一场游戏生成 `T` 个样本

第 `i` 个样本就是：

```text
feature[:i+1]
```

也就是“从开头到当前时刻”的前缀序列。

### 2.1 `feature` 本身长什么样

`feature` 的定义在 [libriichi/src/dataset/grp.rs](/home/tml104/Mortal/libriichi/src/dataset/grp.rs)：

```rust
// [grand_kyoku, honba, kyotaku, [score[i] / 10000]] where i is player_id
pub feature: Array2<f64>,
```

每一个时间步都是 7 维：

```text
[grand_kyoku, honba, kyotaku, s0, s1, s2, s3]
```

其中：

- `grand_kyoku`
  - 表示当前打到哪一局
  - 代码注释写的是：
    - `E1 = 0`
    - `S4 = 7`
    - `W4 = 11`
- `honba`
- `kyotaku`
- `s0 ~ s3`
  - 四位玩家的分数特征
  - 实际是 `score / 10000`

也就是说，`feature.shape = (局数, 7)`。

## 3. 一局游戏的前缀序列具体会长什么样

假设某场游戏一共打了 4 个 `start_kyoku`，那么 `feature` 可能像这样：

```text
t1 = [0, 0, 0, 2.5, 2.5, 2.5, 2.5]   # E1 开始
t2 = [1, 0, 0, 2.8, 2.2, 2.5, 2.5]   # E2 开始
t3 = [2, 1, 0, 3.1, 2.0, 2.4, 2.5]   # E3 开始
t4 = [3, 0, 1, 3.0, 2.1, 2.3, 2.6]   # E4 开始
```

那它会生成 4 个训练样本：

### 样本 1

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5]
]
```

长度是 1，意思是：

- 只看到 E1 开局时的信息
- 去预测最终排名

### 样本 2

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
  [1, 0, 0, 2.8, 2.2, 2.5, 2.5]
]
```

长度是 2，意思是：

- 看到了 E1、E2 两个时点的信息
- 去预测最终排名

### 样本 3

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
  [1, 0, 0, 2.8, 2.2, 2.5, 2.5],
  [2, 1, 0, 3.1, 2.0, 2.4, 2.5]
]
```

### 样本 4

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
  [1, 0, 0, 2.8, 2.2, 2.5, 2.5],
  [2, 1, 0, 3.1, 2.0, 2.4, 2.5],
  [3, 0, 1, 3.0, 2.1, 2.3, 2.6]
]
```

这就是完整前缀，也就是整场游戏到当前为止的全部序列。

而这 4 个样本的标签都是同一个：

```text
rank_by_player
```

例如：

```text
[0, 2, 3, 1]
```

表示：

- 玩家 0 最终第 1
- 玩家 1 最终第 3
- 玩家 2 最终第 4
- 玩家 3 最终第 2

所以同一场游戏会贡献多个“不同观察长度”的样本，但共享同一个最终结果标签。

## 4. 为什么要这样构造前缀序列

这样做的训练目标其实很直观：

- 当对局刚开始时，只知道很少信息
- 当对局进行到中盘时，知道的信息更多
- 当对局接近尾声时，知道的信息最多

作者希望 `GRP` 学会：

> 给定“当前已经发生到这一步”的对局历史，预测最终四家排名。

所以前缀序列长度越长，理论上信息越充分，预测也应该更准。

## 5. `GRU` 的输入为什么会有不同长度

因为不同样本的前缀长度不同。

例如同一个 batch 里可能同时出现：

- 长度 1 的前缀
- 长度 3 的前缀
- 长度 7 的前缀
- 长度 10 的前缀

而 `GRU` 做 batch 计算时，通常要求 batch 内张量有统一的时间维长度，所以这里就需要 padding。

## 6. padding 是怎么做的

关键代码在 [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py) 的 `collate(batch)`：

```python
inputs = []
lengths = []
for inputs_seq, rank_by_player in batch:
    inputs.append(inputs_seq)
    lengths.append(len(inputs_seq))

lengths = torch.tensor(lengths)
...
padded = pad_sequence(inputs, batch_first=True)
packed_inputs = pack_padded_sequence(padded, lengths, batch_first=True, enforce_sorted=False)
```

这里分成两步。

不过在讲这两步之前，先要搞清楚 `collate` 本身在做什么。

## 6.1 `collate` 的逻辑到底是什么

在 PyTorch 里，`DataLoader` 从数据集里一次取出多个样本之后，需要有一个地方把“多个单样本”组装成“一个 batch”。

这个组装函数就叫：

```python
collate_fn
```

在 `train_grp.py` 里，作者自定义了：

```python
def collate(batch):
    ...
```

然后在 `DataLoader(...)` 中传入：

```python
collate_fn = collate
```

也就是说，每当 `DataLoader` 凑够一个 batch 的样本时，就会把这个 batch 交给 `collate(batch)` 来处理。

这里的 `batch` 本质上是一个 Python list，里面每个元素都是：

```python
(inputs_seq, rank_by_player)
```

其中：

- `inputs_seq`
  - 形状是 `(seq_len, 7)`
  - 但不同样本的 `seq_len` 不同
- `rank_by_player`
  - 长度固定为 4

所以 `collate` 的核心任务就是：

1. 把一个 batch 里的所有 `inputs_seq` 收集起来
2. 记录每个序列的真实长度
3. 把这些不同长度的序列整理成 `GRU` 可处理的输入格式
4. 把标签 `rank_by_player` 也拼成一个 batch 张量

具体代码是：

```python
inputs = []
lengths = []
rank_by_players = []
for inputs_seq, rank_by_player in batch:
    inputs.append(inputs_seq)
    lengths.append(len(inputs_seq))
    rank_by_players.append(rank_by_player)
```

这几行只是“收集原材料”：

- `inputs`
  - 保存所有序列张量
- `lengths`
  - 保存每个序列的真实长度
- `rank_by_players`
  - 保存标签

接着：

```python
lengths = torch.tensor(lengths)
rank_by_players = torch.tensor(rank_by_players, dtype=torch.int64, pin_memory=pin_memory)
```

含义是：

- 把 Python list 变成 PyTorch tensor
- 方便后面送到 GPU 或参与计算

因此，你可以把 `collate` 理解成：

> “把一堆长短不一的前缀序列样本，打包成一个标准 batch。”

## 6.2 为什么 `collate` 不能什么都不做

如果没有自定义 `collate`，默认的 PyTorch `collate_fn` 会尝试直接把 batch 里的张量堆叠起来。

但这里每个 `inputs_seq` 的形状不一样，例如：

- 一个样本是 `(1, 7)`
- 一个样本是 `(3, 7)`
- 一个样本是 `(4, 7)`

它们不能直接 `stack` 成一个整齐的三维张量。

所以作者必须自己写 `collate`，把这些不同长度的序列先处理好。

### 第一步：`pad_sequence`

```python
padded = pad_sequence(inputs, batch_first=True)
```

`inputs` 是一个 Python list，里面每个元素形状都是：

```text
(seq_len, 7)
```

但不同元素的 `seq_len` 不一样。

`pad_sequence(..., batch_first=True)` 会把它们补齐成：

```text
(batch_size, max_seq_len, 7)
```

默认 padding 值是：

```text
0
```

也就是说，它会在较短序列的尾部补零。

### 6.1 一个具体例子

假设一个 batch 里有三个样本：

样本 A，长度 1：

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5]
]
```

样本 B，长度 2：

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
  [1, 0, 0, 2.8, 2.2, 2.5, 2.5]
]
```

样本 C，长度 4：

```text
[
  [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
  [1, 0, 0, 2.8, 2.2, 2.5, 2.5],
  [2, 1, 0, 3.1, 2.0, 2.4, 2.5],
  [3, 0, 1, 3.0, 2.1, 2.3, 2.6]
]
```

那么 `pad_sequence` 后会变成：

```text
[
  [
    [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
    [0, 0, 0, 0,   0,   0,   0  ],
    [0, 0, 0, 0,   0,   0,   0  ],
    [0, 0, 0, 0,   0,   0,   0  ],
  ],
  [
    [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
    [1, 0, 0, 2.8, 2.2, 2.5, 2.5],
    [0, 0, 0, 0,   0,   0,   0  ],
    [0, 0, 0, 0,   0,   0,   0  ],
  ],
  [
    [0, 0, 0, 2.5, 2.5, 2.5, 2.5],
    [1, 0, 0, 2.8, 2.2, 2.5, 2.5],
    [2, 1, 0, 3.1, 2.0, 2.4, 2.5],
    [3, 0, 1, 3.0, 2.1, 2.3, 2.6],
  ]
]
```

也就是：

- 短序列在尾部补 0
- 补到和 batch 中最长序列一样长

这里要特别强调：`pad_sequence` 做的事情非常朴素，它只是“补齐形状”，不负责告诉模型哪些位置是真数据，哪些位置是假数据。

如果只做 `pad_sequence` 而不做后面的 `pack_padded_sequence`，那么 `GRU` 就会把这些补零位也当成正常时间步继续算下去。

所以：

- `pad_sequence` 的作用是形状对齐
- 它本身不解决“补零位是否应参与计算”这个问题

## 6.3 `pad_sequence` 的逻辑是什么

你可以把 `pad_sequence(inputs, batch_first=True)` 理解成：

> “找到这个 batch 里最长的序列长度，然后把所有短序列都在末尾补零，补到一样长。”

其中：

- `inputs` 是一个 list
- list 中每个元素形状都是 `(seq_len, 7)`
- 但 `seq_len` 不同

`batch_first=True` 表示返回张量的维度顺序是：

```text
(batch_size, max_seq_len, feature_dim)
```

如果不写 `batch_first=True`，默认会返回：

```text
(max_seq_len, batch_size, feature_dim)
```

而这里 `GRU` 本身在 `model.py` 中也用了：

```python
batch_first=True
```

所以两边保持一致，读起来更直观。

### 6.3.1 补的是哪种 0

这里补的是普通数值 0。

因为每个时间步是 7 维浮点向量，所以补出来的就是：

```text
[0, 0, 0, 0, 0, 0, 0]
```

## 6.4 为什么还要再做 `pack_padded_sequence`

关键点是：

- `pad_sequence` 只负责把张量补齐
- 但补齐后的张量里混入了大量“假时间步”

而在序列建模里，我们通常不希望 RNN 去真正计算这些 padding 部分。

所以作者又马上做了：

```python
packed_inputs = pack_padded_sequence(
    padded,
    lengths,
    batch_first=True,
    enforce_sorted=False,
)
```

这一步的本质是：

> 把“带 padding 的 dense 张量”重新压缩成“只保留真实时间步信息”的紧凑表示。

这个紧凑表示就是 PyTorch 的：

```text
PackedSequence
```

它会同时保存：

- 有效时间步的数据
- 每条序列的长度信息

这样 `GRU` 在处理它时，就知道：

- 哪些时间步是真实存在的
- 哪些只是补出来的占位

## 6.5 `pack_padded_sequence` 的逻辑是什么

如果从概念上解释，它做的是：

1. 输入一个已经补齐到同长度的 batch
2. 再输入每条样本的真实长度 `lengths`
3. 输出一个压缩表示
4. 后续 RNN 只根据真实长度计算，不把 padding 当作真实输入

所以在这个实现里：

- 先 pad 是因为 batch 里的样本原本长短不一，不方便统一处理
- 再 pack 是为了让 RNN 忽略补零部分

你可以把这两步理解成：

```text
pad_sequence:
为了把 batch 摆整齐

pack_padded_sequence:
为了告诉 GRU，摆整齐时补的 0 别当真
```

### 6.5.1 一个具体例子：`PackedSequence` 看起来长什么样

继续沿用前面的例子。假设一个 batch 里有 3 条序列：

- 样本 A，长度 1
- 样本 B，长度 2
- 样本 C，长度 4

它们 padding 之后的 dense 形式是：

```text
[
  [  # A
    a1,
    0,
    0,
    0,
  ],
  [  # B
    b1,
    b2,
    0,
    0,
  ],
  [  # C
    c1,
    c2,
    c3,
    c4,
  ]
]
```

这里为了简单，`a1`、`b1`、`c1` 这些符号分别代表一个 7 维时间步向量。

例如：

- `a1 = [0, 0, 0, 2.5, 2.5, 2.5, 2.5]`
- `b2 = [1, 0, 0, 2.8, 2.2, 2.5, 2.5]`

如果直接把上面这个 padded 张量送给普通 RNN，那么那些 `0` 位置也会被当成真实时间步处理。

而 `pack_padded_sequence(padded, lengths, ...)` 做完以后，概念上会把它压成：

```text
time step 1: a1, b1, c1
time step 2:     b2, c2
time step 3:         c3
time step 4:         c4
```

也就是说，它只保留“每个时间步上真实存在的元素”，补零位被丢掉了。

换成一条线性的直观写法，就是：

```text
packed.data = [a1, b1, c1, b2, c2, c3, c4]
```

同时它还会记录每个时间步还剩多少条有效序列。对于这个例子，可以理解成：

```text
batch_sizes = [3, 2, 1, 1]
```

含义是：

- 第 1 个时间步，有 3 条序列还有效
- 第 2 个时间步，有 2 条序列还有效
- 第 3 个时间步，有 1 条序列还有效
- 第 4 个时间步，有 1 条序列还有效

这正好对应：

- A 只有 1 步
- B 有 2 步
- C 有 4 步

所以 `PackedSequence` 不是一个普通的 `(batch_size, seq_len, feature_dim)` 张量，而更像是：

1. 一条把所有有效时间步拼起来的紧凑数据流
2. 再附带一个“每一层时间步上还剩多少条序列有效”的说明书

这样 `GRU` 在内部遍历时，就能知道：

- 第 1 步要处理 3 条样本
- 第 2 步只处理 2 条样本
- 第 3 步开始只剩 1 条样本

于是它完全不需要去看那些补零位置。

### 6.5.2 为什么这样比“直接喂补零张量”更合理

因为如果不 pack：

- A 在第 2、3、4 步会继续收到全零输入
- B 在第 3、4 步也会继续收到全零输入

这虽然有时也能训练，但：

- 会做很多无意义计算
- 还可能让 RNN 的隐藏状态被这些假时间步继续传播

而 pack 之后：

- 短序列在结束后就真的结束了
- 后面的时间步只处理还活着的长序列

所以它更符合“变长序列”的真实语义。

## 7. 但是 GRU 实际会不会把这些 0 当真数据算进去

不会直接这么算，因为紧接着又做了第二步：

```python
packed_inputs = pack_padded_sequence(
    padded,
    lengths,
    batch_first=True,
    enforce_sorted=False,
)
```

这一步把 padded 张量重新打包成 `PackedSequence`。

意义是：

- 告诉 `GRU` 每条序列真实长度是多少
- 让 `GRU` 忽略后面那些补零部分

所以在这个实现里：

- padding 只是为了把 batch 对齐
- 真正送进 `GRU` 前又通过 `pack_padded_sequence` 把无效 padding 去掉了

因此 `GRU` 实际处理的是：

- 每条样本的真实前缀长度

而不是把所有补零位都当成真实时间步。

## 7.1 用一句更直白的话解释这套流程

如果只从工程角度理解，这套输入处理流程可以概括成：

1. `collate`
   - 把一个 batch 的样本收集起来
2. `pad_sequence`
   - 先把长短不一的序列补到一样长，方便组成一个 batch
3. `pack_padded_sequence`
   - 再把真实长度信息交给 `GRU`，避免它把补零部分也算进去

所以这里不是“padding 之后直接喂 GRU”，而是：

```text
变整齐 -> 再标记哪些是假的 -> 再喂给 GRU
```

## 8. 为什么设置 `enforce_sorted=False`

`pack_padded_sequence` 默认希望输入样本按长度降序排好。

但这里代码没有手工排序，而是直接传：

```python
enforce_sorted=False
```

这表示：

- batch 内样本不需要提前按长度排序
- PyTorch 会接受当前顺序并内部处理

所以 `collate()` 逻辑更简单：

- 收集样本
- 记录长度
- pad
- pack

不需要额外写一个“按长度排序再还原顺序”的步骤。

## 9. `GRU` 最终实际吃到的输入形状是什么

从 [mortal/model.py](/home/tml104/Mortal/mortal/model.py) 看：

```python
self.rnn = nn.GRU(
    input_size=GRP_SIZE,
    hidden_size=hidden_size,
    num_layers=num_layers,
    batch_first=True,
)
```

而 `GRP_SIZE = 7`。

所以每个时间步输入维度固定是：

```text
7
```

也就是：

```text
[grand_kyoku, honba, kyotaku, s0, s1, s2, s3]
```

batch 层面上，送进 `GRU` 前在概念上是：

```text
(batch_size, seq_len, 7)
```

但在代码里实际传入的是 `PackedSequence`，不是普通 dense tensor。

## 10. 一句话总结

如果只记一句话，可以记成：

> `train_grp.py` 会把一场游戏的 `feature` 序列切成所有可能的前缀 `feature[:1], feature[:2], ..., feature[:T]` 作为样本；batch 内不同长度的前缀先用 `pad_sequence` 在尾部补零，再用 `pack_padded_sequence` 告诉 `GRU` 忽略这些补零部分。
