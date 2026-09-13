# `dataloader.py` 解读

[`mortal/dataloader.py`](../mortal/dataloader.py) 负责把 mjai `.json.gz` 牌谱
转换为主策略训练所需的逐动作样本。它连接 Rust 数据提取和 Python 训练循环：

```text
牌谱文件
  -> GameplayLoader 解析事件并按玩家生成 Gameplay
  -> FileDatasetsIter 计算局奖励、顺位标签和折扣步数
  -> buffer 局部混洗
  -> DataLoader 组成 batch
  -> train.py 训练 Brain、DQN 和 AuxNet
```

相关实现：

- [`gameplay.rs`](../libriichi/src/dataset/gameplay.rs)：玩家视角、动作标签和 `apply_gamma`；
- [`grp.rs`](../libriichi/src/dataset/grp.rs)：按局特征和终局结果；
- [`reward_calculator.py`](../mortal/reward_calculator.py)：局奖励计算；
- [`train.py`](../mortal/train.py)：组批和训练目标；
- [`rewards.md`](rewards.md)：奖励公式的进一步说明。

## PyTorch 数据加载基础

PyTorch 的数据加载通常由 `Dataset` 和 `DataLoader` 配合完成。`Dataset` 定义单条
样本的来源与读取方式，`DataLoader` 负责按顺序取样、组成 batch、调用多进程 worker
以及执行自定义拼接逻辑。

```text
Dataset
  -> Sampler 决定样本顺序（索引式数据集）
  -> DataLoader 读取并拼接样本
  -> 训练循环逐个取得 batch
```

### `Dataset` 与 `DataLoader`

能够通过索引访问的数据通常使用 `Dataset`。这类数据集至少实现 `__len__()` 和
`__getitem__()`：

```python
from torch.utils.data import Dataset


class SampleDataset(Dataset):
    def __init__(self, features, labels):
        self.features = features
        self.labels = labels

    def __len__(self):
        return len(self.features)

    def __getitem__(self, index):
        return self.features[index], self.labels[index]
```

`DataLoader` 将这些单条样本组合为训练循环需要的 batch：

```python
import torch
from torch.utils.data import DataLoader, TensorDataset

features = torch.randn(1000, 20)
labels = torch.randint(0, 10, (1000,))
dataset = TensorDataset(features, labels)

loader = DataLoader(
    dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0,
)

for batch_features, batch_labels in loader:
    print(batch_features.shape)  # (32, 20)，最后一批可能更小
    print(batch_labels.shape)    # (32,)
```

DataLoader 通常按需调用数据集，不会预先把整个数据集读入内存。完整遍历一次
DataLoader 通常对应一个 epoch。

常用参数如下：

| 参数 | 作用 | 常见设置 |
| --- | --- | --- |
| `batch_size` | 每个 batch 的样本数 | 根据显存、吞吐量和训练稳定性确定 |
| `shuffle` | 每个 epoch 是否重新打乱索引 | 训练集常用 `True`，验证集常用 `False` |
| `num_workers` | 用于读取数据的子进程数 | `0` 便于调试，增加后需实测性能和内存 |
| `drop_last` | 是否丢弃不足 `batch_size` 的最后一批 | BatchNorm 或固定形状场景可设为 `True` |
| `pin_memory` | 是否将 CPU Tensor 放入锁页内存 | CUDA 训练时常与异步传输配合 |
| `collate_fn` | 多条样本组成 batch 的规则 | 变长序列或复杂样本需要自定义 |

CUDA 训练中可配合锁页内存和异步复制：

```python
loader = DataLoader(dataset, batch_size=32, pin_memory=True)

for features, labels in loader:
    features = features.to("cuda", non_blocking=True)
    labels = labels.to("cuda", non_blocking=True)
```

Windows 使用 `num_workers > 0` 时，创建 DataLoader 和启动训练的入口应放在
`if __name__ == "__main__":` 保护中，避免子进程重复执行主程序。

### 使用 `pad_sequence` 拼接变长序列

默认 `collate_fn` 只能直接堆叠形状一致的 Tensor。文本、音频等数据的序列长度经常
不同，此时可在自定义 `collate_fn` 中使用 `pad_sequence`，将同一 batch 内的序列
补齐到当前 batch 的最大长度。

例如，三个序列分别为：

```text
[1, 2, 3]
[4, 5]
[6]
```

执行以下代码：

```python
import torch
from torch.nn.utils.rnn import pad_sequence

sequences = (
    torch.tensor([1, 2, 3]),
    torch.tensor([4, 5]),
    torch.tensor([6]),
)

padded = pad_sequence(
    sequences,
    batch_first=True,
    padding_value=0,
)
```

输出为：

```text
tensor([
    [1, 2, 3],
    [4, 5, 0],
    [6, 0, 0],
])
```

`batch_first=True` 使输出形状为 `(batch_size, max_length, ...)`；
`padding_value=0` 指定补齐值。补齐值只是占位符，模型不会自动忽略它。训练文本模型
时通常还要提供原始长度或 padding mask，避免补齐位置参与注意力与损失计算。

一个完整的拼接函数可以写成：

```python
def collate_batch(samples):
    sequences, labels = zip(*samples)
    lengths = torch.tensor([len(sequence) for sequence in sequences])
    padded = pad_sequence(sequences, batch_first=True, padding_value=0)
    labels = torch.tensor(labels)
    return padded, lengths, labels


loader = DataLoader(
    dataset,
    batch_size=32,
    collate_fn=collate_batch,
)
```

### `IterableDataset`

`IterableDataset` 通过迭代器连续产生样本，核心接口是 `__iter__()`。它适用于逐行
读取大型文件、数据库游标、网络流、动态生成数据以及无限数据流等缺少随机索引能力的
数据源。

```python
from torch.utils.data import DataLoader, IterableDataset


class NumberDataset(IterableDataset):
    def __iter__(self):
        yield from range(10)


loader = DataLoader(NumberDataset(), batch_size=3)

for batch in loader:
    print(batch)
```

输出为：

```text
tensor([0, 1, 2])
tensor([3, 4, 5])
tensor([6, 7, 8])
tensor([9])
```

流式读取文件时，每次只需保留当前样本及必要的缓冲区：

```python
class TextDataset(IterableDataset):
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, encoding="utf-8") as file:
            for line in file:
                yield line.strip()
```

无限数据流需要由训练循环主动限制步数：

```python
class RandomDataset(IterableDataset):
    def __iter__(self):
        while True:
            yield torch.rand(())


for step, batch in enumerate(DataLoader(RandomDataset(), batch_size=32)):
    if step >= 100:
        break
    # 使用 batch 训练模型
```

`IterableDataset` 有以下使用约束：

- DataLoader 无法通过索引重排数据，因此不能设置 `shuffle=True`；需要随机化时，可在
  数据集内部采用有限缓冲区混洗。
- 数据流可能没有确定长度。只有数据集提供可靠的 `__len__()` 时，`len(loader)` 才有
  参考意义；多 worker 分片和 `drop_last` 仍可能使它成为估算值。
- `num_workers > 0` 时，每个 worker 持有一份数据集副本。各副本若读取同一个数据源，
  将产生重复样本。

可在 `__iter__()` 中通过 `get_worker_info()` 分配互不重叠的数据范围，也可以通过
DataLoader 的 `worker_init_fn` 修改每个 worker 的数据集副本。Mortal 采用后一种方式
划分 `file_list`，具体逻辑见后文“`worker_init_fn` 的作用”。

## `FileDatasetsIter` 做什么

`FileDatasetsIter` 继承 `IterableDataset`。它不把全部牌谱一次性放进内存，也不按
索引读取样本，而是按文件批次解析并逐条 `yield`。

构造参数如下：

| 参数 | 含义 |
| --- | --- |
| `version` | Rust 观测编码版本，决定 `obs` 形状 |
| `file_list` | `.json.gz` 路径列表；迭代时会原地打乱 |
| `pts` | 一至四名对应的效用值，用于计算局奖励 |
| `oracle` | 是否额外生成不可见信息观测 |
| `file_batch_size` | 一次交给 Rust 解析的文件数，与训练 batch size 含义不同 |
| `reserve_ratio` | 每批新增样本中留到下一文件批次混洗的比例 |
| `player_names` | 非空时只提取指定玩家 |
| `excludes` | `player_names` 为空时排除指定玩家 |
| `num_epochs` | 单次迭代内部重复遍历文件的次数 |
| `enable_augmentation` | 是否再遍历一遍相反增强状态的数据 |
| `augmented_first` | 第一遍是否交换万子和筒子 |

`player_names` 优先于 `excludes`。两者都为空时，一份四人牌谱会为四名玩家各生成
一个视角。

### 迭代生命周期

`__iter__()` 会缓存第一次创建的 `build_iter()` 生成器。生成器耗尽后，再次调用
`iter(dataset)` 不会从头开始；需要重复数据应使用 `num_epochs` 或创建新实例。

GRP 在 `build_iter()` 第一次真正执行时才加载，构造函数阶段不加载。这样
DataLoader 可以先复制数据集，再由每个 worker 在 CPU 上各加载一份 GRP，兼容
Windows `spawn`。GRP 只用于奖励推理，不参与主策略反向传播。

每个内部 epoch 的读取方式为：

| `enable_augmentation` | `augmented_first` | 读取顺序 |
| --- | --- | --- |
| `False` | `False` | 仅原始 |
| `False` | `True` | 仅万筒互换 |
| `True` | `False` | 原始，然后万筒互换 |
| `True` | `True` | 万筒互换，然后原始 |

因此总遍历次数是：

```text
num_epochs * (2 if enable_augmentation else 1)
```

## 从 `Gameplay` 生成 `entry`

`GameplayLoader.load_gz_log_files()` 返回：

```text
list[                 # 每个元素对应一个输入文件
    list[Gameplay]    # 文件中被选中的玩家视角
]
```

一个 `Gameplay` 包含两类数据。

逐动作数组长度相同，第 `i` 个位置共同描述一次决策：

| 字段 | 含义 |
| --- | --- |
| `obs[i]` | 当前公开观测 |
| `invisible_obs[i]` | Oracle 模式下的不可见信息 |
| `actions[i]` | 牌谱实际动作，取值 `0..45` |
| `masks[i]` | 长度为 46 的合法动作掩码 |
| `at_kyoku[i]` | 动作所属的零基局索引 |
| `dones[i]` | 是否为该玩家在本局的最后一条决策样本 |
| `apply_gamma[i]` | 是否增加一个折扣步 |

整场数据只有一份：

```python
grp = game.take_grp()
player_id = game.take_player_id()
```

Rust 的 `take_*` 会把内部数组的所有权移交给 Python，同一字段不能再次读取复用。

### `grp` 与 `player_id`

这里有两个名称相近但作用不同的对象：

| 名称 | 类型 | 作用 |
| --- | --- | --- |
| 小写 `grp` | Rust `libriichi.dataset.Grp` | 保存按局特征和真实终局结果 |
| 大写 `GRP` | Python 顺位预测模型 | 根据按局特征预测最终顺位概率 |

Rust `Grp` 包含：

| 字段 | 内容 |
| --- | --- |
| `feature` | `(K, 7)` 的局开始特征序列 |
| `rank_by_player` | 四名玩家的真实终局零基顺位 |
| `final_scores` | 四名玩家的真实终局点数 |

每行 `feature` 为：

```text
[grand_kyoku, honba, kyotaku,
 score_0 / 10000, score_1 / 10000, score_2 / 10000, score_3 / 10000]
```

`player_id` 是玩家在 `start_game.names` 中的固定位置，取值为 `0..3`。它在整场
比赛中不变，其含义与当前顺位、庄家、自风或局编号无关。

同一份牌谱生成的四个 `Gameplay` 共享相同的全局 `grp` 内容，但 `player_id` 和
逐动作数据不同。`player_id` 用于从四家数据中选出当前视角玩家：

```python
matrix[:, player_id]             # 当前玩家的 GRP 顺位概率
rank_by_player[player_id]        # 当前玩家的真实终局顺位
rank_by_player_seq[:, player_id] # 当前玩家随牌局变化的顺位
```

### 局奖励 `kyoku_reward`

假设一场牌谱有 `K` 个 `start_kyoku`，GRP 会分别读取：

```text
feature[:1], feature[:2], ..., feature[:K]
```

它为四名玩家预测每个时刻的最终顺位概率。`RewardCalculator` 取出当前
`player_id`，在末尾追加真实终局顺位，并用 `pts` 计算期望效用：

```text
E_t = rank_probability_t dot pts
kyoku_reward[k] = E_(k+1) - E_k
```

第 `i` 条动作使用：

```python
kyoku_rewards[at_kyoku[i]]
```

所以同一局的动作共享同一个未折扣奖励。该值表示整局带来的效用变化，不表示某个
动作造成的即时点棒变化。

## `player_rank` 如何计算

`player_rank` 是 `AuxNet` 的四分类标签，取值 `0..3`，分别表示一至四名。它直接根据
各局开始分数和终局分数计算真实顺位，GRP 的预测结果不参与这项计算。

第一步，从 `grp_feature` 取出后四列。这些分数曾除以 10000，因此先恢复为点棒，
再追加终局分数：

```python
scores_seq = np.concatenate((
    grp_feature[:, 3:] * 1e4,
    [final_scores],
))
```

若有 `K` 局，`scores_seq.shape == (K + 1, 4)`：前 `K` 行是各局开始分数，最后
一行是真实终局分数。

第二步，对每一行执行两次稳定排序：

```python
rank_by_player_seq = (
    (-scores_seq)
    .argsort(-1, kind='stable')
    .argsort(-1, kind='stable')
)
```

第一次 `argsort` 在负分数上升序排列，等价于按原分数从高到低得到“每个顺位是谁”；
第二次 `argsort` 把它反转成“每名玩家排第几”。

例如某个时刻的分数是：

```text
scores                    = [25000, 31000, 18000, 26000]
(-scores).argsort()       = [1, 3, 0, 2]
再次 argsort 后的顺位     = [2, 0, 3, 1]
```

含义是玩家 0/1/2/3 分别为第三、第一、第四、第二名。顺位从 0 开始，所以结果是
`[2, 0, 3, 1]`。使用稳定排序时，同分玩家按较小的 `player_id` 排在前面。

第三步，取出当前视角玩家的整条顺位序列：

```python
player_ranks = rank_by_player_seq[:, player_id]
```

`player_ranks[k]` 对应第 `k` 局开始时的顺位。第 `k` 局内的动作需要预测该局结束后
的顺位，因此最终标签使用下一行：

```python
player_rank = player_ranks[at_kyoku[i] + 1]
```

对非末局来说，下一行就是下一局开始分数对应的顺位；对末局来说，下一行是终局分数
对应的顺位。这也是代码选择 `at_kyoku + 1` 的原因。

## `apply_gamma` 与 `steps_to_done`

四个相关量的关系是：

| 名称 | 含义 |
| --- | --- |
| `apply_gamma[i]` | 当前样本向后推进时是否增加一步 |
| `dones[i]` | 当前样本是否为本局最后一条决策 |
| `steps_to_done[i]` | 当前样本到本局结束的有效步数 |
| `gamma` | 每个有效步的折扣系数 |

Rust 当前按动作编号设置 `apply_gamma`：

```rust
self.apply_gamma.push(label <= 37);
```

| 动作编号 | 动作 | `apply_gamma` |
| --- | --- | --- |
| `0..36` | 打牌或杠牌种选择 | `True` |
| `37` | 立直 | `True` |
| `38..40` | 吃 | `False` |
| `41` | 碰 | `False` |
| `42` | 决定杠 | `False` |
| `43` | 和牌 | `False` |
| `44` | 主动流局 | `False` |
| `45` | 跳过 | `False` |

源码旁的旧英文注释没有提到立直，但实际条件 `label <= 37` 包含动作 37，应以代码为准。

`dones` 在局内最后一条样本处为 `True`，阻止步数跨局累计。Python 从后向前计算：

```python
steps_to_done = np.zeros(game_size, dtype=np.int64)
for i in reversed(range(game_size)):
    if not dones[i]:
        steps_to_done[i] = steps_to_done[i + 1] + int(apply_gamma[i])
```

例如：

| 动作 | `apply_gamma` | `done` | `steps_to_done` |
| --- | --- | --- | ---: |
| 打牌 | `True` | `False` | 2 |
| 跳过 | `False` | `False` | 1 |
| 打牌 | `True` | `False` | 1 |
| 和牌 | `False` | `True` | 0 |

训练目标为：

```text
q_target = gamma ** steps_to_done * kyoku_reward
```

因此越早的有效动作通常折扣越多；`gamma=1` 时步数仍会计算，但不改变目标。

## `buffer` 会不会一直增长

不会。每次 `load_files()` 都从空 `buffer` 开始；每处理一个文件批次，先追加新样本，
再打乱并产出大部分样本，只留下 `reserve_ratio` 指定的数量。

设上一批留下 `B` 条、当前新增 `N` 条、`r=reserve_ratio`：

```text
解析后大小 = B + N
本轮保留数 = int(N * r)
```

例如每批新增 1000 条、`r=0.1`：

```text
0 -> 1000 -> 保留 100
100 -> 1100 -> 保留 100
100 -> 1100 -> 保留 100
最后把 100 条全部产出并清空
```

所以在合法的 `0..1` 范围内，缓冲区呈周期性增长和缩小，不会随着整个
`file_list` 无限增长。

代码使用：

```python
yield from self.buffer[reserved_size:]
del self.buffer[reserved_size:]
```

切片会临时创建一个只复制 `entry` 引用的新列表。产出过程中原 `buffer` 不会逐条
缩短；全部产出后，`del` 才一次性删除相应部分。单个 worker 会等当前切片产完再解析
下一文件批次，多个 worker 则各自拥有独立的缓冲区。

## `worker_init_fn` 的作用

`IterableDataset` 不会自动替多个 worker 划分数据。若开四个 worker 而不处理，四个
worker 都会遍历完整 `file_list`，使每份牌谱重复四次。

DataLoader 创建 worker 时，每个 worker 会得到原数据集的独立副本。切分前，这些
各副本的 `file_list` 初始内容相同，且分别属于独立的 Python 对象。`worker_init_fn`
随后只修改当前 worker 副本中的列表：

```python
per_worker = ceil(len(file_list) / num_workers)
start = worker_id * per_worker
end = start + per_worker
dataset.file_list = dataset.file_list[start:end]
```

例如 10 个文件、4 个 worker：

| worker | 切片 | 文件数 |
| ---: | --- | ---: |
| 0 | `[0:3]` | 3 |
| 1 | `[3:6]` | 3 |
| 2 | `[6:9]` | 3 |
| 3 | `[9:12]` | 1 |

向上取整保证余数文件不会遗漏；切片终点超过列表长度也是安全的。若文件数少于
worker 数，部分 worker 会得到空列表。

`train.py` 会在多 worker 切分前先打乱完整列表，避免连续路径集中到同一 worker；
切分后，每个 worker 又会在每轮打乱自己的子列表。

不同 `num_workers` 的行为：

| 数量 | 行为 |
| ---: | --- |
| `0` | 不创建 worker，回调不执行，主进程处理全部文件 |
| `1` | 回调执行，唯一 worker 仍取得全部文件 |
| `>1` | 各 worker 取得互不重叠的文件子集 |

这种方法只均分文件数，不保证样本数和耗时相等；不同牌谱长度、玩家过滤结果和动作数
都可能造成负载差异。每个 worker 还会单独加载 GRP，因此 worker 越多，CPU 内存占用
也越高。

## 最终样本

普通模式输出：

```text
[obs, action, mask, steps_to_done, kyoku_reward, player_rank]
```

Oracle 模式输出：

```text
[obs, invisible_obs, action, mask,
 steps_to_done, kyoku_reward, player_rank]
```

当前 `train.py` 使用普通六字段样本。DataLoader 将连续 `entry` 合并后，`obs` 形状为
`(B, *obs_shape(version))`，`mask` 为 `(B, 46)`，其余字段为 `(B,)`。
