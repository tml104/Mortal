# Mortal 网络结构

模型定义集中在 [`mortal/model.py`](../mortal/model.py)。如果不熟悉张量、全连接层、softmax、Q 值、卷积或残差网络，先读 [深度学习基础概念与用到的网络介绍](deep_learning_basics.md)。

先记住各模块的边界：

| 模块 | 定义位置 | 输入 | 输出 | 是否参与普通决策 |
| --- | --- | --- | --- | --- |
| `Brain` | [`mortal/model.py`](../mortal/model.py) | 当前状态观测 | 状态表示 | 是 |
| `DQN` | [`mortal/model.py`](../mortal/model.py) | 状态表示、合法动作 mask | 46 个动作的 Q 值 | 是 |
| `AuxNet` | [`mortal/model.py`](../mortal/model.py) | `Brain` 的状态表示 | 下一局面顺位分类 logits | 否，仅训练 |
| `GRP` | [`mortal/model.py`](../mortal/model.py) | 按局推进的 7 维序列 | 最终顺位排列 logits | 否，用于奖励和复盘 |
| `GRP_Transformer` | [`mortal/model.py`](../mortal/model.py) | 同上 | 最终顺位排列 logits | 否，实验实现 |
| `MortalEngine` | [`mortal/engine.py`](../mortal/engine.py) | `obs`、mask 和模型 | 动作编号、Q 值等推理结果 | 是，负责调用和选择 |

需要模型判断时，普通决策只有一条主线：

流程中的 `PlayerState` 定义在 [`libriichi/src/state/player_state.rs`](../libriichi/src/state/player_state.rs)，负责保存随 mjai 事件更新的牌局状态；它先生成模型能读取的数字输入，模型再进行评价和选择。

```text
mjai 事件 -> PlayerState -> obs + mask -> Brain -> DQN
          -> 46 个动作的 Q 值 -> MortalEngine -> 动作编号 -> mjai 动作
```

这里的 `obs` 是 observation（观测）的缩写，即用数字描述的当前牌局；`mask` 是一个长度为 46 的 bool（真假值）数组，标出此刻哪些动作符合规则。两者由同一次 [`encode_obs()`](../libriichi/src/state/obs_repr.rs) 调用一起生成。

完整过程如下：

1. [`mortal/mortal.py`](../mortal/mortal.py) 逐行读取 mjai JSON 事件，并交给 [`libriichi/src/mjai/bot.rs`](../libriichi/src/mjai/bot.rs) 定义的 `Bot`。mjai 是本项目表示摸牌、打牌、鸣牌等对局事件的协议，详见 [mjai.md](mjai.md)。
2. `Bot` 解析事件，并调用 [`libriichi/src/state/update.rs`](../libriichi/src/state/update.rs) 实现的 `PlayerState.update()`。`PlayerState` 定义在 [`libriichi/src/state/player_state.rs`](../libriichi/src/state/player_state.rs)，表示当前玩家视角的牌局状态；每收到一个事件，它就更新手牌、牌河、分数和可执行动作等信息。
3. 轮到该玩家响应时，[`libriichi/src/agent/mortal.rs`](../libriichi/src/agent/mortal.rs) 中的 `MortalBatchAgent.set_scene()` 调用 [`PlayerState.encode_obs()`](../libriichi/src/state/obs_repr.rs)，把同一个 `PlayerState` 转成 `obs` 和 `mask`。这就是麻将状态被编码为 `(channels, 34)` 数字表的具体位置。
4. [`mortal/engine.py`](../mortal/engine.py) 中的 `MortalEngine.react_batch()` 把一个或多个局面的 `obs` 和 `mask` 沿新增的批次维堆叠成 PyTorch 张量，详见下文“批量堆叠”。
5. [`mortal/model.py`](../mortal/model.py) 中的 `Brain` 通过卷积、残差块和全连接层，把每个观测转换成 1024 维状态向量 `phi`；这是一种学习得到的特征变换，详见下文“`Brain`：状态编码”。
6. 同一文件中的 `DQN` 接收 `phi` 和 `mask`，计算 46 个动作的 Q 值，并把非法动作设为负无穷。Q 值是动作的长期价值评分，不是概率，详见 [Q 值不是概率](deep_learning_basics.md#q-值不是概率)。
7. `MortalEngine` 按配置选择最高 Q 值动作或随机采样。它是模型和麻将程序之间的推理控制器，负责数据转换、模型调用和选择策略，但不维护麻将规则或牌局状态。
8. [`MortalBatchAgent.get_reaction()`](../libriichi/src/agent/mortal.rs) 将动作编号解码为 `dahai`、`reach`、`chi`、`hora` 等 mjai 事件，再由 [`mortal/mortal.py`](../mortal/mortal.py) 输出 JSON。

若当前只有唯一可打的牌且没有立直、和牌或杠等特殊选择，默认启用的 `quick_eval` 会直接返回该动作，不调用网络。这只是避免无意义计算的优化，不改变存在多个选择时的主线。

## 输入和动作契约

### 观测

[`PlayerState`](../libriichi/src/state/player_state.rs) 本身是随 mjai 事件不断更新的麻将状态对象。只有当 [`Bot`](../libriichi/src/mjai/bot.rs) 判断当前事件需要玩家响应时，[`MortalBatchAgent`](../libriichi/src/agent/mortal.rs) 才调用 [`PlayerState.encode_obs()`](../libriichi/src/state/obs_repr.rs)，一次生成：

- `obs`：形状为 `(channels, 34)` 的浮点数表，交给神经网络理解局面；
- `mask`：长度为 46 的真假值表，交给 `DQN` 排除非法动作。

34 是基本牌种轴：万、筒、索各 1 到 9，再加东南西北白发中，共 34 种。赤五万仍归在五万的位置，是否为赤牌用额外通道表示。`channels` 是从不同角度描述牌局的特征通道；每个通道都有 34 个位置。

有些通道逐牌填写，例如“手里有几张一万”；有些是整个局面的标量，例如“剩余牌数”。标量会重复写满该通道的 34 个位置，使所有输入仍保持相同形状。所有值在编码时都限制在 0 到 1 之间。

| 版本 | 普通观测 | Oracle 额外观测 |
| --- | ---: | ---: |
| v1 | `(938, 34)` | `(211, 34)` |
| v2 | `(942, 34)` | `(217, 34)` |
| v3 | `(934, 34)` | `(217, 34)` |
| v4 | `(1012, 34)` | `(217, 34)` |

维度由 [`libriichi/src/consts.rs`](../libriichi/src/consts.rs) 定义。批量前向时再在最前面增加 `batch_size` 维。

### `MortalEngine` 如何批量堆叠观测

[`MortalBatchAgent.set_scene()`](../libriichi/src/agent/mortal.rs) 会把待计算的观测和 mask 分别加入 `states`、`masks` 列表；同一文件中的 `MortalBatchAgent.evaluate()` 再把每个数组转换为 Python 可接收的 NumPy 数组，并调用 [`MortalEngine.react_batch()`](../mortal/engine.py)。其内部的 `_react_batch()` 因而接收到两个一一对应的数组列表。

以两个 v4 局面 A、B 为例，每个局面各有一个 `(1012, 34)` 的 `obs` 和一个 `(46,)` 的 `mask`：

```text
obs 列表   = [obs_A, obs_B]       每项形状都是 (1012, 34)
mask 列表  = [mask_A, mask_B]     每项形状都是 (46,)
```

代码分别执行 `np.stack(obs, axis=0)` 和 `np.stack(masks, axis=0)`。`stack` 不会把两个局面相加、求平均或首尾拼接；它只在最前面新增一层，用这一层的编号区分不同局面：

```text
obs_batch[0] = obs_A              obs_batch 形状为 (2, 1012, 34)
obs_batch[1] = obs_B

mask_batch[0] = mask_A            mask_batch 形状为 (2, 46)
mask_batch[1] = mask_B
```

用较小的数字表可以直观看出这一点：

```text
obs_A = [[1, 0, 0],               # 形状 (2, 3)
         [0, 1, 0]]
obs_B = [[0, 0, 1],
         [1, 1, 0]]

np.stack([obs_A, obs_B], axis=0)
= [[[1, 0, 0],                     # 第 0 项仍是完整的 obs_A
    [0, 1, 0]],
   [[0, 0, 1],                     # 第 1 项仍是完整的 obs_B
    [1, 1, 0]]]                    # 形状 (2, 2, 3)

mask_A = [True,  False, True]
mask_B = [False, True,  True]
stack 后 = [[True,  False, True],  # 与 obs_A 对应
            [False, True,  True]]  # 与 obs_B 对应
```

随后 `torch.as_tensor(..., device=self.device)` 在不改变排列顺序的前提下把数组转成 PyTorch 张量，并放到配置指定的 CPU 或 GPU。此时最前面的下标仍然表示局面编号：`obs_batch[0]` 是 A，`obs_batch[1]` 是 B。形状中常写的字母 `B` 是 batch size（批大小），不是这里作为局面名称的 B；本例的 batch size 等于 2。

`Brain` 接收整个 `obs_batch` 后，输出形状为 `(2, 1024)` 的 `phi_batch`：

```text
phi_batch = Brain(obs_batch)

phi_batch[0] = obs_A 经过 Brain 后得到的 1024 个状态特征
phi_batch[1] = obs_B 经过 Brain 后得到的 1024 个状态特征
```

这里的“一次处理”是指 PyTorch 用同一组模型参数并行计算两份输入，并不是先把 A、B 合并成一个麻将局面。卷积层和全连接层都保留最前面的批次维，不会沿这个维度把两个局面相加。普通推理时 `Brain` 处于 `.eval()` 模式，BatchNorm 也使用训练期间保存的累计统计量，不会根据 A、B 重新计算一组批次统计量。

接下来 `DQN` 同时接收 `phi_batch` 和 `mask_batch`，得到 `(2, 46)` 的 `q_out`。DQN 的所有计算都按最前面的下标逐行对应：

```text
DQN(phi_batch[0], mask_batch[0]) -> q_out[0]
DQN(phi_batch[1], mask_batch[1]) -> q_out[1]
```

因此，`q_out[0]` 是局面 A 的 46 个动作 Q 值，并且只由 `mask_A` 屏蔽非法动作；`q_out[1]` 是局面 B 的 46 个动作 Q 值，并且只由 `mask_B` 屏蔽非法动作。A 的 mask 不会作用到 B，两个局面的 Q 值也不会放在一起比较。

例如 A 只有动作 0 和 43 合法，B 只有动作 5 和 45 合法，屏蔽后的关键位置可能是：

```text
q_out[0][0]  = 1.7                 # A 的动作 0
q_out[0][43] = 2.3                 # A 的动作 43
q_out[0] 的其余位置 = -inf

q_out[1][5]  = 0.8                 # B 的动作 5
q_out[1][45] = 1.2                 # B 的动作 45
q_out[1] 的其余位置 = -inf
```

最后，`q_out.argmax(-1)` 中的 `-1` 表示分别在每一行最后一维的 46 个动作里寻找最大值，而不是在整个 `(2, 46)` 数字表中只找一个最大值：

```text
actions[0] = 43                    # 只比较 q_out[0]，得到 A 的动作
actions[1] = 45                    # 只比较 q_out[1]，得到 B 的动作
```

所以从数据对应关系看，批量处理等价于分别计算 A、B 后再把结果放在一起；它的主要目的，是让 CPU 或 GPU 用较少的调用开销并行完成多个局面的计算。

### v4 的 1012 个观测通道

需要先澄清两个容易混淆的数字：**v4 输入有 1012 个通道；1024 是 `Brain` 输出的状态向量长度**。因此 v4 的单个观测是 `(1012, 34)`，经过网络后才变成 1024 个数。

下表按 [`obs_repr.rs`](../libriichi/src/state/obs_repr.rs) 中的实际写入顺序列出全部 1012 个通道。编号从 0 开始，范围两端都包含在内。

| 通道 | 数量 | 编码内容 |
| --- | ---: | --- |
| `0..3` | 4 | 手牌张数。第 1 至第 4 张同种牌分别占一层；例如有三张一万，则一万位置在前三层为 1。 |
| `4..6` | 3 | 手牌中的赤五万、赤五筒、赤五索；相应赤牌存在时整层为 1。 |
| `7..14` | 8 | 四家分数，每家两层：分别按 100000 点和 30000 点截断并缩放到 0..1。玩家顺序相对于当前观察者排列。 |
| `15..18` | 4 | 自己当前第 1 至第 4 名的 one-hot（独热）表示；只有对应名次的一层为 1。 |
| `19..22` | 4 | 当前场内第几局的 one-hot 表示。 |
| `23..24` | 2 | 本场数 `honba` 和场上立直棒数 `kyotaku`，分别除以 10，超过 10 时截断。 |
| `25..26` | 2 | 场风和自风；只在该风牌对应的 34 轴位置写 1。 |
| `27` | 1 | 从东一局到南四局的全局局序，缩放到 0..1。 |
| `28..34` | 7 | 宝牌指示牌：4 层表示同种牌出现 1 至 4 张，另 3 层表示三种赤五。 |
| `35..58` | 24 | 自己最早 6 次舍牌，每次 4 层：该次杠牌、舍牌牌种、是否赤牌、是否为宝牌。 |
| `59..130` | 72 | 自己最近 18 次舍牌，仍是每次 4 层。早期不足或与前 6 次重叠的部分保留固定槽位。 |
| `131` | 1 | 自己各牌种最近一次出现在牌河中的时间权重；越近的舍牌数值越大。 |
| `132..716` | 585 | 三名对手的牌河，每人 195 层：最早 6 次各 8 层、最近 18 次各 8 层，再加 3 层时间衰减汇总。8 层会记录被吃碰的两张牌、杠牌、舍牌、赤牌、宝牌、手切和立直宣言。 |
| `717` | 1 | 牌山剩余张数除以 69。 |
| `718..721` | 4 | 四家已经持有的宝牌数量，各自除以 12。 |
| `722` | 1 | 尚未看见的宝牌张数，按最多 23 张缩放。 |
| `723..750` | 28 | 四家牌河的逐牌种总览，每家 7 层，编码方式与宝牌指示牌相同。 |
| `751..830` | 80 | 四家的公开副露，每家最多 4 组、每组 5 层：4 层记录同种牌的张数，1 层记录该组是否含赤牌。 |
| `831..834` | 4 | 四家的暗杠牌种，每家一层。 |
| `835` | 1 | 当前可见的每种牌数量除以 4。 |
| `836..844` | 9 | 三名对手最近一次手切，每人 3 层：牌种、赤牌、宝牌。 |
| `845..853` | 9 | 三名对手的立直宣言牌，每人同样使用牌种、赤牌、宝牌 3 层。 |
| `854..859` | 6 | 三名对手是否已宣告立直，以及立直是否已成立，各 3 层。 |
| `860` | 1 | 自己当前的听牌牌种。听牌是再得到其中一种牌即可和牌的状态。 |
| `861` | 1 | 自己是否振听；振听时不能荣和他人的舍牌。 |
| `862..868` | 7 | 向听数 0..6 的 one-hot 表示；向听数表示距离听牌还需改善几步。 |
| `869` | 1 | 自己的立直是否已经成立。 |
| `870` | 1 | 当前是否处在多个杠候选牌种的二次选择阶段。 |
| `871..873` | 3 | 等待响应的上一张舍牌：牌种、是否赤牌、是否为宝牌。没有吃碰杠和机会时均为 0。 |
| `874..878` | 5 | 可打牌信息：合法舍牌、保持当前向听数的舍牌、只增加一向听的舍牌、保证进入听牌的舍牌，以及自己是否已经宣告立直。 |
| `879..888` | 10 | 特殊动作是否可用：立直 1 层，三种吃法 3 层，碰、大明杠、暗杠、加杠、和牌、九种九牌流局各 1 层。 |
| `889..890` | 2 | 单人牌效率计算得到的最大预期得点，分别按 100000 点和 30000 点缩放。 |
| `891..960` | 70 | 进张关系。舍牌阶段用 68 层组成“候选舍牌 × 有效进张牌”的两个 `34 × 34` 关系表，区分是否会增加向听数，再用 1 层标记优先避免退向听的候选，另 1 层留空。非舍牌阶段只在最后 1 层汇总当前有效进张牌，其余留空。 |
| `961..1011` | 51 | 未来 1 至 17 巡的听牌概率、和牌概率和相对预期得点，各 17 层。 |

这里的“单人牌效率计算”只根据当前玩家可用的信息，估计不同舍牌的进张、听牌、和牌和得点前景。若计算条件不满足，相应的 `889..1011` 通道保持为 0；这部分是 v4 相比 v3 新增通道的主要来源。

Oracle 会把不可见信息沿通道维与 `obs` 拼接。模型类支持这一模式，但当前 [`mortal/train.py`](../mortal/train.py)、[`mortal/mortal.py`](../mortal/mortal.py) 和常用评测入口都没有启用它。

### 动作空间

[`libriichi/src/consts.rs`](../libriichi/src/consts.rs) 将 `ACTION_SPACE` 定义为 46。[`PlayerState.encode_obs()`](../libriichi/src/state/obs_repr.rs) 按下表填写 mask，[`DQN`](../mortal/model.py) 按相同编号输出 Q 值，[`MortalBatchAgent.get_reaction()`](../libriichi/src/agent/mortal.rs) 再把选中的编号解码为 mjai 事件：

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

合法动作由 bool mask 给出。`true` 表示当前规则允许该动作，`false` 表示不允许。例如玩家已经自摸和牌，同时也可以继续打牌时，mask 中若干个舍牌位置和编号 43 可以为 `true`；吃、碰、荣和只能在响应他人舍牌时出现，自己的摸牌回合中相应位置为 `false`。

再看一个简化示例：假设现在只能打一万、打九筒或自摸和牌，则 46 个位置中只有相应两个舍牌编号和编号 43 为 `true`。`DQN` 仍计算完整的 46 个动作评分，但会把其余 43 个非法动作设为 `-inf`。因此无论取最大值还是随机采样，都不可能选中非法动作。

### Q 值何时会变成概率

模型并非先为每个动作直接输出概率。v4 的最后一个线性层一次输出 47 个原始数值：1 个状态价值 `V(s)` 和 46 个动作优势 `A(s,a)`。`DQN` 用下文的 dueling 公式合成 46 个 Q 值，并将非法位置设为 `-inf`。

默认配置直接对 Q 值执行 `argmax`，即选择最大值，此时没有“归一化到和为 1”这一步。只有启用随机采样时，`MortalEngine` 才进行以下计算：

```text
合法动作的 logits = Q / temperature
非法动作的 logits = -inf
概率 = softmax(logits)
```

例如只允许“一万”和“和牌”，两者经温度调整后的 logits 分别为 `1.2` 和 `2.0`，其余都是 `-inf`。softmax 后概率约为 `31%` 和 `69%`，非法动作都是 `0%`，总和为 `100%`。这正是“模型先输出评分，再做归一化”；但它只用于采样，不是 DQN 的固定输出。softmax 的公式和 logits 一词见 [logits、softmax 和概率](deep_learning_basics.md#logitssoftmax-和概率)。若还配置了 `top_p < 1`，引擎会只保留累计概率达到阈值的一小组高概率动作，再从其中采样。

## 主决策网络

### `Brain`：状态编码

[`mortal/model.py`](../mortal/model.py) 中的 `Brain` 是面向 34 种牌轴的一维残差网络。所谓“把复杂局面压缩成状态向量”，是训练得到的非线性特征变换：许多带可学习权重的层不断组合原始特征，最后用较少的数值表示对选动作有用的信息。

以 v4 和示例配置的单个局面为例，实际过程是：

```text
(1012, 34)  原始观测，共 1012 * 34 = 34408 个数
-> (192, 34) 第一层 Conv1d：每个牌种位置生成 192 个初步特征
-> (192, 34) 40 个 ResBlock：反复修正特征，形状不变
-> (32, 34)  最后一层 Conv1d：把每个位置收束为 32 个特征
-> (1088,)   Flatten：按固定顺序摊平 32 * 34 个数
-> (1024,)   Linear：对全部 1088 个数做可学习的加权组合
-> (1024,)   Mish：加入非线性，得到状态向量 phi
```

第一层卷积会在每个牌种位置同时组合 1012 个输入通道以及相邻牌种的信息。例如它可以把“手里有三万”“四万已经看见三张”“五万是有效进张”等原始线索组合成更抽象的局部特征。40 个残差块继续调整这些特征；最后的 `Flatten + Linear` 才让原本分散在所有 34 个位置的信息互相组合，形成描述整个局面的向量。

`phi` 的 1024 个位置可以理解为模型自行学习的“局面特征坐标”。训练过程决定哪些信息应该被保留和组合，因此通常不能断言 `phi[0]` 固定表示分数、`phi[1]` 固定表示向听数。`DQN` 只需要这些数值足以帮助判断动作价值，不要求它们具有人能逐项命名的含义。

批量推理时，最前面再保留批次维，形状依次为 `(B, 1012, 34) -> (B, 192, 34) -> (B, 32, 34) -> (B, 1088) -> (B, 1024)`。每个局面最终各得到一行 `phi`。`conv_channels` 和 `num_blocks` 来自 `[resnet]`；示例配置是 192 个通道、40 个残差块，但这两个值不是写死的。卷积、残差块和每一步计算的进一步说明见 [深度学习基础概念与用到的网络介绍](deep_learning_basics.md#conv1d沿牌种轴观察相邻位置)。

### `ResBlock` 和通道注意力

每个残差块保留输入形状 `(B, conv_channels, 34)`。v4 使用 pre-activation（预激活）布局：

```text
x -> BatchNorm -> Mish -> Conv1d
  -> BatchNorm -> Mish -> Conv1d -> ChannelAttention -> + x
```

`ChannelAttention`（通道注意力）分别沿牌轴取平均值和最大值，经共享两层 MLP 后相加并取 sigmoid，得到每个通道的缩放权重。它只重标定通道，不改变张量形状。逐步说明和论文引用见 [ChannelAttention：调整各通道的重要程度](deep_learning_basics.md#channelattention调整各通道的重要程度)。

### `Brain` 版本差异

| 版本 | 激活/残差布局 | BatchNorm | 输出 |
| --- | --- | --- | --- |
| v1 | ReLU，post-activation | `eps` 默认值 | `mu` 和 `logsig`，各 512 维 |
| v2 | Mish，pre-activation | `eps` 默认值 | 1024 维 `phi` |
| v3 | Mish，pre-activation | `eps=1e-3` | 1024 维 `phi` |
| v4 | 同 v3 | `eps=1e-3` | 1024 维 `phi` |

v1 先执行 `Linear(1024, 512) + ReLU`，再生成高斯分布参数。推理时可使用 `mu`，也可在 `stochastic_latent=true` 时从 `Normal(mu, exp(logsig) + 1e-6)` 采样。v2–v4 使用确定性的 `phi`。

这是模型类支持的版本范围，不等于所有入口都支持所有版本。当前 [`mortal/train.py`](../mortal/train.py) 和 1024 维 `AuxNet` 没有处理 v1 的二元输出，标准主训练实际应使用 v2–v4；[`MortalEngine`](../mortal/engine.py) 的推理路径则显式支持 v1。

`Brain.freeze_bn(True)` 让 BatchNorm 在模型处于训练模式时仍使用累计统计量；它没有关闭 BatchNorm 参数的梯度。

### `DQN`：合法动作价值

[`mortal/model.py`](../mortal/model.py) 中的 `DQN` 不直接读取原始 `obs`。[`MortalEngine._react_batch()`](../mortal/engine.py) 先执行 `phi = Brain(obs)`，再执行 `q_out = DQN(phi, mask)`：

```text
obs:  (B, 1012, 34) -> Brain -> phi: (B, 1024)
mask: (B, 46) ---------------------> DQN -> q_out: (B, 46)
```

因此，`obs` 只进入 `Brain`，`mask` 只在 `DQN` 和后续选动作时使用。mask 不会改变 `Brain` 对局面的编码，DQN 也不会绕过 `Brain` 读取 1012 个原始通道。

v4 先用 `Linear(1024, 47)` 把每行 `phi` 变成 47 个原始数值，再拆成 1 个状态价值 `V(s)` 和 46 个动作优势 `A(s,a)`。状态价值评价“这个局面总体有多好”，动作优势评价“某个动作相对该局面的平均水平好多少”。DQN 使用 dueling 公式将二者合成 Q 值：

```text
Q(s,a) = V(s) + A(s,a) - mean_legal(A(s,*))
```

mask 在这里有两次作用：

1. 计算 `mean_legal` 时，非法动作的优势先被替换为 0，分母则是 mask 中 `true` 的数量，所以平均值只统计合法动作；
2. 得到 Q 值后，所有 mask 为 `false` 的位置被设为 `-inf`，从而不能被 `argmax` 或概率采样选中。

例如当前只有动作 0 和动作 43 合法，模型给出 `V = 2.0`，两者的优势分别是 `A(0) = 0.5`、`A(43) = 1.1`。合法优势平均值为 `(0.5 + 1.1) / 2 = 0.8`，因此：

```text
Q(0)  = 2.0 + 0.5 - 0.8 = 1.7
Q(43) = 2.0 + 1.1 - 0.8 = 2.3
其余 44 个动作 = -inf
```

最终会优先选择动作 43。这里不需要非法动作的优势参与比较；即使线性层为它们算出了很大的原始数值，最终的 `-inf` 也会将其排除。牌局状态机保证每次要求模型决策时至少有一个合法动作，否则合法动作数为 0 会使平均值无法计算。

| 版本 | 输入 | Value/Advantage 头 |
| --- | ---: | --- |
| v1 | 512 | 两个线性层：`512 -> 1/46` |
| v2 | 1024 | 两个 MLP：`1024 -> 512 -> 1/46` |
| v3 | 1024 | 两个 MLP：`1024 -> 256 -> 1/46` |
| v4 | 1024 | 单个 `Linear(1024, 47)` 后拆成 1 和 46 |

v4 仍按相同公式组合 `V` 与 `A`，只是用一个线性层共同生成两部分输出。

### `AuxNet`：辅助分类头

当前主干训练构造 `AuxNet((4,))`：

```text
phi: (B, 1024) -> Linear(1024, 4, bias=False) -> next-rank logits
```

标签是当前动作所在局结束后的玩家顺位。它用交叉熵为 `Brain` 增加结果相关监督，只参与训练；普通推理 checkpoint 虽包含其参数，选动作时不会调用它。

## 顺位预测网络

### 共同任务

[`libriichi/src/dataset/grp.rs`](../libriichi/src/dataset/grp.rs) 实现并导出的 `libriichi.dataset.Grp` 在每个 `start_kyoku` 提取一行 7 维特征。它负责构造数据，不要与 [`mortal/model.py`](../mortal/model.py) 中负责预测的神经网络类 `GRP` 混淆：

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

[`mortal/engine.py`](../mortal/engine.py) 中的 `MortalEngine` 默认对每行 Q 值执行 `argmax`，即返回最大 Q 值所在的动作编号。由于 `DQN` 已把非法动作设为 `-inf`，最大值一定来自合法动作。

这种做法的优势是：

- **确定且容易复现**：同一模型、局面和设备条件下总是选择同一个最高分动作，便于评测和排查问题；
- **计算简单**：只需比较 46 个数，不需要额外搜索或重复运行模型；
- **直接利用当前估计**：如果 Q 值准确，选择最大值就是网络认为长期回报最高的动作。

它也有明显局限：

- **会放大估计误差**：最高 Q 值可能只是模型高估了某个动作，`argmax` 仍会毫无保留地选择它；
- **忽略接近的候选**：Q 值为 `2.001` 和 `2.000` 的动作几乎并列，但 `argmax` 会始终选择前者；
- **没有探索和多样性**：相同局面不会尝试其他合理动作，不适合需要探索新策略的自对战训练；
- **不表达不确定性**：单个最高分不能说明模型对这个选择是否有把握。

项目已经提供一种低成本的折中方案。当 `boltzmann_epsilon > 0` 时，每个样本以 `1 - epsilon` 的概率继续使用 `argmax`，以 `epsilon` 的概率对 `Q / temperature` 做 top-p 采样：

- `top_p >= 1`：从全部合法动作采样；
- `0 < top_p < 1`：保留按概率降序构成 nucleus 的高概率动作（包含跨过阈值的那个动作）；
- `top_p <= 0`：退化为 argmax。

其中 `boltzmann_epsilon` 控制“有多少次不直接选最大值”，`boltzmann_temp` 控制采样概率的尖锐程度：温度越低，概率越集中在最高 Q 值附近；温度越高，其他合法动作越容易被选中。`top_p` 则限制参与采样的高概率候选范围。这三个选项由 [`mortal/config.example.toml`](../mortal/config.example.toml) 的自对战配置展示，并由 `MortalEngine` 实现；它们改变选择策略，不改变网络结构。

还可以考虑更复杂的改进，但当前仓库没有实现：例如用多个模型的平均 Q 值和分歧程度估计不确定性，或在候选动作后进行有限步搜索再综合即时结果与 Q 值。这些方法可能降低单次估计误差的影响，但会增加模型存储、推理次数或搜索开销，不能只通过调整现有三个采样参数启用。

完整推理路径仍是：

```text
PlayerState.encode_obs() -> Brain -> DQN -> MortalEngine 决定 argmax 或采样
```
