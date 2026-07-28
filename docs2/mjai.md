# mjai 协议速查

Mortal 使用逐行 JSON 的 mjai 事件流表示对局。事件结构以 [`libriichi/src/mjai/event.rs`](../libriichi/src/mjai/event.rs) 为准，牌字符串以 [`tile.rs`](../libriichi/src/tile.rs) 为准。

```text
一行 JSON = 一个事件
多行按发生顺序组成一局或整场对局
type 字段决定事件种类
```

示例：

```json
{"type":"tsumo","actor":0,"pai":"1m"}
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
```

## 基本约束

### 行与顺序

解析器按行调用 `json.loads`/Serde，不接受一个跨多行格式化的事件作为日志记录。典型生命周期是：

```text
start_game
-> start_kyoku
-> tsumo / dahai / chi / pon / kan / reach / ...
-> 一个或多个 hora，或 ryukyoku
-> end_kyoku
-> 下一局 start_kyoku ...
-> end_game
```

JSON 结构合法不代表事件序列合法。`PlayerState` 会按历史状态校验牌、动作和时序；`GameplayLoader` 还要求整场日志以 `start_game` 开始。

### 玩家编号

`actor`、`target` 和 `oya` 是绝对玩家 ID，只能为 `0..3`。超出范围会在反序列化时失败。

- `actor`：执行动作的人；
- `target`：提供被鸣牌的人，或和牌时的放铳者；自摸日志通常令 `target == actor`；
- `oya`：当前庄家。

`start_kyoku.kyoku` 从 1 开始，只能为 `1..4`。`PlayerState` 内部会转换成自己的零基表示，不应把内部值直接写回 mjai。

### 牌字符串

| 类别 | 合法值 |
| --- | --- |
| 万子 | `1m` .. `9m` |
| 筒子 | `1p` .. `9p` |
| 索子 | `1s` .. `9s` |
| 风牌 | `E`、`S`、`W`、`N` |
| 三元牌 | `P`（白）、`F`（发）、`C`（中） |
| 赤五 | `5mr`、`5pr`、`5sr` |
| 未知牌 | `?` |

字符串区分大小写；`0m`、`5M` 等不会被接受。`?` 主要用于隐藏其他玩家的手牌。

## 事件字段表

表中 `Tile` 表示上述牌字符串，`?` 表示字段可省略或为 null。

| `type` | 字段 | 说明 |
| --- | --- | --- |
| `none` | 无 | 无动作/跳过 |
| `start_game` | `names: [str;4]`，`seed?: [u64;2]` | `names` 缺省时为四个空字符串；seed 是 `(nonce, key)` |
| `start_kyoku` | `bakaze`、`dora_marker`、`kyoku`、`honba`、`kyotaku`、`oya`、`scores: [i32;4]`、`tehais: [[Tile;13];4]` | 开始一局，字段均必需 |
| `tsumo` | `actor`、`pai` | 摸牌 |
| `dahai` | `actor`、`pai`、`tsumogiri: bool` | `true` 为摸切，`false` 为手切 |
| `chi` | `actor`、`target`、`pai`、`consumed: [Tile;2]` | 吃牌；状态机要求来自上家 |
| `pon` | `actor`、`target`、`pai`、`consumed: [Tile;2]` | 碰牌 |
| `daiminkan` | `actor`、`target`、`pai`、`consumed: [Tile;3]` | 大明杠 |
| `kakan` | `actor`、`pai`、`consumed: [Tile;3]` | 加杠，`consumed` 描述原碰 |
| `ankan` | `actor`、`consumed: [Tile;4]` | 暗杠 |
| `dora` | `dora_marker` | 新宝牌指示牌公告 |
| `reach` | `actor` | 宣告立直 |
| `reach_accepted` | `actor` | 立直成立，与 `reach` 分开 |
| `hora` | `actor`、`target`、`deltas?: [i32;4]`、`ura_markers?: [Tile]` | 和牌；可连续出现以表示多人和牌 |
| `ryukyoku` | `deltas?: [i32;4]` | 流局 |
| `end_kyoku` | 无 | 当前局结束 |
| `end_game` | 无 | 整场结束 |

### 关键事件示例

开始整场：

```json
{"type":"start_game","names":["A","B","C","D"],"seed":[123,456]}
```

开始一局：

```json
{
  "type":"start_kyoku",
  "bakaze":"E",
  "dora_marker":"5s",
  "kyoku":1,
  "honba":0,
  "kyotaku":0,
  "oya":0,
  "scores":[25000,25000,25000,25000],
  "tehais":[
    ["1m","2m","3m","4m","5m","6m","7m","8m","9m","1p","2p","3p","E"],
    ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
    ["?","?","?","?","?","?","?","?","?","?","?","?","?"],
    ["?","?","?","?","?","?","?","?","?","?","?","?","?"]
  ]
}
```

鸣牌：

```json
{"type":"chi","actor":1,"target":0,"pai":"6s","consumed":["5sr","7s"]}
{"type":"pon","actor":2,"target":0,"pai":"C","consumed":["C","C"]}
{"type":"daiminkan","actor":3,"target":0,"pai":"5p","consumed":["5pr","5p","5p"]}
```

终局：

```json
{"type":"hora","actor":3,"target":1,"deltas":[0,-8000,0,9000],"ura_markers":["4p"]}
{"type":"end_kyoku"}
{"type":"end_game"}
```

## 字段语义与数据要求

### `pai` 与 `consumed`

鸣牌事件中：

- `pai` 是从 `target` 获得或用于加杠的那张牌；
- `consumed` 是 `actor` 原本持有并被组合使用的牌；
- 数组长度由事件类型固定，不能用可变长度代替。

状态机不仅检查数组长度，还会验证牌是否在手、吃牌方向和当前是否允许该动作。

### `deltas`

`deltas` 是四家的点数变化，按绝对玩家 ID 排列。`Event` 类型允许 `hora`/`ryukyoku` 省略它，方便 Bot 输出“我要和牌/流局”的动作请求；完整结果日志通常需要带上它。

特别是 `libriichi.dataset.Grp` 为了重建最终分数，要求最后一局结果阶段的 `hora` 或 `ryukyoku` 带 `deltas`。因此“Bot 可输出的事件”不一定已经是“可用于训练的完整结果日志”。

### `ura_markers`

只用于 `hora`，记录里宝牌指示牌列表。字段在协议结构上可选。

### 立直是两个事件

```text
reach -> dahai -> reach_accepted
```

`reach` 表示声明，`reach_accepted` 表示成立和扣除立直棒。两者不能合并，否则状态、分数和 GRP 特征可能不一致。

### 公告事件

`reach_accepted` 和 `dora` 常插在一个决策动作与下一个实际动作之间。`GameplayLoader` 的动作标签窗口会跳过这两类公告寻找下一动作；不能据此随意删除公告，它们仍会更新状态。

## Mortal 扩展字段

### `meta`

`EventExt` 在普通事件上扁平追加可选 `meta`，用于记录模型推理信息：

```json
{
  "type":"dahai",
  "actor":0,
  "pai":"2m",
  "tsumogiri":false,
  "meta":{
    "q_values":[1.2,0.3],
    "mask_bits":5,
    "is_greedy":true,
    "batch_size":1,
    "eval_time_ns":100000,
    "shanten":1,
    "at_furiten":false,
    "kan_select":{"q_values":[0.1,0.2]}
  }
}
```

| 字段 | 含义 |
| --- | --- |
| `q_values` | 按合法 mask 压缩后的 Q 值 |
| `mask_bits` | 46 位合法动作 bitset |
| `is_greedy` | 是否由贪心分支选择 |
| `batch_size` | 本次推理 batch 大小 |
| `eval_time_ns` | 推理耗时 |
| `shanten` | 当前向听数 |
| `at_furiten` | 当前是否振听 |
| `kan_select` | 杠牌二次牌种选择的嵌套元数据 |

这些字段均可选，不属于状态推进所需的核心事件字段。

### `can_act`

`EventWithCanAct` 允许在任意事件顶层附加：

```json
{"type":"tsumo","actor":0,"pai":"1m","can_act":false}
```

`Bot.react(line, can_act=True)` 会先更新状态，再判断是否决策。函数参数为 false，或 JSON 的 `can_act` 明确为 false，都会强制只更新状态并返回 `None`。

## `libriichi.mjai.Bot` 的输入输出

Python 调用签名：

```python
reaction = bot.react(line, can_act=True)
```

- `line`：恰好一条事件的 JSON 字符串；
- 返回 `str`：当前玩家可行动，内容是动作事件，通常带 `meta`；
- 返回 `None`：当前不能行动或被 `can_act` 禁止；
- 非法 JSON、非法时序或非法状态：抛出异常。

`mortal/mortal.py` 从 stdin 逐行调用该方法，只在返回字符串时写 stdout。复盘模式会为无动作时额外输出带空 mask 的 `none`，这是 Python 入口行为，不是 `Bot.react()` 自身返回值。

## 排错顺序

解析或训练日志失败时，按以下顺序检查：

1. 每一行是否是独立且完整的 JSON；
2. `type` 名是否为 snake_case；
3. 牌字符串、玩家 ID、`kyoku` 和定长数组是否满足范围；
4. 是否以 `start_game` 开始，每局是否有完整开始/结果/结束事件；
5. `reach`、`reach_accepted`、`dora` 的顺序是否保留；
6. 最后一局结果是否带 `deltas`；
7. 动作是否与此前状态、手牌和合法候选一致。
