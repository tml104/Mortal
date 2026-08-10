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

解析器逐行读取日志，并对每一行分别调用 Python 的 json.loads 或 Rust 的 serde_json::from_str。不接受一个跨多行格式化的事件作为日志记录。典型生命周期是：

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

表中 `Tile` 表示上述牌字符串；字段名后的 `?` 表示该字段可省略或为 `null`，不要与表示未知牌的字符串 `"?"` 混淆。

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

完整的单局示例：

下面是一场只包含东一局的精简完整日志。玩家 0 摸切 `9p` 后，玩家 1 自摸 `5s`，以场风东的刻子和牌；每一行都是一个独立 JSON 事件。实际半庄或东风战会在 `end_game` 前重复出现多组 `start_kyoku` 到 `end_kyoku`。

```jsonl
{"type":"start_game","names":["A","B","C","D"],"seed":[123,456]}
{"type":"start_kyoku","bakaze":"E","dora_marker":"C","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["1p","1p","1p","6p","7p","8p","1s","2s","3s","9m","N","P","P"],["1m","2m","3m","4m","5m","6m","2p","3p","4p","5s","E","E","E"],["7m","8m","9m","5p","6p","7p","4s","5s","6s","S","S","F","C"],["2m","2m","2m","8p","8p","8p","7s","8s","9s","W","W","N","N"]]}
{"type":"tsumo","actor":0,"pai":"9p"}
{"type":"dahai","actor":0,"pai":"9p","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"5s"}
{"type":"hora","actor":1,"target":1,"deltas":[-1000,2000,-500,-500],"ura_markers":[]}
{"type":"end_kyoku"}
{"type":"end_game"}
```

## 字段语义与数据要求

以下类型采用 Rust/Serde 的写法：`u8`、`u64` 是无符号整数，`i32` 是有符号整数，`[T; N]` 是长度必须恰好为 `N` 的数组，`Vec<T>` 是可变长度数组。除明确标有“可选”的字段外，字段缺失或为 `null` 都会导致反序列化失败。示例中的单个 `"字段":值` 是 JSON 对象片段，需要与相应事件的其他必需字段一起使用。

### 事件种类：`type`

- **`type: str`**：每个事件都必须有此字段，它决定整行 JSON 被反序列化为哪个 `Event` 变体。值使用 snake_case，只能是字段表列出的事件名，例如 `"type":"start_kyoku"`、`"type":"reach_accepted"`。名称未知、大小写错误或缺失都会解析失败。
- **无业务字段的事件**：`none`、`end_kyoku`、`end_game` 除 `type` 外没有核心字段，完整事件分别为 `{"type":"none"}`、`{"type":"end_kyoku"}`、`{"type":"end_game"}`。`none` 表示当前决策选择不行动；后两者分别关闭当前局和整场对局。

### 整场字段：`names` 与 `seed`

- **`names: [str; 4]`**：仅用于 `start_game`，按绝对玩家 ID `0..3` 排列四个玩家名，数组长度必须恰好为 4，例如 `"names":["A","B","C","D"]` 表示玩家 0 名为 A、玩家 3 名为 D。该字段可以省略；Serde 会补成 `"names":["","","",""]`，但显式写成 `null` 不合法。
- **`seed?: [u64; 2]`**：仅用于 `start_game`，可选的二元数组依次表示 `(nonce, key)`，用于可复现的牌山或对局初始化，例如 `"seed":[123,456]`。两个元素都必须是 `0..2^64-1` 的整数；省略或写 `"seed":null` 表示没有种子信息，数组长度不是 2 会解析失败。

完整示例：

```json
{"type":"start_game","names":["A","B","C","D"],"seed":[123,456]}
```

### 开局字段：`bakaze`、`dora_marker`、`kyoku`、`honba`、`kyotaku`、`oya`、`scores` 与 `tehais`

- **`bakaze: Tile`**：仅用于 `start_kyoku`，表示当前场风。正常对局使用 `E`（东）、`S`（南）、`W`（西）或 `N`（北），例如 `"bakaze":"E"` 表示东场。Rust 字段类型本身是 `Tile`，所以其他合法牌字符串也能通过基础反序列化，但不具有正确的场风语义，不应使用。
- **`dora_marker: Tile`**：用于 `start_kyoku` 或 `dora`，表示公开的宝牌指示牌，而不是由它推导出的宝牌。例如 `"dora_marker":"4p"` 表示指示牌为四筒，对应宝牌为五筒；杠后翻出新指示牌时发送完整事件 `{"type":"dora","dora_marker":"3s"}`。实际日志中应提供可见的具体牌，不应写 `"?"`。
- **`kyoku: u8`**：仅用于 `start_kyoku`，表示当前场内的第几局，按 mjai 约定从 1 开始，且源码明确限制为 `1..4`。例如 `"bakaze":"S","kyoku":4` 表示南四局；不要写成零基的 `3`。
- **`honba: u8`**：仅用于 `start_kyoku`，表示本场数，即连续流局或庄家连庄累积的计数，不能为负数。例如 `"honba":2` 表示二本场。它参与点数结算，但不是点数本身。
- **`kyotaku: u8`**：仅用于 `start_kyoku`，表示桌面上尚未被领取的立直棒数量，不能为负数。例如 `"kyotaku":1` 表示当前有一根 1000 点立直棒。它与四家的 `scores` 分开记录。
- **`oya: u8`**：仅用于 `start_kyoku`，表示当前庄家的绝对玩家 ID，必须在 `0..3`。例如 `"oya":2` 表示玩家 2 是庄家；它不是相对于观察者的座位编号。
- **`scores: [i32; 4]`**：仅用于 `start_kyoku`，表示开局时四家的持点，依绝对玩家 ID `0..3` 排列，数组长度必须恰好为 4。例如 `"scores":[25000,24000,26000,25000]` 表示玩家 1 有 24000 点、玩家 2 有 26000 点。分数允许为负数；数组顺序不随庄家或观察者旋转。
- **`tehais: [[Tile; 13]; 4]`**：仅用于 `start_kyoku`，外层四项依绝对玩家 ID 排列，每名玩家的内层数组必须恰好包含 13 张起手牌。全信息日志可以写出四家实牌；单一玩家视角通常只写该玩家的实牌，其他三家各写 13 个 `"?"`。例如玩家 0 的一手牌可写为 `["1m","2m","3m","4p","5p","6p","7s","8s","9s","E","E","P","P"]`，但使用时仍须放进含四手牌的外层数组。

这些字段组成一个不可拆分的 `start_kyoku` 事件，完整写法见“关键事件示例”。`honba` 和 `kyotaku` 在 Rust 中是 `u8`，基础解析范围为 `0..255`；是否符合真实牌局进程仍由日志生产端和状态时序保证。

### 玩家字段：`actor` 与 `target`

- **`actor: u8`**：表示执行事件动作的玩家，必须为绝对玩家 ID `0..3`。它出现在 `tsumo`、`dahai`、`chi`、`pon`、`daiminkan`、`kakan`、`ankan`、`reach`、`reach_accepted` 和 `hora`。例如 `{"type":"tsumo","actor":2,"pai":"7s"}` 表示玩家 2 摸到七索，`{"type":"reach","actor":2}` 表示玩家 2 宣告立直。
- **`target: u8`**：表示动作所针对或取牌来源的绝对玩家 ID，必须为 `0..3`。在 `chi`、`pon`、`daiminkan` 中，它是打出 `pai` 的玩家，例如 `{"type":"pon","actor":2,"target":0,"pai":"C","consumed":["C","C"]}` 表示玩家 2 碰了玩家 0 打出的中；在 `hora` 中，它是放铳者，自摸则必须令 `target == actor`。例如 `{"type":"hora","actor":3,"target":1}` 是玩家 3 荣和玩家 1，`{"type":"hora","actor":3,"target":3}` 是玩家 3 自摸。

`actor`、`target` 和 `oya` 始终使用同一套绝对编号。状态机还会检查关系约束，例如不能碰自己的牌，吃牌必须满足 `(target + 1) % 4 == actor`。

### 牌动作字段：`pai`、`tsumogiri` 与 `consumed`

- **`pai: Tile`**：在 `tsumo` 中是刚摸到的牌，在 `dahai` 中是打出的牌，在 `chi`、`pon`、`daiminkan` 中是从 `target` 的最近一次舍牌取得的牌，在 `kakan` 中是加入原有碰组的第四张牌。例如 `{"type":"tsumo","actor":0,"pai":"5mr"}` 表示玩家 0 摸到赤五万；`{"type":"kakan","actor":0,"pai":"5mr","consumed":["5m","5m","5m"]}` 表示用赤五万给原有的五万碰组加杠。
- **`tsumogiri: bool`**：仅用于 `dahai`，表示舍牌是否就是该玩家刚摸到的牌。`true` 是摸切，例如摸到并立即打出一万时写 `{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}`；`false` 是从原手牌中切出，例如 `{"type":"dahai","actor":0,"pai":"2m","tsumogiri":false}`。该字段不能省略，且状态机会核对 `true` 时的 `pai` 与最近摸牌是否一致。
- **`consumed`**：表示 `actor` 原本持有并用于组成副露或杠子的牌，数组长度由事件类型固定。`chi` 和 `pon` 必须是 `[Tile; 2]`，如吃 `6s` 时写 `"consumed":["4s","5s"]`；`daiminkan` 和 `kakan` 必须是 `[Tile; 3]`，前者表示手中的三张同牌，后者表示原碰的三张牌；`ankan` 必须是 `[Tile; 4]`，如 `"consumed":["9m","9m","9m","9m"]`。`ankan` 没有 `pai`，四张牌全部放在 `consumed` 中。

完整的四种组合示例：

```jsonl
{"type":"chi","actor":1,"target":0,"pai":"6s","consumed":["4s","5sr"]}
{"type":"pon","actor":2,"target":0,"pai":"C","consumed":["C","C"]}
{"type":"daiminkan","actor":3,"target":0,"pai":"5p","consumed":["5pr","5p","5p"]}
{"type":"kakan","actor":3,"pai":"S","consumed":["S","S","S"]}
{"type":"ankan","actor":0,"consumed":["9m","9m","9m","9m"]}
```

数组长度和 JSON 类型只属于反序列化检查；`PlayerState` 还会验证这些牌是否在手、鸣牌是否针对最近舍牌、吃牌方向以及当前状态是否允许该动作。

### 结果字段：`deltas` 与 `ura_markers`

- **`deltas?: [i32; 4]`**：用于 `hora` 或 `ryukyoku`，表示该结果事件造成的四家点数变化，依绝对玩家 ID `0..3` 排列，通常总和为 0。例如 `"deltas":[0,-8000,0,8000]` 表示玩家 1 向玩家 3 支付 8000 点，`"deltas":[1500,-1500,1500,-1500]` 可表示流局时两家听牌、两家未听的罚符结算。多人和牌可以连续记录多个 `hora`，每个事件只写该次和牌对应的变化。
- **`ura_markers?: Vec<Tile>`**：仅用于 `hora`，表示和牌结算时公开的里宝牌指示牌列表，而不是里宝牌本身。例如 `"ura_markers":["4p"]` 表示一张里宝牌指示牌为四筒；没有里宝牌信息时可以省略、写 `null`，完整结果日志也常写空数组 `[]`。

这两个字段在 `Event` 结构中都是可选字段。Bot 可以用 `{"type":"hora","actor":3,"target":1}` 表示“选择和牌”，也可以用 `{"type":"ryukyoku"}` 表示“选择流局”，此时尚无结算数据。结果日志则通常写成：

```jsonl
{"type":"hora","actor":3,"target":1,"deltas":[0,-8000,0,8000],"ura_markers":["4p"]}
{"type":"ryukyoku","deltas":[1500,-1500,1500,-1500]}
```

特别是 `libriichi.dataset.Grp` 为了重建最终分数，要求最后一局结果阶段的 `hora` 或 `ryukyoku` 带 `deltas`。因此“Bot 可输出的事件”不一定已经是“可用于训练的完整结果日志”。

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
