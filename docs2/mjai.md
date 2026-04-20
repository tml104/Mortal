# `mjai` 格式说明

这份文档基于 `libriichi` 里的实际代码，总结这个项目真正接受和产出的 `mjai` JSON 格式。

对应核心代码：

- [libriichi/src/mjai/event.rs](/home/tml104/Mortal/libriichi/src/mjai/event.rs)
- [libriichi/src/mjai/bot.rs](/home/tml104/Mortal/libriichi/src/mjai/bot.rs)
- [libriichi/src/tile.rs](/home/tml104/Mortal/libriichi/src/tile.rs)
- [libriichi/src/dataset/gameplay.rs](/home/tml104/Mortal/libriichi/src/dataset/gameplay.rs)

## 1. `mjai` 在这个项目里是什么

在 `libriichi` 中，`mjai` 是一种：

- 一行一个 JSON 事件
- 事件按时间顺序排列
- 每条事件用 `"type"` 区分种类

的麻将对局表示格式。

例如：

```json
{"type":"tsumo","actor":0,"pai":"1m"}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":true}
{"type":"reach","actor":1}
```

Rust 里对应的是 [Event](/home/tml104/Mortal/libriichi/src/mjai/event.rs) 这个枚举。它使用：

```rust
#[serde(tag = "type")]
#[serde(rename_all = "snake_case")]
```

所以 JSON 中的事件名是 snake_case，例如：

- `start_game`
- `start_kyoku`
- `reach_accepted`
- `end_kyoku`

## 2. 顶层约定

### 2.1 一行一个事件

日志通常是多行文本，每行都是一条 JSON：

```text
{"type":"start_game", ...}
{"type":"start_kyoku", ...}
{"type":"tsumo", ...}
{"type":"dahai", ...}
...
{"type":"end_game"}
```

Python 侧和 Rust 侧都按“逐行解析 JSON”的方式处理。

### 2.2 事件顺序很重要

`libriichi` 假设日志按真实对局时间顺序出现。典型顺序是：

```text
start_game
-> start_kyoku
-> tsumo / dahai / chi / pon / kan / reach / ...
-> hora 或 ryukyoku
-> end_kyoku
-> 下一局 start_kyoku
-> ...
-> end_game
```

### 2.3 actor / target 范围

代码中所有 `actor` / `target` 都限制为：

- `0`
- `1`
- `2`
- `3`

超出这个范围会在反序列化时报错。

### 2.4 `kyoku` 范围

`start_kyoku` 里的 `kyoku` 被限制为：

- `1..=4`

也就是说，局编号从 `1` 开始。

## 3. 牌字符串格式

牌的字符串格式由 [libriichi/src/tile.rs](/home/tml104/Mortal/libriichi/src/tile.rs) 定义。

### 3.1 数牌

- 万子：`1m` ~ `9m`
- 筒子：`1p` ~ `9p`
- 索子：`1s` ~ `9s`

### 3.2 字牌

- `E` 东
- `S` 南
- `W` 西
- `N` 北
- `P` 白
- `F` 发
- `C` 中

### 3.3 赤宝牌

- `5mr`
- `5pr`
- `5sr`

### 3.4 未知牌

- `?`

项目内部支持 `?`，但普通对局日志里一般不会主动出现。

## 4. 核心事件类型

下面按 `Event` 枚举中的顺序介绍。

### 4.1 `none`

格式：

```json
{"type":"none"}
```

作用：

- 表示“没有动作”或“空动作”。

### 4.2 `start_game`

格式：

```json
{
  "type":"start_game",
  "names":["A","B","C","D"],
  "seed":[123,456]
}
```

字段：

- `names: [string; 4]`
- `seed: [u64, u64] | null`

说明：

- `names` 缺省时会补默认值
- `seed` 是可选字段，代码注释写的是 `(nonce, key)`

### 4.3 `start_kyoku`

格式：

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
    ["N","3p","W","W","7m","N","S","C","7m","P","8p","2m","5m"],
    ["7p","1p","2m","3m","4m","C","7s","7s","9s","9p","1m","C","1s"],
    ["3s","E","5m","P","5m","F","7p","6m","5s","9p","1s","S","N"],
    ["2p","4s","4p","E","5p","F","3p","1s","8p","6s","8s","7s","5p"]
  ]
}
```

字段：

- `bakaze: Tile`
- `dora_marker: Tile`
- `kyoku: 1..=4`
- `honba: u8`
- `kyotaku: u8`
- `oya: 0..=3`
- `scores: [i32; 4]`
- `tehais: [[Tile; 13]; 4]`

### 4.4 `tsumo`

格式：

```json
{"type":"tsumo","actor":0,"pai":"1m"}
```

字段：

- `actor: 0..=3`
- `pai: Tile`

### 4.5 `dahai`

格式：

```json
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":true}
```

字段：

- `actor: 0..=3`
- `pai: Tile`
- `tsumogiri: bool`

说明：

- `tsumogiri=true` 表示摸切
- `tsumogiri=false` 表示手切

### 4.6 `chi`

格式：

```json
{"type":"chi","actor":1,"target":0,"pai":"6s","consumed":["5sr","7s"]}
```

字段：

- `actor`
- `target`
- `pai`
- `consumed: [Tile; 2]`

从状态校验逻辑看，`chi` 必须来自上家。

### 4.7 `pon`

格式：

```json
{"type":"pon","actor":1,"target":0,"pai":"C","consumed":["C","C"]}
```

字段：

- `actor`
- `target`
- `pai`
- `consumed: [Tile; 2]`

### 4.8 `daiminkan`

格式：

```json
{"type":"daiminkan","actor":2,"target":0,"pai":"5p","consumed":["5pr","5p","5p"]}
```

字段：

- `actor`
- `target`
- `pai`
- `consumed: [Tile; 3]`

### 4.9 `kakan`

格式：

```json
{"type":"kakan","actor":3,"pai":"S","consumed":["S","S","S"]}
```

字段：

- `actor`
- `pai`
- `consumed: [Tile; 3]`

### 4.10 `ankan`

格式：

```json
{"type":"ankan","actor":0,"consumed":["9m","9m","9m","9m"]}
```

字段：

- `actor`
- `consumed: [Tile; 4]`

### 4.11 `dora`

格式：

```json
{"type":"dora","dora_marker":"3s"}
```

字段：

- `dora_marker: Tile`

### 4.12 `reach`

格式：

```json
{"type":"reach","actor":1}
```

字段：

- `actor`

### 4.13 `reach_accepted`

格式：

```json
{"type":"reach_accepted","actor":2}
```

字段：

- `actor`

说明：

- `reach` 和 `reach_accepted` 是分开的两个事件

### 4.14 `hora`

格式：

```json
{
  "type":"hora",
  "actor":3,
  "target":1,
  "deltas":[0,-8000,0,9000],
  "ura_markers":["4p"]
}
```

字段：

- `actor`
- `target`
- `deltas: [i32; 4] | null`
- `ura_markers: list[Tile] | null`

说明：

- `deltas` 可选
- `ura_markers` 可选

### 4.15 `ryukyoku`

格式：

```json
{"type":"ryukyoku","deltas":[0,1500,0,-1500]}
```

或：

```json
{"type":"ryukyoku"}
```

字段：

- `deltas: [i32; 4] | null`

### 4.16 `end_kyoku`

格式：

```json
{"type":"end_kyoku"}
```

### 4.17 `end_game`

格式：

```json
{"type":"end_game"}
```

## 5. 项目里额外支持的扩展结构

除了标准 `Event` 以外，`libriichi` 还定义了两个常用扩展包装。

### 5.1 `EventExt`

定义位置：

- [libriichi/src/mjai/event.rs](/home/tml104/Mortal/libriichi/src/mjai/event.rs)

格式：

```json
{
  "type":"dahai",
  "actor":0,
  "pai":"2m",
  "tsumogiri":true,
  "meta":{
    "q_values":[1.2,0.3],
    "mask_bits":123,
    "is_greedy":true,
    "batch_size":1,
    "eval_time_ns":100000,
    "shanten":1,
    "at_furiten":false,
    "kan_select":{"q_values":[0.1,0.2]}
  }
}
```

其中 `meta` 是可选字段，对应：

- `q_values: list[float] | null`
- `mask_bits: u64 | null`
- `is_greedy: bool | null`
- `batch_size: usize | null`
- `eval_time_ns: u64 | null`
- `shanten: i8 | null`
- `at_furiten: bool | null`
- `kan_select: Metadata | null`

作用：

- 给原始 mjai 事件附加模型推理元信息

### 5.2 `EventWithCanAct`

定义位置：

- [libriichi/src/mjai/event.rs](/home/tml104/Mortal/libriichi/src/mjai/event.rs)

格式：

```json
{
  "type":"tsumo",
  "actor":0,
  "pai":"1m",
  "can_act":false
}
```

作用：

- 在普通事件外增加一个可选布尔字段 `can_act`

用途：

- `mjai.Bot.react(...)` 在收到事件后，会检查函数参数里的 `can_act` 和 JSON 中的 `can_act`
- 如果任一方禁止行动，bot 只更新状态，不返回动作

## 6. 项目里隐含的事件顺序约束

从 `dataset/gameplay.rs`、`mjai/bot.rs`、`state/update.rs` 的使用方式可以看出，这个项目默认日志满足下面这些约束。

### 6.1 整场牌局必须以 `start_game` 开头

否则某些解析逻辑会直接报错。

### 6.2 每局必须有 `start_kyoku`

很多特征提取逻辑都以 `start_kyoku` 为一局的开始。

### 6.3 每局结束后应该出现 `hora` 或 `ryukyoku`

然后再接：

- `end_kyoku`

### 6.4 `reach_accepted` 和 `dora` 常被视作“公告事件”

在 `dataset/gameplay.rs` 中，窗口逻辑会特别跳过：

- `reach_accepted`
- `dora`

来寻找真正的下一动作事件。

## 7. `target`、`deltas`、`ura_markers` 等字段的实际语义

### 7.1 `target`

在：

- `chi`
- `pon`
- `daiminkan`
- `hora`

里都可能出现 `target`。

通常表示：

- 此次动作针对的是谁的牌 / 谁放铳

### 7.2 `deltas`

出现在：

- `hora`
- `ryukyoku`

含义是：

- 四家点数变化数组

### 7.3 `ura_markers`

只出现在 `hora` 中，表示里宝指示牌列表。

## 8. `mjai.Bot` 实际接受和输出的格式

在 [libriichi/src/mjai/bot.rs](/home/tml104/Mortal/libriichi/src/mjai/bot.rs) 中：

- `Bot.react(line, can_act=True)` 接收的是一行 JSON 字符串
- 返回的是一个动作 JSON 字符串或 `None`

也就是说，Python 侧常见交互方式是：

输入：

```json
{"type":"tsumo","actor":0,"pai":"1m"}
```

输出：

```json
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":true}
```

## 9. 本项目中最常见的一段日志长什么样

下面是一段典型且合法的简化示例：

```text
{"type":"start_game","names":["A","B","C","D"],"seed":[123,456]}
{"type":"start_kyoku","bakaze":"E","dora_marker":"5s","kyoku":1,"honba":0,"kyotaku":0,"oya":0,"scores":[25000,25000,25000,25000],"tehais":[["N","3p","W","W","7m","N","S","C","7m","P","8p","2m","5m"],["7p","1p","2m","3m","4m","C","7s","7s","9s","9p","1m","C","1s"],["3s","E","5m","P","5m","F","7p","6m","5s","9p","1s","S","N"],["2p","4s","4p","E","5p","F","3p","1s","8p","6s","8s","7s","5p"]]}
{"type":"tsumo","actor":0,"pai":"1m"}
{"type":"dahai","actor":0,"pai":"2m","tsumogiri":true}
{"type":"chi","actor":1,"target":0,"pai":"6s","consumed":["5sr","7s"]}
{"type":"reach","actor":1}
{"type":"reach_accepted","actor":1}
{"type":"hora","actor":3,"target":1,"deltas":[0,-8000,0,9000],"ura_markers":["4p"]}
{"type":"end_kyoku"}
{"type":"end_game"}
```

## 10. 一句话总结

如果只记一句话，可以记成：

> 在这个项目里，`mjai` 就是“按时间顺序逐行排列的一组 JSON 事件”，每条事件由 `type` 区分，牌用 `1m/5pr/E/C` 这类字符串表示；在此基础上，项目还扩展支持了 `meta` 和 `can_act` 两种额外包装字段，用于模型分析和控制 bot 是否出手。

## 11. 其他示例

Mortal官方给出的一个示例：

```
{"type":"start_game"}
{"type":"start_kyoku","bakaze":"E","dora_marker":"3s","kyoku":3,"honba":0,"kyotaku":0,"oya":2,"scores":[22000,23700,26000,28300],"tehais":[["?","?","?","?","?","?","?","?","?","?","?","?","?"],["?","?","?","?","?","?","?","?","?","?","?","?","?"],["1m","1m","4m","5m","1p","5p","8p","1s","4s","4s","6s","8s","N"],["?","?","?","?","?","?","?","?","?","?","?","?","?"]]}
{"type":"tsumo","actor":2,"pai":"6p"}
{"type":"dahai","actor":2,"pai":"1s","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"1s","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"3s"}
{"type":"dahai","actor":2,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"9m","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"1m","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"1s","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"7s"}
{"type":"dahai","actor":2,"pai":"1p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"W","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"1p","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"W","tsumogiri":false}
{"type":"pon","actor":0,"target":1,"pai":"W","consumed":["W","W"]}
{"type":"dahai","actor":0,"pai":"2p","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"P"}
{"type":"dahai","actor":2,"pai":"8p","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"C","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"9p","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"1s","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"7m"}
{"type":"dahai","actor":2,"pai":"7m","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"7p","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"C","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"2s","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"6s"}
{"type":"dahai","actor":2,"pai":"6s","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"E","tsumogiri":true}
{"type":"pon","actor":1,"target":3,"pai":"E","consumed":["E","E"]}
{"type":"dahai","actor":1,"pai":"2m","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"7p"}
{"type":"dahai","actor":2,"pai":"P","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"8s","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"E","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"4p","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"8p"}
{"type":"dahai","actor":2,"pai":"8p","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"9s","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"S","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"6s","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"9m"}
{"type":"dahai","actor":2,"pai":"9m","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"2p","tsumogiri":false}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"2s","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"8p","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"F"}
{"type":"dahai","actor":2,"pai":"F","tsumogiri":true}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"dahai","actor":3,"pai":"4p","tsumogiri":true}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"4m","tsumogiri":true}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"S","tsumogiri":true}
{"type":"tsumo","actor":2,"pai":"5mr"}
{"type":"dahai","actor":2,"pai":"5m","tsumogiri":false}
{"type":"tsumo","actor":3,"pai":"?"}
{"type":"reach","actor":3}
{"type":"dahai","actor":3,"pai":"N","tsumogiri":false}
{"type":"reach_accepted","actor":3}
{"type":"tsumo","actor":0,"pai":"?"}
{"type":"dahai","actor":0,"pai":"N","tsumogiri":false}
{"type":"tsumo","actor":1,"pai":"?"}
{"type":"dahai","actor":1,"pai":"F","tsumogiri":false}
{"type":"tsumo","actor":2,"pai":"9p"}
```

