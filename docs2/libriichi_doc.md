# `libriichi` Python API 速查

`libriichi` 是由 Rust + PyO3 构建的 Python 扩展，提供麻将状态、训练数据、对战环境、统计和 mjai Bot。模块注册入口是 [`libriichi/src/lib.rs`](../libriichi/src/lib.rs)。

```python
import libriichi

print(libriichi.__version__)
print(libriichi.__profile__)  # "debug" 或 "release"
```

当前公开子模块：

| 子模块 | 核心内容 | `mortal/` 是否直接使用 |
| --- | --- | --- |
| `libriichi.consts` | 观测/动作维度 | 是 |
| `libriichi.state` | 单玩家视角状态机 | 否，由其他 Rust 接口间接使用 |
| `libriichi.dataset` | 日志到训练数据 | 是 |
| `libriichi.arena` | 一对三、二对二对战 | 是 |
| `libriichi.stat` | 日志统计 | 是 |
| `libriichi.mjai` | 逐事件 Bot | 是 |

## 构建与导入

Linux：

```bash
cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
(cd mortal && python -c 'import libriichi; print(libriichi.__version__)')
```

Windows 将 `target/release/riichi.dll` 复制为 `mortal/libriichi.pyd`。修改 Rust 导出后，必须重新构建并替换 `mortal/` 下的文件；Python 导入的是复制后的扩展，不是 `target/` 中刚生成的产物。

本文使用 Python 侧可见类型描述返回值。错误通常由 PyO3 转为 Python 异常。

## `libriichi.consts`

定义见 [`consts.rs`](../libriichi/src/consts.rs)。

| 接口 | 值/返回值 | 用途 |
| --- | --- | --- |
| `MAX_VERSION` | `4` | 最大观测/模型版本 |
| `ACTION_SPACE` | `46` | DQN 输出和合法 mask 长度 |
| `GRP_SIZE` | `7` | 顺位模型单时间步特征数 |
| `obs_shape(version)` | 见下表 | 普通观测形状 |
| `oracle_obs_shape(version)` | 见下表 | Oracle 额外观测形状 |

| version | `obs_shape` | `oracle_obs_shape` |
| --- | --- | --- |
| 1 | `(938, 34)` | `(211, 34)` |
| 2 | `(942, 34)` | `(217, 34)` |
| 3 | `(934, 34)` | `(217, 34)` |
| 4 | `(1012, 34)` | `(217, 34)` |

函数对其他版本走 Rust `unreachable!()`，调用方应先限制在 `1..=MAX_VERSION`。

## `libriichi.state`

定义集中在 [`state/`](../libriichi/src/state)。核心关系：

```text
mjai JSON -> PlayerState.update() -> ActionCandidate
                     |
                     -> encode_obs() -> NumPy obs + mask
```

### `PlayerState`

```python
from libriichi.state import PlayerState

state = PlayerState(player_id=0)
cans = state.update('{"type":"start_game"}')
```

`player_id` 必须为 `0..3`，否则构造时会触发 Rust 断言并由 PyO3 报错；调用前应自行校验。

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `update(mjai_json)` | `ActionCandidate` | 解析一条事件、推进状态、返回当前候选动作 |
| `validate_reaction(mjai_json)` | `None` | 校验动作是否符合当前状态，非法时抛异常 |
| `brief_info()` | `str` | 人类可读调试状态，计算量可能较大 |
| `encode_obs(version, at_kan_select)` | `(np.ndarray, np.ndarray)` | 返回 `(channels,34)` float obs 和 `(46,)` bool mask |
| `last_self_tsumo()` | `Optional[str]` | 最近一次自己摸到的牌 |
| `last_kawa_tile()` | `Optional[str]` | 最近一张河牌 |
| `ankan_candidates()` | `list[str]` | 可暗杠牌种 |
| `kakan_candidates()` | `list[str]` | 可加杠牌种 |

`at_kan_select=True` 表示动作 42 已决定“要杠”，当前 mask 的 0..36 用来选择具体牌种；普通决策传 false。

只读属性按用途分组：

| 类别 | 属性 |
| --- | --- |
| 身份/局况 | `player_id`、`kyoku`、`honba`、`kyotaku`、`is_oya`、`at_turn` |
| 手牌/副露 | `tehai`、`akas_in_hand`、`chis`、`pons`、`minkans`、`ankans` |
| 牌效 | `shanten`、`waits` |
| 动作状态 | `last_cans`、`can_w_riichi`、`self_riichi_declared`、`self_riichi_accepted`、`at_furiten` |

`tehai` 和 `waits` 都按 34 种基本牌排列；赤牌状态另由 `akas_in_hand` 表达。

### `ActionCandidate`

这是 `PlayerState.update()` 和 `last_cans` 返回的只读候选集合。

原始字段：

```text
can_discard
can_chi_low, can_chi_mid, can_chi_high
can_pon
can_daiminkan, can_kakan, can_ankan
can_riichi
can_tsumo_agari, can_ron_agari
can_ryukyoku
target_actor
```

组合属性：

| 属性 | 定义 |
| --- | --- |
| `can_chi` | 任一种吃成立 |
| `can_kan` | 任一种杠成立 |
| `can_agari` | 自摸或荣和成立 |
| `can_pass` | 当前可以主动跳过吃/碰/大明杠/荣和 |
| `can_act` | 任一实际动作成立 |

另有 `__repr__()` 用于调试。不能在 Python 侧直接构造有业务含义的候选状态，应以 `PlayerState` 的结果为准。

## `libriichi.dataset`

定义见 [`dataset/`](../libriichi/src/dataset)。这里有两个容易混淆的 `Grp`/`GRP`：

- `libriichi.dataset.Grp`：Rust 日志特征容器；
- `mortal.model.GRP`：Python 神经网络。

### `GameplayLoader`

```python
from libriichi.dataset import GameplayLoader

loader = GameplayLoader(
    version,
    oracle=True,
    player_names=None,
    excludes=None,
    trust_seed=False,
    always_include_kan_select=True,
    augmented=False,
)
```

除 `version` 外都是 keyword-only 参数。

| 参数 | 作用 |
| --- | --- |
| `oracle` | 是否提取不可见观测；API 默认 true，标准主训练显式传 false |
| `player_names` | 非空时只保留指定名字的玩家视角 |
| `excludes` | `player_names` 为空时排除指定名字 |
| `trust_seed` | Oracle 是否信任 `start_game.seed` 重建隐藏信息 |
| `always_include_kan_select` | 杠后是否始终生成牌种二次选择样本 |
| `augmented` | 是否交换万/筒做数据增强 |

同名只读属性可检查实际配置，并支持 `repr(loader)`。

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `load_log(raw_log)` | `list[Gameplay]` | 一份日志按选中玩家视角展开，最多四项 |
| `load_gz_log_files(filenames)` | `list[list[Gameplay]]` | 外层与输入文件一一对应，Rust 内部并行读取 |

日志必须以 `start_game` 开始；解析、状态推进或特征提取失败会抛异常。

### `Gameplay`

`Gameplay` 保存一个玩家视角的逐动作训练数据。接口采用 `take_*`，多数调用会把 Rust 内部向量移出；同一字段第二次读取通常得到空数据。

| 方法 | 粒度 | 返回内容 |
| --- | --- | --- |
| `take_obs()` | 每动作 | `(channels,34)` NumPy 数组列表 |
| `take_invisible_obs()` | 每动作 | Oracle 额外观测列表 |
| `take_actions()` | 每动作 | 0..45 动作标签 |
| `take_masks()` | 每动作 | `(46,)` bool mask 列表 |
| `take_at_kyoku()` | 每动作 | 所属局的零基索引 |
| `take_dones()` | 每动作 | 是否为该局最后一个样本 |
| `take_apply_gamma()` | 每动作 | 是否增加折扣步数 |
| `take_at_turns()` | 每动作 | 当前巡目 |
| `take_shantens()` | 每动作 | 当前向听数 |
| `take_grp()` | 每玩家视角 | 对应 `Grp` 容器 |
| `take_player_id()` | 每玩家视角 | 绝对玩家 ID |

各逐动作数组必须按相同索引一起消费，不要在中途重复 `take` 某一字段。

### `Grp`

```python
from libriichi.dataset import Grp

game = Grp.load_log(raw_log)
games = Grp.load_gz_log_files(filenames)
```

| 方法 | 返回值 | 说明 |
| --- | --- | --- |
| `Grp.load_log(raw_log)` | `Grp` | 从一份逐行 mjai 日志提取特征 |
| `Grp.load_gz_log_files(filenames)` | `list[Grp]` | 每个文件返回一个容器 |
| `take_feature()` | `np.ndarray`，`(T,7)` float64 | 移出按局序列 |
| `take_rank_by_player()` | 4 个 int | 最终零基顺位 |
| `take_final_scores()` | 4 个 int | 最终分数 |

`take_feature()` 会移出特征；两个定长数组方法只是复制值。最终局结果缺少 `deltas` 时，特征提取可能失败。

## `libriichi.arena`

对战环境定义见 [`arena/`](../libriichi/src/arena)。构造签名相同：

```python
OneVsThree(disable_progress_bar=False, log_dir=None)
TwoVsTwo(disable_progress_bar=False, log_dir=None)
```

参数都是 keyword-only。`log_dir` 非空时写入 gzip mjai 日志；arena 会创建目录，但不会主动清空。项目上层的 `TrainPlayer`/`TestPlayer` 会在调用前删除其配置目录，这是 Python 包装层行为。

### `OneVsThree`

| 方法 | 返回值 |
| --- | --- |
| `py_vs_py(challenger, champion, seed_start, seed_count)` | challenger 的 `[一位数, 二位数, 三位数, 四位数]` |
| `ako_vs_py(engine, seed_start, seed_count)` | Akochan 作为 challenger 的名次计数 |
| `py_vs_ako(engine, seed_start, seed_count)` | Python agent 作为 challenger 的名次计数 |

`seed_start` 是 `(nonce, key)`。每个 seed 让 challenger 轮流坐四个位置，共运行 `4 * seed_count` 个半庄，因此返回计数之和也是该值。

### `TwoVsTwo`

| 方法 | 返回值 |
| --- | --- |
| `py_vs_py(challenger, champion, seed_start, seed_count)` | `None` |
| `ako_vs_py(engine, seed_start, seed_count)` | `None` |
| `py_vs_ako(engine, seed_start, seed_count)` | `None` |
| `py_vs_ako_one(engine, seed, split)` | `None` |

每个 seed 运行两个座位拆分，共 `2 * seed_count` 个半庄。该类当前没有被 `mortal/` 直接调用。

Python agent 最稳妥的实现参考 [`MortalEngine`](../mortal/engine.py)；arena 会要求引擎提供批量反应及若干生命周期属性/方法，而不只是一个普通 callable。

## `libriichi.stat`

```python
from libriichi.stat import Stat

stat = Stat.from_dir('/path/to/logs', 'mortal', disable_progress_bar=False)
one = Stat.from_log(raw_log, player_id=0)
```

| 方法 | 说明 |
| --- | --- |
| `from_dir(dir, player_name, disable_progress_bar=False)` | 递归读取 `.json`/`.json.gz`，按 `start_game.names` 聚合目标玩家 |
| `from_log(log, player_id)` | 统计一份日志的指定座位 |
| `total_pt([p1,p2,p3,p4])` | 名次计数加权总 pt |
| `avg_pt([p1,p2,p3,p4])` | 每场平均 pt |
| `str(stat)` / `repr(stat)` | 文本汇总/调试表示 |

可读写累计字段：

| 类别 | 字段 |
| --- | --- |
| 基础 | `game`、`round`、`oya`、`point`、`rank_1..rank_4`、`tobi` |
| 副露 | `fuuro`、`fuuro_num`、`fuuro_point`、`fuuro_agari`、`fuuro_agari_jun`、`fuuro_agari_point`、`fuuro_houjuu` |
| 和牌 | `agari`、`agari_as_oya`、`agari_jun`、`agari_point_oya`、`agari_point_ko` |
| 放铳 | `houjuu`、`houjuu_jun`、`houjuu_to_oya`、`houjuu_point_to_oya`、`houjuu_point_to_ko` |
| 立直 | `riichi`、`riichi_as_oya`、`riichi_jun`、`riichi_agari`、`riichi_agari_point`、`riichi_agari_jun`、`riichi_houjuu`、`riichi_ryukyoku`、`riichi_point`、`chasing_riichi`、`riichi_got_chased` |
| 其他 | `dama_agari`、`dama_agari_jun`、`dama_agari_point`、`ryukyoku`、`ryukyoku_point`、`yakuman`、`nagashi_mangan` |

只读计算属性：

| 类别 | 属性 |
| --- | --- |
| 名次 | `avg_rank`、`rank_1_rate..rank_4_rate`、`tobi_rate` |
| 平均得点 | `avg_point_per_game`、`avg_point_per_round`、`avg_point_per_agari`、`avg_point_per_oya_agari`、`avg_point_per_ko_agari`、`avg_point_per_riichi_agari`、`avg_point_per_fuuro_agari`、`avg_point_per_dama_agari`、`avg_point_per_ryukyoku` |
| 放铳得点 | `avg_point_per_houjuu`、`avg_point_per_houjuu_to_oya`、`avg_point_per_houjuu_to_ko` |
| 平均巡目 | `avg_agari_jun`、`avg_riichi_agari_jun`、`avg_fuuro_agari_jun`、`avg_dama_agari_jun`、`avg_houjuu_jun`、`avg_riichi_jun` |
| 基础比率 | `agari_rate`、`houjuu_rate`、`riichi_rate`、`fuuro_rate`、`ryukyoku_rate` |
| 立直 | `agari_rate_after_riichi`、`houjuu_rate_after_riichi`、`chasing_riichi_rate`、`riichi_chased_rate`、`avg_riichi_point` |
| 庄家/副露 | `agari_rate_as_oya`、`agari_as_oya_rate`、`houjuu_to_oya_rate`、`avg_fuuro_num`、`agari_rate_after_fuuro`、`houjuu_rate_after_fuuro`、`avg_fuuro_point` |
| 特殊 | `yakuman_rate`、`nagashi_mangan_rate` |

很多计算属性直接执行“累计值 / 次数”。当分母为 0 时可能得到 `NaN`，消费方应自行处理，而不要假设所有指标都是有限数。

## `libriichi.mjai`

当前 Python 子模块只公开 `Bot`；Rust 的 `Event`、`EventExt` 等没有注册为 Python 类。

```python
from libriichi.mjai import Bot

bot = Bot(engine, player_id=0)
reaction = bot.react(line, can_act=True)
```

| 接口 | 说明 |
| --- | --- |
| `Bot(engine, player_id)` | 用 Python engine 和 0..3 seat 创建状态化 Bot |
| `react(line, *, can_act=True)` | 更新一条 mjai 事件；可行动时返回动作 JSON，否则返回 `None` |

engine 必须满足 Rust `MortalBatchAgent` 需要的接口。当前项目传入 [`MortalEngine`](../mortal/engine.py)，其关键成员包括 `react_batch`、`name`、`is_oracle`、`version`、`enable_quick_eval` 和 `enable_rule_based_agari_guard`。

JSON 顶层也可以带 `can_act:false`。函数参数或 JSON 任一方禁止行动时，Bot 仍推进状态，但不请求模型动作。协议细节见 [mjai.md](mjai.md)。

## 当前 Python 调用关系

```text
model.py       <- consts
dataloader.py  <- dataset.GameplayLoader
train_grp*.py  <- dataset.Grp
player.py      <- arena.OneVsThree + stat.Stat
one_vs_three.py<- arena.OneVsThree
mortal.py      <- mjai.Bot + dataset.Grp（仅复盘额外使用 Grp）
```

如果修改观测、动作、数据容器或 PyO3 签名，应同时核对这些调用方，并重新构建/复制扩展后再做 Python 导入验证。
