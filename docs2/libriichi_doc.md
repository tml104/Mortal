# `libriichi` Python 接口文档

这份文档总结的是：`libriichi` 这个 Rust crate 通过 `pyo3` 暴露给 Python 侧的所有模块，以及这些模块里实际可用的接口。

对应入口在 [libriichi/src/lib.rs](/home/tml104/Mortal/libriichi/src/lib.rs)。

Python 侧导入的是一个扩展模块：

```python
import libriichi
```

在该模块下，当前对 Python 暴露的子模块有：

- `libriichi.consts`
- `libriichi.state`
- `libriichi.dataset`
- `libriichi.arena`
- `libriichi.stat`
- `libriichi.mjai`

其中 `mortal/` 目录里目前实际直接使用到的是：

- `libriichi.consts`
- `libriichi.dataset`
- `libriichi.arena`
- `libriichi.stat`
- `libriichi.mjai`

`libriichi.state` 当前没有被 `mortal/` 直接导入使用，但它也是已公开的 Python 接口。

---

## 1. 顶层模块 `libriichi`

### 暴露内容

来自 [libriichi/src/lib.rs](/home/tml104/Mortal/libriichi/src/lib.rs)：

- `__version__`
- `__profile__`
- 子模块：
  - `consts`
  - `state`
  - `dataset`
  - `arena`
  - `stat`
  - `mjai`

### 说明

- `__version__` 对应 Rust crate 的版本号。
- `__profile__` 用来标识当前是 debug 还是 release 构建。

### Python 侧使用情况

当前 `mortal/` 中没有直接读取 `libriichi.__version__` 或 `libriichi.__profile__`。

---

## 2. `libriichi.consts`

注册位置：

- [libriichi/src/consts.rs](/home/tml104/Mortal/libriichi/src/consts.rs)

### 暴露函数

#### `obs_shape(version: int) -> tuple[int, int]`

作用：

- 返回对应版本的主观测张量形状。

返回值：

- v1: `(938, 34)`
- v2: `(942, 34)`
- v3: `(934, 34)`
- v4: `(1012, 34)`

Python 侧使用：

- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)
- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)

用途：

- 决定 `Brain` 的输入通道数
- 打印当前训练版本的观测形状

#### `oracle_obs_shape(version: int) -> tuple[int, int]`

作用：

- 返回 oracle 模式下额外不可见信息张量的形状。

返回值：

- v1: `(211, 34)`
- v2/v3/v4: `(217, 34)`

Python 侧使用：

- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)

用途：

- `Brain(is_oracle=True)` 时计算额外输入通道数

### 暴露常量

#### `MAX_VERSION`

作用：

- 当前最大支持版本号，值为 `4`。

#### `ACTION_SPACE`

作用：

- 动作空间大小，值为 `46`。

Python 侧使用：

- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)

用途：

- 决定 `DQN` 输出动作维度

#### `GRP_SIZE`

作用：

- `GRP` 网络单步输入特征维度，值为 `7`。

Python 侧使用：

- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)

用途：

- 决定 `GRP` 的 `GRU(input_size=GRP_SIZE, ...)`

---

## 3. `libriichi.state`

注册位置：

- [libriichi/src/state/mod.rs](/home/tml104/Mortal/libriichi/src/state/mod.rs)

### 暴露类

- `ActionCandidate`
- `PlayerState`

### 3.1 `ActionCandidate`

定义位置：

- [libriichi/src/state/action.rs](/home/tml104/Mortal/libriichi/src/state/action.rs)

这是一个描述“当前玩家此刻有哪些动作可以做”的对象。

#### 可读字段

- `can_discard`
- `can_chi_low`
- `can_chi_mid`
- `can_chi_high`
- `can_pon`
- `can_daiminkan`
- `can_kakan`
- `can_ankan`
- `can_riichi`
- `can_tsumo_agari`
- `can_ron_agari`
- `can_ryukyoku`
- `target_actor`

#### 计算属性

- `can_chi`
- `can_kan`
- `can_agari`
- `can_pass`
- `can_act`

#### 其他方法

- `__repr__()`

### Python 侧使用情况

`mortal/` 没有直接创建 `ActionCandidate`，但它会被：

- `PlayerState.last_cans`
- `mjai.Bot`

等 Rust 逻辑内部使用。

### 3.2 `PlayerState`

定义位置：

- [libriichi/src/state/player_state.rs](/home/tml104/Mortal/libriichi/src/state/player_state.rs)
- [libriichi/src/state/getter.rs](/home/tml104/Mortal/libriichi/src/state/getter.rs)
- [libriichi/src/state/obs_repr.rs](/home/tml104/Mortal/libriichi/src/state/obs_repr.rs)

这是 `libriichi` 最核心的状态类，用来维护某个 seat 视角下的麻将可观测状态。

#### 构造方法

`PlayerState(player_id: int)`

作用：

- 创建指定座位视角的状态机对象。

#### 核心方法

##### `update(mjai_json: str) -> ActionCandidate`

作用：

- 输入一条 mjai JSON 事件，更新状态，并返回当前可行动作集合。

##### `validate_reaction(mjai_json: str) -> None`

作用：

- 校验某条 mjai JSON 动作是否是当前状态下的合法反应。
- 非法时抛异常。

##### `brief_info() -> str`

作用：

- 返回当前状态的人类可读调试信息。

##### `encode_obs(version: int, at_kan_select: bool) -> tuple[np.ndarray, np.ndarray]`

作用：

- 把当前状态编码成：
  - 观测特征 `obs`
  - 动作 mask `mask`

这正是 Mortal 训练和推理所需的网络输入格式。

#### 只读属性

- `player_id`
- `kyoku`
- `honba`
- `kyotaku`
- `is_oya`
- `tehai`
- `akas_in_hand`
- `chis`
- `pons`
- `minkans`
- `ankans`
- `at_turn`
- `shanten`
- `waits`
- `last_self_tsumo()`
- `last_kawa_tile()`
- `last_cans`
- `ankan_candidates()`
- `kakan_candidates()`
- `can_w_riichi`
- `self_riichi_declared`
- `self_riichi_accepted`
- `at_furiten`

### Python 侧使用情况

当前 `mortal/` 没有直接导入 `PlayerState`，但它被公开后主要用于：

- `libriichi.dataset.GameplayLoader` 内部构造训练样本
- `libriichi.mjai.Bot` 内部维护对局状态

也就是说，它虽然没在 Python 训练代码里直接出现，但其实是很多 Rust 暴露接口的底层基础。

---

## 4. `libriichi.dataset`

注册位置：

- [libriichi/src/dataset/mod.rs](/home/tml104/Mortal/libriichi/src/dataset/mod.rs)

### 暴露类

- `Gameplay`
- `GameplayLoader`
- `Grp`

其中 Python 侧实际直接用到的是：

- `GameplayLoader`
- `Grp`

### 4.1 `GameplayLoader`

定义位置：

- [libriichi/src/dataset/gameplay.rs](/home/tml104/Mortal/libriichi/src/dataset/gameplay.rs)

#### 构造方法

```python
GameplayLoader(
    version,
    *,
    oracle=True,
    player_names=None,
    excludes=None,
    trust_seed=False,
    always_include_kan_select=True,
    augmented=False,
)
```

#### 可读属性

- `version`
- `oracle`
- `player_names`
- `excludes`
- `trust_seed`
- `always_include_kan_select`
- `augmented`

#### 方法

##### `load_log(raw_log: str) -> list[Gameplay]`

作用：

- 读取一份原始 mjai 文本日志，解析成若干 `Gameplay` 实例。

##### `load_gz_log_files(gzip_filenames: list[str]) -> list[list[Gameplay]]`

作用：

- 批量读取 `.json.gz` 日志文件。

##### `__repr__()`

#### Python 侧使用

- [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py)

具体用途：

- 读取离线或在线生成的日志
- 提取逐步训练样本

### 4.2 `Gameplay`

定义位置：

- [libriichi/src/dataset/gameplay.rs](/home/tml104/Mortal/libriichi/src/dataset/gameplay.rs)

这个类代表从单局日志中提取出的、面向训练的一份结构化数据。

它没有暴露可直接读取的大量属性，而是通过一组 `take_*` 方法把内部缓存“取走”。

#### 方法

- `take_obs() -> list[np.ndarray]`
- `take_invisible_obs() -> list[np.ndarray]`
- `take_actions() -> list[int]`
- `take_masks() -> list[np.ndarray]`
- `take_at_kyoku() -> list[int]`
- `take_dones() -> list[bool]`
- `take_apply_gamma() -> list[bool]`
- `take_at_turns() -> list[int]`
- `take_shantens() -> list[int]`
- `take_grp() -> Grp`
- `take_player_id() -> int`

#### Python 侧使用

- [mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py)

具体用途：

- 拆出 `obs/actions/masks/...`
- 再拼成主干训练样本

### 4.3 `Grp`

定义位置：

- [libriichi/src/dataset/grp.rs](/home/tml104/Mortal/libriichi/src/dataset/grp.rs)

这个类不是 Python 侧的神经网络 `GRP`，而是从对局日志中提取出的“排名预测特征数据容器”。

#### 静态方法

##### `Grp.load_log(raw_log: str) -> Grp`

作用：

- 从一份原始日志中提取 `grp` 特征。

##### `Grp.load_gz_log_files(gzip_filenames: list[str]) -> list[Grp]`

作用：

- 批量读取压缩日志并提取 `grp` 特征。

#### 实例方法

- `take_feature() -> np.ndarray`
- `take_rank_by_player() -> list[int] / tuple[int, ...]`
- `take_final_scores() -> list[int] / tuple[int, ...]`

#### Python 侧使用

- [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py)
- [mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)

具体用途：

- `train_grp.py`：构造 `GRP` 网络的训练集
- `mortal.py`：在 `review_mode` 下，从整局日志提取特征，再送给 Python 侧的 `GRP` 网络做排名概率分析

---

## 5. `libriichi.arena`

注册位置：

- [libriichi/src/arena/mod.rs](/home/tml104/Mortal/libriichi/src/arena/mod.rs)

### 暴露类

- `OneVsThree`
- `TwoVsTwo`

Python 侧当前实际用到的是：

- `OneVsThree`

### 5.1 `OneVsThree`

定义位置：

- [libriichi/src/arena/one_vs_three.rs](/home/tml104/Mortal/libriichi/src/arena/one_vs_three.rs)

#### 构造方法

```python
OneVsThree(*, disable_progress_bar=False, log_dir=None)
```

#### 方法

##### `py_vs_py(challenger, champion, seed_start, seed_count) -> list[int]`

作用：

- 让两个 Python agent 进行 1v3 对战。
- 返回 challenger 的排名计数。

##### `ako_vs_py(engine, seed_start, seed_count) -> list[int]`

作用：

- 让 Akochan 作为 challenger，对战 Python agent。

##### `py_vs_ako(engine, seed_start, seed_count) -> list[int]`

作用：

- 让 Python agent 作为 challenger，对战 Akochan。

#### Python 侧使用

- [mortal/player.py](/home/tml104/Mortal/mortal/player.py)
- [mortal/one_vs_three.py](/home/tml104/Mortal/mortal/one_vs_three.py)

具体用途：

- 定期测试当前模型 vs baseline
- 在线训练中产生日志
- 单独的 1v3 评测脚本

### 5.2 `TwoVsTwo`

定义位置：

- [libriichi/src/arena/two_vs_two.rs](/home/tml104/Mortal/libriichi/src/arena/two_vs_two.rs)

#### 构造方法

```python
TwoVsTwo(*, disable_progress_bar=False, log_dir=None)
```

#### 方法

- `py_vs_py(challenger, champion, seed_start, seed_count) -> None`
- `ako_vs_py(engine, seed_start, seed_count) -> None`
- `py_vs_ako(engine, seed_start, seed_count) -> None`
- `py_vs_ako_one(engine, seed, split) -> None`

#### Python 侧使用

当前 `mortal/` 中没有直接使用 `TwoVsTwo`。

---

## 6. `libriichi.stat`

注册位置：

- [libriichi/src/stat.rs](/home/tml104/Mortal/libriichi/src/stat.rs)

### 暴露类

- `Stat`

### `Stat`

这是一个从日志中统计玩家表现的聚合对象。

#### 静态方法

##### `Stat.from_dir(dir: str, player_name: str, disable_progress_bar=False) -> Stat`

作用：

- 递归统计目录下所有 `.json` / `.json.gz` 日志中，指定玩家名的对局结果。

##### `Stat.from_log(log: str, player_id: int) -> Stat`

作用：

- 直接从一份原始日志中统计指定 seat 的结果。

#### 普通方法

- `total_pt(pts: list[int]) -> int`
- `avg_pt(pts: list[int]) -> float`
- `__str__()`
- `__repr__()`

#### 可读写字段

基础计数：

- `game`
- `round`
- `oya`
- `point`
- `rank_1`
- `rank_2`
- `rank_3`
- `rank_4`
- `tobi`

副露 / 和牌 / 放铳相关：

- `fuuro`
- `fuuro_num`
- `fuuro_point`
- `fuuro_agari`
- `fuuro_agari_jun`
- `fuuro_agari_point`
- `fuuro_houjuu`
- `agari`
- `agari_as_oya`
- `agari_jun`
- `agari_point_oya`
- `agari_point_ko`
- `houjuu`
- `houjuu_jun`
- `houjuu_to_oya`
- `houjuu_point_to_oya`
- `houjuu_point_to_ko`

立直相关：

- `riichi`
- `riichi_as_oya`
- `riichi_jun`
- `riichi_agari`
- `riichi_agari_point`
- `riichi_agari_jun`
- `riichi_houjuu`
- `riichi_ryukyoku`
- `riichi_point`
- `chasing_riichi`
- `riichi_got_chased`

默听 / 流局 / 大牌：

- `dama_agari`
- `dama_agari_jun`
- `dama_agari_point`
- `ryukyoku`
- `ryukyoku_point`
- `yakuman`
- `nagashi_mangan`

#### 计算属性（getter）

排名相关：

- `avg_rank`
- `rank_1_rate`
- `rank_2_rate`
- `rank_3_rate`
- `rank_4_rate`
- `tobi_rate`

得点相关：

- `avg_point_per_game`
- `avg_point_per_round`
- `avg_point_per_agari`
- `avg_point_per_oya_agari`
- `avg_point_per_ko_agari`
- `avg_point_per_riichi_agari`
- `avg_point_per_fuuro_agari`
- `avg_point_per_dama_agari`
- `avg_point_per_ryukyoku`
- `avg_point_per_houjuu`
- `avg_point_per_houjuu_to_oya`
- `avg_point_per_houjuu_to_ko`

巡目相关：

- `avg_agari_jun`
- `avg_riichi_agari_jun`
- `avg_fuuro_agari_jun`
- `avg_dama_agari_jun`
- `avg_houjuu_jun`
- `avg_riichi_jun`

行为率相关：

- `agari_rate`
- `houjuu_rate`
- `riichi_rate`
- `fuuro_rate`
- `ryukyoku_rate`
- `agari_rate_after_riichi`
- `houjuu_rate_after_riichi`
- `chasing_riichi_rate`
- `riichi_chased_rate`
- `agari_rate_as_oya`
- `agari_as_oya_rate`
- `houjuu_to_oya_rate`
- `avg_fuuro_num`
- `agari_rate_after_fuuro`
- `houjuu_rate_after_fuuro`

其他：

- `avg_riichi_point`
- `avg_fuuro_point`
- `yakuman_rate`
- `nagashi_mangan_rate`

#### Python 侧使用

- [mortal/player.py](/home/tml104/Mortal/mortal/player.py)

具体用途：

- `TestPlayer.test_play()` 调用 `Stat.from_dir(...)`
- 读取各种统计指标写入 TensorBoard
- 用 `avg_rank` 和 `avg_pt(...)` 参与 best model 筛选

---

## 7. `libriichi.mjai`

注册位置：

- [libriichi/src/mjai/mod.rs](/home/tml104/Mortal/libriichi/src/mjai/mod.rs)

### 暴露类

- `Bot`

当前对 Python 公开的只有 `Bot`；`Event` 等结构在 Rust 内部可见，但没有注册到 Python 子模块里。

### `Bot`

定义位置：

- [libriichi/src/mjai/bot.rs](/home/tml104/Mortal/libriichi/src/mjai/bot.rs)

#### 构造方法

`Bot(engine, player_id: int)`

作用：

- 用一个 Python 侧 engine 对象和 seat id 构造 bot。

这里的 `engine` 需要满足 Rust 侧 agent 封装所期望的 Python 接口；在当前项目里传入的是：

- [mortal/engine.py](/home/tml104/Mortal/mortal/engine.py) 中的 `MortalEngine`

#### 方法

##### `react(line: str, *, can_act=True) -> Optional[str]`

作用：

- 输入一条 mjai JSON 事件字符串
- 更新内部状态
- 若当前可行动，则向 Python engine 请求反应
- 返回动作 JSON 字符串
- 若当前不能行动，则返回 `None`

额外规则：

- 如果 `can_act=False`
- 或输入 JSON 里 `can_act` 字段显式是 `false`
- 则 bot 只更新状态，不做决策

#### Python 侧使用

- [mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)

具体用途：

- 读取标准输入中的 mjai 事件流
- 调用 `bot.react(line)`
- 将返回的动作 JSON 输出到标准输出

---

## 8. Python 侧实际使用关系总表

下面这张表只看 `mortal/` 当前实际直接使用的接口。

### `libriichi.consts`

- `obs_shape`
  - 使用位置：[mortal/model.py](/home/tml104/Mortal/mortal/model.py), [mortal/train.py](/home/tml104/Mortal/mortal/train.py)
- `oracle_obs_shape`
  - 使用位置：[mortal/model.py](/home/tml104/Mortal/mortal/model.py)
- `ACTION_SPACE`
  - 使用位置：[mortal/model.py](/home/tml104/Mortal/mortal/model.py)
- `GRP_SIZE`
  - 使用位置：[mortal/model.py](/home/tml104/Mortal/mortal/model.py)

### `libriichi.dataset`

- `GameplayLoader`
  - 使用位置：[mortal/dataloader.py](/home/tml104/Mortal/mortal/dataloader.py)
- `Grp`
  - 使用位置：[mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py), [mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)

### `libriichi.arena`

- `OneVsThree`
  - 使用位置：[mortal/player.py](/home/tml104/Mortal/mortal/player.py), [mortal/one_vs_three.py](/home/tml104/Mortal/mortal/one_vs_three.py)

### `libriichi.stat`

- `Stat`
  - 使用位置：[mortal/player.py](/home/tml104/Mortal/mortal/player.py)

### `libriichi.mjai`

- `Bot`
  - 使用位置：[mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)

---

## 9. 一句话总结

`libriichi` 对 Python 暴露的接口，本质上覆盖了 Mortal 所需的整个“麻将环境与数据层”：

- `consts` 负责空间定义
- `state` 负责单 seat 状态机
- `dataset` 负责日志到样本的转换
- `arena` 负责对战环境
- `stat` 负责日志统计
- `mjai` 负责协议层 bot 封装

Python 侧主要负责神经网络、训练循环和参数管理，而不是自己重新实现麻将规则、日志解析或对战环境。
