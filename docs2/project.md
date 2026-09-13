# Mortal 项目导览

Mortal 是一个基于深度强化学习的日式立直麻将 AI。理解仓库时先抓住一条主线：

```text
mjai 事件流
  -> libriichi：规则、状态、观测、动作、数据提取与对战环境
  -> Python 扩展模块 libriichi
  -> mortal：PyTorch 模型、训练、推理与评测
```

Rust 层负责“麻将规则和数据表达”，Python 层负责“如何学习和决策”。

## 文档导航

| 想了解什么 | 阅读文档 |
| --- | --- |
| 仓库整体结构和入口 | 本文 |
| 模型的数据流、结构和版本差异 | [network.md](network.md) |
| 深度学习基础概念与用到的网络介绍 | [deep_learning_basics.md](deep_learning_basics.md) |
| 主策略离线/在线训练 | [train.md](train.md) |
| GRP 的前缀样本和变长序列 | [train_grp_1.md](train_grp_1.md) |
| 主干训练奖励如何生成 | [rewards.md](rewards.md) |
| mjai JSON 协议 | [mjai.md](mjai.md) |
| `libriichi` Python API | [libriichi_doc.md](libriichi_doc.md) |

`docs2/` 是中文阅读笔记，不参与 `docs/` 的 mdBook 构建。接口和行为发生冲突时，以源码和测试为准。

## 核心目录

| 路径 | 职责 | 关键入口 |
| --- | --- | --- |
| `libriichi/` | Rust 核心库和 PyO3 扩展 | [`src/lib.rs`](../libriichi/src/lib.rs) |
| `libriichi/src/state/` | 事件驱动状态机、合法动作、观测编码 | [`player_state.rs`](../libriichi/src/state/player_state.rs) |
| `libriichi/src/algo/` | 和牌、向听数、点数和牌效率算法 | [`mod.rs`](../libriichi/src/algo/mod.rs) |
| `libriichi/src/dataset/` | 从日志提取主干和 GRP 样本 | [`gameplay.rs`](../libriichi/src/dataset/gameplay.rs) |
| `libriichi/src/arena/` | 一对三、二对二对战环境 | [`mod.rs`](../libriichi/src/arena/mod.rs) |
| `libriichi/src/mjai/` | mjai 事件和 Bot 接口 | [`event.rs`](../libriichi/src/mjai/event.rs) |
| `mortal/` | 模型、训练、推理、评测和在线训练 | [`model.py`](../mortal/model.py) |
| `docs/` | 官方 mdBook 文档站源码 | [`book.toml`](../docs/book.toml) |
| `log-viewer/` | 静态 mjai 日志查看器 | [`README.md`](../log-viewer/README.md) |
| `exe-wrapper/` | Windows 分发用 Rust 包装器 | [`main.rs`](../exe-wrapper/src/main.rs) |

根目录 Rust workspace 包含 `libriichi` 和 `exe-wrapper`。`libriichi` 同时生成 Rust 库和 Python 动态扩展；Python 代码并没有重新实现麻将规则。

## Python 脚本与模块

以下文件既包括可直接运行的入口，也包括被入口复用的模型、数据和基础设施模块。名称中带有 `grp_transformer` 的主策略训练脚本，是“使用冻结的 Transformer GRP 计算奖励”，不是继续训练 GRP Transformer 本身。

### 推理与模型

| 文件 | 用途 |
| --- | --- |
| [`mortal.py`](../mortal/mortal.py) | mjai Bot 推理入口：从 stdin 逐行读取事件、更新玩家状态并向 stdout 输出动作；复盘模式还会加载 GRU 版 `GRP` 输出顺位信息 |
| [`engine.py`](../mortal/engine.py) | `MortalEngine` 批量推理：执行 `Brain + DQN`、合法动作屏蔽、贪心或 Boltzmann/top-p 采样；另含用于对战接口的摸切示例引擎 |
| [`model.py`](../mortal/model.py) | 定义共享网络组件以及 `Brain`、`DQN`、`AuxNet`、GRU 版 `GRP` 和实验性 `GRP_Transformer` |

### 训练、数据与预计算

后续新增的 Transformer-GRP 实验链路集中在 `mortal/transformer_pipeline/`，与原始
Mortal 入口分开。该目录仍按文件路径直接启动脚本，不使用 `python -m` 包式入口。

| 文件 | 用途 |
| --- | --- |
| [`train.py`](../mortal/train.py) | 通用主策略训练入口：支持离线日志训练和在线 replay 训练，更新 `Brain + DQN + AuxNet`，并使用冻结的 GRU 版 `GRP` 估计局间奖励 |
| [`dataloader.py`](../mortal/dataloader.py) | 通用主策略流式数据集：调用 Rust `GameplayLoader` 解析牌谱，使用 GRU 版 `GRP` 计算奖励并组装逐动作训练样本 |
| [`reward_calculator.py`](../mortal/reward_calculator.py) | 把 GRP 的最终顺位概率转换为相邻局之间的期望 pt 变化，也提供实际点棒变化计算 |
| [`train_grp.py`](../mortal/train_grp.py) | 训练 GRU 版顺位预测模型：构造对局历史前缀、执行训练与验证并保存可恢复 checkpoint |
| [`train_grp_transformer.py`](../mortal/transformer_pipeline/train_grp_transformer.py) | 训练实验性 Transformer 顺位预测模型：使用 padding mask 处理变长历史前缀，并独立保存模型和优化器状态 |
| [`dataloader_grp_transformer.py`](../mortal/transformer_pipeline/dataloader_grp_transformer.py) | Transformer-GRP 主策略数据集：复用通用加载器，以冻结的 `GRP_Transformer` 计算奖励，并让同一牌谱四个玩家视角共享一次 GRP 前向结果 |
| [`train_offline_grp_transformer.py`](../mortal/transformer_pipeline/train_offline_grp_transformer.py) | 使用 Transformer GRP 奖励的首个完整离线 epoch 训练器：按文件分片训练，支持严格训练契约、分片 checkpoint、断点恢复和完成状态校验 |
| [`train_offline_grp_transformer_bs1024.py`](../mortal/transformer_pipeline/train_offline_grp_transformer_bs1024.py) | 上述离线训练器的 batch size 1024 专用启动包装器：固定输出目录，提前设置 Rayon 线程和随机种子，并记录 CUDA 峰值显存；不包含另一套训练循环 |
| [`precompute_grp_transformer_cache.py`](../mortal/transformer_pipeline/precompute_grp_transformer_cache.py) | 预计算缓存生产入口：提前完成牌谱解析、observation 构造、Transformer GRP 推理和奖励计算，按 chunk 压缩落盘并生成可恢复进度、校验记录和最终 manifest |
| [`precomputed_cache.py`](../mortal/transformer_pipeline/precomputed_cache.py) | 预计算缓存格式与读取模块：按 observation plane 分类编码，使用 bit-pack、float16 和 Zstd 压缩，并在 CPU 或 GPU 上解码为训练 batch |
| [`train_precomputed_cache.py`](../mortal/transformer_pipeline/train_precomputed_cache.py) | 预计算缓存专用训练入口：从完整 manifest 确定性打乱并恰好遍历缓存一次，复用 Transformer-GRP 离线训练器的模型和单 batch 优化逻辑，使用独立 checkpoint schema 支持恢复 |

### 评测与性能基准

| 文件 | 用途 |
| --- | --- |
| [`one_vs_three.py`](../mortal/one_vs_three.py) | 独立一对三强度评测入口：加载挑战者与基线模型，通过 Rust arena 运行对局并保存日志 |
| [`player.py`](../mortal/player.py) | 封装定期评测和在线训练自对战：`TestPlayer` 对固定基线评测，`TrainPlayer` 按 profile 生成带探索的 replay 日志 |
| [`benchmark_offline_grp_transformer_bs1024.py`](../mortal/transformer_pipeline/benchmark_offline_grp_transformer_bs1024.py) | 传统 Transformer-GRP DataLoader 供数基准：比较 worker 数、每批文件数和 Rayon 线程数组合，同时采集吞吐、CPU 与内存指标 |
| [`benchmark_precomputed_cache.py`](../mortal/transformer_pipeline/benchmark_precomputed_cache.py) | 预计算缓存基准：分别测量 CPU 解码、GPU 供数和包含前向/反向/优化器更新的端到端训练吞吐，不回写输入 checkpoint |

### 在线训练与公共基础设施

| 文件 | 用途 |
| --- | --- |
| [`server.py`](../mortal/server.py) | 在线训练 TCP 服务端：保存最新参数、接收客户端 replay，并把缓冲日志批量转交训练进程 |
| [`client.py`](../mortal/client.py) | 在线自对战客户端：拉取最新参数、调用 `TrainPlayer` 生成牌谱，并把 replay 上传给服务端 |
| [`common.py`](../mortal/common.py) | 共享工具：TCP 消息封包/读取、在线训练参数提交和 replay drain、进度条与参数统计 |
| [`config.py`](../mortal/config.py) | 配置加载模块：优先读取 `MORTAL_CFG` 指定的 TOML，否则读取当前工作目录的 `config.toml` |
| [`lr_scheduler.py`](../mortal/lr_scheduler.py) | 实现线性 warm-up 后接余弦退火的学习率调度器，并支持步数偏移和按 epoch 周期重置 |
| [`prelude.py`](../mortal/prelude.py) | 进程初始化辅助模块：配置 stderr 日志和 stdin 编码，并集中处理部分 NumPy/PyTorch 警告 |

Python 文件保持平铺导入，通常从仓库根目录执行脚本；实验目录通过 `_bootstrap.py`
查找原有 `mortal/` 模块。不要把单个文件改成包式相对导入，除非同时迁移整个目录
结构。预计算训练仍属于实验链路；它的缓存和 checkpoint schema 与原始文件分片训练器
相互独立，不能直接混用恢复状态。

## 四条关键链路

### 推理

```text
stdin 中的 mjai 事件
-> libriichi.mjai.Bot 更新 PlayerState
-> encode_obs() 生成 obs 和合法动作 mask
-> Brain 编码状态
-> DQN 计算 46 个动作的 Q 值
-> MortalEngine 贪心或采样
-> Bot 转回 mjai 动作 JSON
-> stdout
```

普通推理只需要 `Brain + DQN`。`MORTAL_REVIEW_MODE=1` 才会额外加载 GRU 版 `GRP` 输出顺位概率。

### GRP 训练

```text
mjai .json.gz 日志
-> libriichi.dataset.Grp
-> 按 start_kyoku 提取 7 维序列
-> 展开为所有历史前缀
-> GRP 或 GRP_Transformer
-> 预测四家最终顺位的 24 种排列
```

两种模型任务相同，但结构、配置节和 checkpoint 不同。当前主干数据加载与复盘仍固定使用 GRU 版 `GRP`。

### 主策略离线训练

```text
日志
-> GameplayLoader 提取 obs/action/mask 等逐动作数据
-> 已训练的 GRP 估计按局奖励
-> dataloader.py 组装样本
-> Brain + DQN + AuxNet
-> checkpoint、TensorBoard、定期一对三评测
```

主策略训练依赖一个现成的 GRU 版 GRP checkpoint，但不会更新 GRP 参数。

### 主策略在线训练

```text
train.py --提交参数--> server.py
client.py --拉取参数--> 自对战 --上传日志--> server.py
train.py --drain 日志--> 更新主策略 --再次提交参数
```

这是轻量参数服务器流程，不是单进程端到端环境训练。具体启动顺序和目录删除风险见 [train.md](train.md)。

## 配置与运行环境

[`config.py`](../mortal/config.py) 按以下顺序选择配置：

1. 环境变量 `MORTAL_CFG` 指向的文件；
2. 当前工作目录下的 `config.toml`。

路径不是相对 `config.py` 解析的。因此推荐从仓库根目录运行，并为自定义配置使用绝对路径：

```bash
MORTAL_CFG=/absolute/path/config.toml python mortal/train_grp.py
```

需要区分两份配置：

- [`mortal/config.example.toml`](../mortal/config.example.toml) 展示主策略、GRP、在线训练和评测的完整配置面，但路径是占位值；
- 根目录 [`config.toml`](../config.toml) 是当前实验配置，只包含 `grp` 和 `grp_transformer`，不能直接运行主策略训练或普通推理。

Python 环境由 [`environment.yml`](../environment.yml) 起步，但 PyTorch 需按操作系统、GPU 和 CUDA 版本单独安装。项目还直接依赖 NumPy 和本地编译的 `libriichi` 扩展。

## 首次构建

```bash
conda env create -f environment.yml
conda activate mortal

cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
(cd mortal && python -c 'import libriichi; print(libriichi.__version__)')
```

运行构建命令前，需另行安装与当前硬件匹配的 PyTorch。

Windows 应把 `target/release/riichi.dll` 复制为 `mortal/libriichi.pyd`。修改 PyO3 接口后，必须重新构建并替换 `mortal/` 下的扩展副本；只运行 `cargo build` 不会更新 Python 实际导入的文件。

## 推荐阅读顺序

1. 本文：先建立目录和数据流心智模型。
2. [mjai.md](mjai.md)：理解系统的输入协议。
3. [network.md](network.md)：理解 `obs -> Brain -> DQN -> action`。
4. [train_grp_1.md](train_grp_1.md) 和 [rewards.md](rewards.md)：理解主干奖励来源。
5. [train.md](train.md)：理解训练执行和配置。
6. [libriichi_doc.md](libriichi_doc.md)：需要调用 Rust 扩展时再查 API。

随后回到对应源码核对细节。最重要的跨语言契约集中在 [`consts.rs`](../libriichi/src/consts.rs)、[`model.py`](../mortal/model.py)、[`engine.py`](../mortal/engine.py) 和 [`dataloader.py`](../mortal/dataloader.py)。
