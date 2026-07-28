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
| 神经网络、张量和版本差异 | [network.md](network.md) |
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

## Python 入口

| 文件 | 用途 |
| --- | --- |
| [`mortal.py`](../mortal/mortal.py) | 从 stdin 读取 mjai 事件，向 stdout 输出动作 |
| [`engine.py`](../mortal/engine.py) | 批量模型推理、合法动作屏蔽和采样 |
| [`model.py`](../mortal/model.py) | `Brain`、`DQN`、`AuxNet`、`GRP`、`GRP_Transformer` |
| [`train.py`](../mortal/train.py) | 主策略离线/在线训练 |
| [`train_grp.py`](../mortal/train_grp.py) | GRU 版顺位预测训练 |
| [`train_grp_transformer.py`](../mortal/train_grp_transformer.py) | Transformer 版顺位预测实验训练 |
| [`dataloader.py`](../mortal/dataloader.py) | 主策略日志加载、奖励计算和样本组装 |
| [`server.py`](../mortal/server.py) | 在线训练参数分发和 replay 缓冲 |
| [`client.py`](../mortal/client.py) | 拉取参数、自对战并上传 replay |
| [`player.py`](../mortal/player.py) | 训练和测试对局封装 |
| [`one_vs_three.py`](../mortal/one_vs_three.py) | 独立一对三评测 |

Python 文件采用同目录平铺导入，通常从仓库根目录执行脚本。不要把单个文件改成包式相对导入，除非同时迁移整个目录结构。

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
