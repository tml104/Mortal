# Mortal 项目协作指南

## 适用范围与信息来源

- 本文件适用于仓库根目录及全部子目录；如果子目录中以后出现更具体的 `AGENTS.md`，以更深层文件为准。
- 判断当前行为时，优先级为：源码与测试 > 实际配置 > `docs2/` 阅读笔记 > `docs/` 和 `README.md`。文档可能落后于代码，不能只根据文档推断接口。
- 开始修改前先查看 `git status` 和相关文件，保留用户已有改动，不顺手整理无关代码、配置或生成物。
- 推荐先阅读 `README.md` 了解项目定位，再按任务阅读 `docs2/project.md`、`docs2/network.md`、`docs2/train.md`、`docs2/mjai.md`、`docs2/libriichi_doc.md`。最终仍需回到对应源码核实。

## 项目定位与核心架构

Mortal 是一个基于深度强化学习的日式立直麻将 AI。项目采用 Rust 与 Python/PyTorch 混合架构，并以 mjai 的逐行 JSON 事件作为主要对局协议：

```text
mjai 日志/事件
  -> libriichi：规则、状态推进、观测与动作、数据提取、自对战
  -> PyO3 扩展模块 libriichi
  -> mortal：PyTorch 模型、训练、推理、评测和在线训练
```

主要模型链路如下：

- 普通决策：观测和合法动作掩码 -> `Brain` -> `DQN` -> mjai 动作。
- 主干训练：对局日志 -> `GameplayLoader`/`dataloader.py` -> `Brain + DQN + AuxNet` -> checkpoint 与 TensorBoard。
- `GRP`：根据按局推进的特征序列预测最终顺位；它单独训练，并在主干训练的数据加载阶段参与奖励估计。
- `GRP_Transformer`：顺位预测的 Transformer 实验实现，有独立训练脚本和配置；当前普通复盘模式仍使用 `GRP`，不要默认二者可以直接互换。
- 在线训练：`server.py` 分发参数和收集日志，多个 `client.py` 自对战，`train.py` 消费新日志并更新参数。

## 目录与关键入口

| 路径 | 职责 |
| --- | --- |
| `libriichi/` | Rust 核心库，同时构建为 PyO3 扩展模块和 Rust 库。 |
| `libriichi/src/state/` | mjai 事件驱动的玩家状态、动作合法性和观测编码。 |
| `libriichi/src/algo/` | 和牌、向听数、点数及牌效率算法。 |
| `libriichi/src/dataset/` | 从对局日志提取主训练样本、不可见信息和 GRP 特征。 |
| `libriichi/src/arena/` | 一对三、二对二自对战与评测环境。 |
| `libriichi/src/mjai/` | mjai 事件定义和 Bot 接口。 |
| `libriichi/src/agent/` | Mortal、Akochan、摸切等代理适配。 |
| `libriichi/src/consts.rs` | 观测形状、动作空间和 GRP 特征维度等跨语言契约。 |
| `mortal/model.py` | `Brain`、`DQN`、`AuxNet`、`GRP`、`GRP_Transformer`。 |
| `mortal/engine.py` | 推理、合法动作屏蔽和采样策略。 |
| `mortal/mortal.py` | mjai 推理入口，从 stdin 读取事件并向 stdout 输出动作。 |
| `mortal/train.py` | 主策略的离线/在线训练入口。 |
| `mortal/train_grp.py` | GRU 版顺位预测模型训练入口。 |
| `mortal/transformer_pipeline/` | 后续新增的 Transformer-GRP 训练、离线训练、预计算缓存与基准脚本。 |
| `mortal/server.py`、`mortal/client.py` | 在线训练的参数服务端和自对战客户端。 |
| `mortal/one_vs_three.py`、`mortal/player.py` | 强度评测和自对战。 |
| `exe-wrapper/` | Windows 分发所用的可执行包装器。 |
| `docs/` | 会由 mdBook 构建和部署的官方文档站源码。 |
| `docs2/` | 当前仓库补充的中文阅读笔记，不属于 mdBook 构建。 |
| `log-viewer/` | 从 mjai 项目复制并修改的静态日志查看器，不是核心训练代码。 |

## 环境与配置

- Rust 是 workspace，成员为 `libriichi` 和 `exe-wrapper`；使用当前 stable 工具链，代码采用 Rust 2024 edition，并依赖 `rustfmt`、`clippy`。
- `environment.yml` 创建 Python 3.12 Conda 环境。PyTorch 有意未固定在该文件中，需根据操作系统、GPU 和 CUDA 版本单独安装；Python 代码还会直接导入 NumPy。
- 项目不依赖数据库。训练和评测依赖外部 `.json.gz` 对局日志、`.pth` 权重、文件索引、TensorBoard 目录，在线训练还依赖配置中的 TCP 地址和缓冲目录。
- `mortal/config.py` 优先读取环境变量 `MORTAL_CFG`，否则读取“当前工作目录”下的 `config.toml`，并非相对脚本目录读取。执行命令时优先从仓库根目录运行，并为非默认配置使用绝对路径。
- `mortal/config.example.toml` 是主训练、推理和在线训练的完整配置参考，但其中路径多为占位值。根目录 `config.toml` 是当前实验配置，目前只包含 `grp` 和 `grp_transformer`，不能直接用于主干训练或普通推理。
- 常用环境变量：`MORTAL_CFG` 指定配置；`TRAIN_PLAY_PROFILE` 选择自对战配置（默认 `default`）；`MORTAL_REVIEW_MODE=1` 开启复盘附加输出并要求 GRP 权重。`MORTAL_IS_SUB_PROC` 由 `train.py` 内部管理，不要手动设置。
- Python 文件采用平铺式同目录导入。`mortal/transformer_pipeline/` 通过 `_bootstrap.py` 暴露原有 `mortal/` 模块，仍使用直接文件路径启动；除非任务明确要求包结构迁移，否则不要改成包式相对导入。

## 首次构建

```bash
conda env create -f environment.yml
conda activate mortal
# 按当前硬件环境另行安装 torch

cargo build -p libriichi --lib --release
cp target/release/libriichi.so mortal/libriichi.so
(cd mortal && python -c 'import libriichi; print(libriichi.__version__)')
```

Windows 构建时应将 `target/release/riichi.dll` 复制为 `mortal/libriichi.pyd`。修改 Rust 暴露给 Python 的接口后，必须重新构建并替换扩展文件；只执行 `cargo build` 不会更新 `mortal/` 中正在被 Python 导入的副本。

## 常用运行入口

以下命令通常从仓库根目录执行，并建议显式指定配置：

```bash
MORTAL_CFG=/absolute/path/config.toml python mortal/train_grp.py
MORTAL_CFG=/absolute/path/config.toml python mortal/transformer_pipeline/train_grp_transformer.py
MORTAL_CFG=/absolute/path/config.toml python mortal/train.py
MORTAL_CFG=/absolute/path/config.toml python mortal/mortal.py 0 < log.json
```

玩家 ID 必须在 `0..3`。mjai 输入通常是一行一个 JSON 事件，事件顺序、牌字符串、`actor`/`target` 含义属于协议契约，修改时参考 `libriichi/src/mjai/event.rs`、`libriichi/src/tile.rs` 和 `docs2/mjai.md`。

## 编码与修改要求

- 新增或修改函数时，如果不会破坏公共协议或上游文档风格，尽可能添加简洁的中文函数级注释。注释应解释业务规则、张量形状、状态约束或不直观的设计原因，不要逐行复述代码。
- 协议字段、API 名称、模型类名、麻将领域固定术语和错误消息可保留英文；不要为了满足中文注释要求而翻译机器可读标识符。
- Rust 代码遵循默认 `rustfmt`，并受 `libriichi/src/lib.rs` 中严格的 Clippy 规则约束。新增依赖时同步更新并提交根目录 `Cargo.lock`。
- 仓库没有统一的 Python formatter、lint、类型检查或 pytest 配置。Python 修改应保持邻近代码的四空格缩进、命名和局部排版，避免全文件格式化。
- 保持 `torch.load(..., weights_only=True)` 的安全加载约定；不要恢复为不受限的 pickle 加载。
- 修改模型结构、输入维度、checkpoint 字段或配置键时，要考虑已有权重的兼容性。无法兼容时必须明确迁移方式或在汇报中说明。
- 修改观测编码、动作编号/mask、`ACTION_SPACE`、`GRP_SIZE`、mjai 事件或 PyO3 导出时，必须同时检查 Rust 生产端和 Python 消费端，重点包括 `libriichi/src/consts.rs`、`state/`、`dataset/`、`mortal/model.py`、`mortal/engine.py`、`mortal/dataloader.py`。
- 算法或状态机行为变化应在相邻 Rust 模块添加回归测试；不要只依赖一次完整训练来判断正确性。
- `docs/` 与 `docs2/` 定位不同。修改行为或配置后同步更新相关说明，但不要把 `docs2/` 当作自动部署文档。
- 个人新增文档与工程技能配置统一放在 `docs2/`；技能模板中的 `docs/` 路径在本项目中改为 `docs2/`（例如 `docs2/agents/`、`docs2/adr/`）。`docs/` 保留为上游官方文档目录，除非任务明确要求修改上游文档，否则不向其中新增或迁移个人文档。已约定的根目录 `AGENTS.md`、单一 `CONTEXT.md` 和本地任务目录 `.scratch/` 保持各自位置。

## 数据、生成物与受保护内容

- `data/`、`artifacts*/`、`target/`、`__pycache__/`、模型权重、TensorBoard 日志、编译出的 `*.so`/`*.pyd` 等是用户数据或生成物。不要提交、清理、移动或覆盖，除非任务明确要求并已确认具体目标。
- 根目录 `config.toml` 含当前实验超参数和路径。修改它会改变真实训练行为，不要用示例配置覆盖，也不要在无关任务中重排。
- `libriichi/src/algo/data/*.bin.gz` 和 `uradora_prob_table.txt` 是被 Git 跟踪且在编译期使用的查表数据，不是普通缓存；修改必须有明确来源并运行相关算法测试。
- `log-viewer/files/` 和 `docs/css/mdbook-admonish.css` 属于 vendored/generated 内容；除非任务直接涉及其来源或功能，否则不要手工批量改写。
- 不执行 CI 中的部署、`gh-pages` 强制推送或其他发布步骤；本地只验证构建结果。

## 高风险运行边界

- 不要把训练、在线服务或长时间评测当作普通冒烟测试。它们可能占用大量 GPU/CPU、读取数 GB 数据并写入 checkpoint、文件索引和 TensorBoard 日志。
- 未经用户明确要求且未核对配置路径，不要运行 `mortal/server.py`：它启动时会递归删除并重建配置中的 `buffer_dir` 和 `drain_dir`。
- 同样不要擅自运行 `client.py`、`one_vs_three.py` 或会创建 `TestPlayer`/`TrainPlayer` 的流程；对战初始化可能清空配置中的 `log_dir`。
- Docker 镜像只面向 mjai 推理，不面向训练；构建或运行容器也不属于常规验证。

## 验证要求

根据改动范围选择最小充分验证，并在最终汇报中列出实际执行的命令和结果：

### Rust 与 `libriichi`

优先对齐 CI：

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -Dwarnings
cargo test --workspace --no-default-features --features flate2/zlib -- --nocapture
cargo test --workspace --no-default-features --features flate2/zlib,sp_reproduce_cpp_ver -- --nocapture
cargo test -p libriichi --no-default-features --benches
```

小范围改动可先运行目标测试；涉及共享状态、算法、feature 或跨语言契约时，再运行完整矩阵。涉及构建目标时补充：

```bash
cargo build -p libriichi --lib
cargo build -p libriichi --bins --no-default-features
cargo build -p exe-wrapper
```

### Python 与模型

- 当前没有自动化 Python 测试套件，至少对改动文件做语法检查，并在依赖齐全时执行目标模块导入或小型 CPU 张量前向验证。
- 涉及 PyO3 接口时，重新构建/复制 `libriichi` 后从 `mortal/` 验证导入和受影响 API。
- 涉及配置时验证 TOML 可解析，并核对目标脚本实际读取的全部键。
- 完整训练、GPU 性能、真实数据集和 checkpoint 兼容性若未验证，必须明确列为未验证项。

### 文档

修改 `docs/` 时，在安装 `mdbook`、`mdbook-admonish`、`mdbook-pagetoc` 后执行：

```bash
cd docs
mdbook build
```

只修改 `docs2/`、`README.md` 或 `AGENTS.md` 时不需要伪装成已运行代码测试，应检查 Markdown 结构、命令和路径是否与源码一致。

## 工作完成前检查

- 查看最终 `git diff`，确认只包含任务范围内的修改，没有模型、数据、配置或格式化噪音。
- 确认新增说明与当前源码一致，命令注明正确工作目录和前置条件。
- 明确区分“已验证”“因缺少 GPU/数据/权重未验证”和“未执行以避免破坏数据”。

## 如何与我沟通输出

每次修改完成后，最终回复必须使用以下三个部分，并用中文汇报：

### 总结变更摘要

- 说明做了什么、为什么这样改，以及实际执行的验证及结果。

### 所有改动的文件列表

- 逐项列出本次实际改动的文件及用途；不要把只读检查过的文件算作改动。

### 潜在的风险或未验证项

- 说明未执行的测试、缺失的环境/数据/权重、兼容性风险；如果没有，也明确写“无”。

## Agent skills

### Issue tracker

任务和规格保存在本地 `.scratch/<feature>/`。创建、读取或更新任务前，阅读 `docs2/agents/issue-tracker.md`。

### Triage labels

分诊使用五种标准角色；分诊任务时，阅读 `docs2/agents/triage-labels.md`，将对应标签写入本地任务的 `Status:` 字段。

### Domain docs

采用 single-context：根目录 `CONTEXT.md` 与 `docs2/adr/`。探索领域、使用领域术语或记录架构决策前，阅读 `docs2/agents/domain.md`。
