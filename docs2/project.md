# Mortal 项目结构分析

## 1. 项目概览

这个仓库是一个围绕日式麻将 AI `Mortal` 构建的多语言项目，整体上由三层组成：

1. `libriichi/`：Rust 核心库，负责麻将规则、状态推进、数据集处理、自对战环境、mjai 接口等底层能力。
2. `mortal/`：Python 侧训练与推理层，负责神经网络定义、离线训练、在线训练、对战评测、模型加载与对局决策。
3. 辅助子项目：`docs/` 用于生成文档站，`log-viewer/` 用于查看 mjai 日志，`exe-wrapper/` 是一个较小的 Rust 可执行包装程序。

从顶层看，这不是一个单体应用，而是“Rust 引擎 + Python 模型/训练 + 文档/工具”的组合仓库。

## 2. 顶层目录职责

仓库主要目录如下：

```text
.
├── Cargo.toml                # Rust workspace 根配置
├── Cargo.lock
├── Dockerfile                # 运行 Mortal Python 入口的容器镜像
├── environment.yml           # Python/Conda 环境定义
├── README.md
├── docs/                     # mdBook 文档站
├── docs2/                    # 当前补充文档目录
├── exe-wrapper/              # Rust 可执行包装
├── libriichi/                # Rust 核心库 + Python 扩展模块
├── log-viewer/               # mjai 日志查看器
└── mortal/                   # Python 训练、推理、服务、评测代码
```

### 2.1 Rust workspace

根目录 [Cargo.toml](/home/tml104/Mortal/Cargo.toml) 定义了一个 workspace，目前包含两个成员：

- `libriichi`
- `exe-wrapper`

说明仓库的 Rust 部分是按工作区组织的，核心功能集中在 `libriichi`，其余 Rust crate 只是辅助。

### 2.2 Python 运行环境

[environment.yml](/home/tml104/Mortal/environment.yml) 说明 Python 侧依赖相对精简，主要包括：

- `python=3.12`
- `tqdm`
- `toml`
- `tensorboard`

但从代码实际使用来看，运行时还依赖 `torch` 和通过 `pyo3` 暴露出来的 `libriichi` Python 扩展，因此真实运行环境是 “PyTorch + 本地编译 Rust 扩展” 的混合环境。

### 2.3 Docker 支持

[Dockerfile](/home/tml104/Mortal/Dockerfile) 体现了项目的典型部署方式：

- 第一阶段编译 Rust 的 `libriichi` 动态库；
- 第二阶段安装 Python + PyTorch 运行环境；
- 将 `mortal/` 目录复制进容器；
- 以 `python mortal.py` 作为入口。

这说明生产/部署使用的核心运行单元其实是 Python 程序，但它依赖 Rust 编译产物作为底层能力。

## 3. `libriichi/`：Rust 核心库

`libriichi` 是整个项目最基础的部分。根据 [libriichi/src/lib.rs](/home/tml104/Mortal/libriichi/src/lib.rs)，它同时承担两个角色：

- 作为 Rust 库提供规则与算法能力；
- 通过 `pyo3` 编译成 Python 模块 `libriichi`，供 `mortal/` 直接导入。

### 3.1 crate 定位

[libriichi/Cargo.toml](/home/tml104/Mortal/libriichi/Cargo.toml) 中 `crate-type = ["cdylib", "rlib"]`，说明它既能被 Rust 代码链接，也能生成 Python 扩展动态库。

默认特性里启用了：

- `pymod`：支持 Python 扩展模块导出；
- `mimalloc`：自定义内存分配器。

### 3.2 主要模块划分

`libriichi/src/` 可按职责拆成几组：

- `state/`
  - 维护玩家状态、动作、观测表达、状态更新。
  - 是“事件驱动状态机”的核心。
- `mjai/`
  - 处理 mjai 协议事件与 bot 接口。
  - Python 侧 `mortal.py` 通过 `libriichi.mjai.Bot` 驱动对局交互。
- `dataset/`
  - 从 mjai 日志中提取训练样本。
  - 包含普通 gameplay 数据和 `grp` 相关数据。
- `arena/`
  - 自对战与评测环境。
  - 支持 `one_vs_three`、`two_vs_two` 等对战模式。
- `algo/`
  - 麻将底层算法，如和牌、向听数、点数计算等。
- `agent/`
  - 各类代理封装，包括 Mortal、自摸切、Akochan 等。
- 其他基础模块
  - `consts.rs`：动作空间、观测空间等常量定义。
  - `tile.rs`、`hand.rs`、`chi_type.rs`：麻将牌与手牌相关表示。
  - `stat.rs`：日志统计分析。

### 3.3 对 Python 层的意义

Python 侧大量逻辑不是自己解析麻将规则，而是直接建立在 `libriichi` 暴露的接口之上，例如：

- `libriichi.consts`：提供观测维度、动作空间；
- `libriichi.dataset`：提供日志数据读取与特征构建；
- `libriichi.mjai.Bot`：提供与 mjai 流交互的 bot 接口；
- `libriichi.arena.OneVsThree`：提供评测环境。

可以认为：`libriichi` 负责“麻将世界的规则与数据表达”，`mortal/` 负责“基于这些表达训练和运行神经网络”。

## 4. `mortal/`：Python 训练与推理层

`mortal/` 是项目最像“应用层”的目录。它不只是一个脚本集合，而是一组围绕模型训练、评测、在线更新和推理运行的 Python 模块。

### 4.1 关键文件职责

- [mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)
  - 推理入口。
  - 从模型权重文件加载 `Brain` 和 `DQN`，构造 `MortalEngine`，通过 `libriichi.mjai.Bot` 接收输入事件并输出动作。
- [mortal/train.py](/home/tml104/Mortal/mortal/train.py)
  - 主训练脚本。
  - 负责离线/在线训练、优化器与调度器管理、模型保存、测试对局、TensorBoard 记录。
- [mortal/server.py](/home/tml104/Mortal/mortal/server.py)
  - 在线训练协调服务。
  - 维护参数版本、接收客户端提交的对局日志、提供训练数据 drain 机制。
- [mortal/client.py](/home/tml104/Mortal/mortal/client.py)
  - 在线训练 worker。
  - 从服务端拉取最新参数，执行对战生成日志，再将 replay 提交回服务端。
- [mortal/one_vs_three.py](/home/tml104/Mortal/mortal/one_vs_three.py)
  - 1v3 评测脚本。
  - 用 challenger 对 champion 或 Akochan 进行评测。
- [mortal/train_grp.py](/home/tml104/Mortal/mortal/train_grp.py)
  - 训练 `GRP` 网络，处理顺位预测等额外任务。
- [mortal/model.py](/home/tml104/Mortal/mortal/model.py)
  - 定义主要神经网络结构：`Brain`、`DQN`、`AuxNet`、`GRP`。
- [mortal/engine.py](/home/tml104/Mortal/mortal/engine.py)
  - 模型推理封装。
  - 将观测转成张量，执行前向传播，按策略输出动作。
- [mortal/config.py](/home/tml104/Mortal/mortal/config.py)
  - 配置加载入口，从 `MORTAL_CFG` 或 `config.toml` 读取 TOML 配置。
- [mortal/config.example.toml](/home/tml104/Mortal/mortal/config.example.toml)
  - 全局配置样例，基本展示了训练、在线训练、评测、GRP 训练的完整参数面。

### 4.2 Python 层内部结构

可以把 `mortal/` 再划成四类：

#### A. 模型与推理

- `model.py`
- `engine.py`
- `reward_calculator.py`

这一层处理神经网络结构、动作价值估计、推理采样逻辑和奖励相关计算。

#### B. 训练与数据加载

- `train.py`
- `train_grp.py`
- `dataloader.py`
- `lr_scheduler.py`

这一层负责训练流程本身，包括：

- 数据集迭代与多进程加载；
- 学习率调度；
- 梯度缩放与优化；
- 模型 checkpoint 持久化。

#### C. 对战与评测

- `player.py`
- `one_vs_three.py`

这一层把模型封装成可参与自对战或评测的 player/engine。

#### D. 在线训练基础设施

- `server.py`
- `client.py`
- `common.py`

这一层实现参数分发、日志上传、简单消息协议与 replay 缓冲。

### 4.3 配置驱动很强

从 `config.example.toml` 可以看出，Python 层高度依赖配置：

- `control`：训练总控参数；
- `dataset`：日志数据路径与采样策略；
- `resnet`：模型结构参数；
- `optim`：优化器和 scheduler；
- `online`：在线训练服务器/客户端配置；
- `1v3`：评测配置；
- `grp`：顺位预测网络训练配置。

这意味着这个项目不是“硬编码流程”，而是通过配置切换不同训练和评测模式。

## 5. 关键运行链路

### 5.1 推理链路

推理主链路是：

1. `mortal/mortal.py` 加载配置和权重；
2. 创建 `Brain` + `DQN`；
3. 封装成 `MortalEngine`；
4. 再交给 `libriichi.mjai.Bot`；
5. Bot 从标准输入读取 mjai 事件流并输出动作 JSON。

这一链路说明：

- Python 负责模型；
- Rust `libriichi` 负责 mjai 协议状态维护；
- 二者在 bot 层完成集成。

### 5.2 离线训练链路

离线训练主链路大致是：

1. `train.py` 读取 `config.toml`；
2. 通过 `libriichi.dataset` 和 `dataloader.py` 从 gz 日志中构建样本；
3. 使用 `Brain`、`DQN`、`AuxNet` 做前向与反向；
4. 定期保存状态文件、写 TensorBoard、执行测试对局。

项目里的 `state_file`/`best_state_file` 是训练状态的核心持久化文件。

### 5.3 在线训练链路

在线训练由 `server.py` 和 `client.py` 协作：

1. `server.py` 持有当前参数版本；
2. `client.py` 向服务端请求最新参数；
3. client 本地打对局，生成日志；
4. client 把 replay 提交回 server；
5. `train.py` 在 online 模式下从 server drain 数据继续训练；
6. 新参数再被提交回 server，进入下一轮。

这是一个比较轻量的参数服务器模式。

### 5.4 评测链路

`one_vs_three.py` 使用 `libriichi.arena.OneVsThree`，让 challenger 与 champion 或 Akochan 进行多轮对战，统计 rank 和平均 pt。

因此评测并不是单纯离线打分，而是基于环境模拟的对局结果。

## 6. 其他子项目

### 6.1 `docs/`

`docs/` 是 mdBook 文档站源码，[docs/book.toml](/home/tml104/Mortal/docs/book.toml) 中可以看到：

- 文档工具是 `mdBook`；
- 使用了 `admonish`、`pagetoc` 等预处理/前端增强；
- `docs/src/` 中维护实际文档内容。

这个目录与模型训练无直接耦合，主要用于项目说明、用户指南和性能说明。

### 6.2 `log-viewer/`

`log-viewer/` 是 mjai 日志查看器，从 [log-viewer/README.md](/home/tml104/Mortal/log-viewer/README.md) 看，它是基于上游产物复制并修改的前端资源集合。

目录中主要是：

- `index.example.html`
- `files/js/`
- `files/css/`
- `files/images/`

它更像一个静态工具目录，而不是核心业务代码。

### 6.3 `exe-wrapper/`

`exe-wrapper/` 很小，只有一个 Rust crate，推测用途是将某些运行入口包装成更方便分发的可执行程序。它不是项目主逻辑中心。

## 7. 技术栈总结

这个仓库的技术栈比较清晰：

- Rust
  - 麻将规则与状态机
  - 数据集构建
  - 自对战环境
  - Python 扩展导出（`pyo3`）
- Python
  - PyTorch 模型定义与训练
  - 推理调度
  - 在线训练服务与客户端
- 文档与工具
  - mdBook
  - 简单静态前端日志查看器
- 容器与环境
  - Docker
  - Conda environment

## 8. 代码阅读建议

如果要理解这个项目，推荐按下面顺序读：

1. 先看 [README.md](/home/tml104/Mortal/README.md) 和 [mortal/config.example.toml](/home/tml104/Mortal/mortal/config.example.toml)，建立整体心智模型。
2. 再看 [libriichi/src/lib.rs](/home/tml104/Mortal/libriichi/src/lib.rs) 和 `libriichi/src/state/`、`libriichi/src/mjai/`，理解 Rust 层提供了什么。
3. 接着看 [mortal/model.py](/home/tml104/Mortal/mortal/model.py) 和 [mortal/engine.py](/home/tml104/Mortal/mortal/engine.py)，理解模型和动作选择。
4. 然后看 [mortal/mortal.py](/home/tml104/Mortal/mortal/mortal.py)，理解单局推理入口。
5. 最后看 [mortal/train.py](/home/tml104/Mortal/mortal/train.py)、[mortal/server.py](/home/tml104/Mortal/mortal/server.py)、[mortal/client.py](/home/tml104/Mortal/mortal/client.py)，理解训练体系。

## 9. 一句话结论

`Mortal` 本质上是一个“以 Rust 实现麻将规则和环境、以 Python/PyTorch 实现模型训练与推理”的混合架构项目；`libriichi` 是核心基础设施，`mortal/` 是上层 AI 工作流，而 `docs/`、`log-viewer/`、`exe-wrapper/` 则是围绕主系统提供文档、可视化和包装能力的辅助部分。
