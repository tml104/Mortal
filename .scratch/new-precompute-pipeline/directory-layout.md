# 按入口分组的目录设计

状态：用户已接受 Q30–Q32，确认按 `quarantine_logs`、`precompute`、`train` 分类，共用模块放入 `common`，继续直接文件路径启动，专属测试就近放置。已同步至主实施方案；未创建源码目录或开始实现。

本次讨论只涉及源码归属、导入和测试位置。已确认的样本语义、数据格式、C/D 数据目录、线程配置和恢复规则继续适用；训练入口与核心流程仍合并为一个 `train.py`。

## 已确认结构

```text
mortal/precompute_pipeline/
  config.example.toml
  common/
    settings.py
    codec.py
    dataset.py
    files.py
    tests/
  quarantine_logs/
    quarantine_logs.py
    log_validation.py
    tests/
  precompute/
    precompute.py
    samples.py
    storage.py
    tests/
  train/
    train.py
    reader.py
    tests/
  tests/                       # 跨阶段集成测试
```

图中省略空的包标记文件。它们仅为新链路提供明确的模块命名空间，不迁移现有 `mortal/model.py` 等平铺模块。顶层继续保留一份共享配置示例。

## 文件归属及共享依据

| 文件 | 职责 | 实际使用者 |
| --- | --- | --- |
| `common/settings.py` | 新配置的读取、默认值和必要检查 | 三个入口 |
| `common/codec.py` | 固定语义布局、格式/位宽约束、编码及解码 | 预计算和训练；筛选引用必要约束 |
| `common/dataset.py` | 预计算进度/完成记录的共享结构、生成 ID、chunk ID 区间及路径解析 | 预计算和训练 |
| `common/files.py` | 原子文件提交、统一的有序牌谱路径扫描等已有共用操作 | 三个入口按需调用 |
| `quarantine_logs/log_validation.py` | 基础规则与必要样本约束检查 | 筛选 |
| `quarantine_logs/quarantine_logs.py` | CLI、扫描、隔离、原因报告和筛选进度 | 筛选入口 |
| `precompute/samples.py` | Rust 批量提取、CPU GRP 奖励、六元样本构造 | 预计算 |
| `precompute/storage.py` | 两盘配额、目标盘选择、chunk 写入、预计算进度提交和未提交文件清理 | 预计算 |
| `precompute/precompute.py` | CLI、按文件序消费、固定样本 buffer 和整体恢复流程 | 预计算入口 |
| `train/reader.py` | 两级打乱、预取、设备 batch 解码编排和跨块组批 | 训练及其验证 |
| `train/train.py` | CLI、模型初始化、训练/验证、checkpoint 和指标 | 训练入口 |

这里的“共享”按整个入口链路是否使用来判断。例如 codec 的编码和解码分别由两个入口使用，但它们共同定义同一种磁盘格式，适合放在同一个公共模块中；无需要求每个函数都被多个入口调用。

`log_validation.py` 会检查将来预计算需要的输入，但运行时只由筛选调用，因此归筛选目录。训练和验证都属于 train 链路，二者使用同一个 reader 并不意味着 reader 要移入 common。

存储职责已明确拆开：写盘/配额在 `precompute/storage.py`，共享描述在 `common/dataset.py`，原子文件提交等小型操作在 `common/files.py`。common 中不放隔离流程、GRP 奖励计算或训练循环。

## 生产代码依赖方向

各入口目录可以依赖本目录模块和 common。入口目录之间不直接导入对方的私有模块；common 不反向导入入口目录。

```text
quarantine_logs ─┐
precompute ─────┼──→ common ──→ 标准库/现有依赖
train ──────────┘
```

入口链路按已确认范围使用 libriichi 与 `mortal/model.py`。common 的包初始化不主动导入全部子模块，也不创建模型、初始化 CUDA 或启动训练。

只有确认被多个入口使用的能力才提升到 common，不预先建设通用框架。跨阶段测试可引用多个入口，不受上述生产依赖方向限制。

## 启动与导入

继续从仓库根目录按文件路径启动：

```bash
python mortal/precompute_pipeline/quarantine_logs/quarantine_logs.py --config /absolute/path/precompute.toml
python mortal/precompute_pipeline/precompute/precompute.py --config /absolute/path/precompute.toml
python mortal/precompute_pipeline/train/train.py --config /absolute/path/precompute.toml
```

三个入口都先按自己的 `__file__` 定位并暴露 `mortal/` 搜索路径。只需少量启动代码，保留在各入口中，不单列公共启动脚本。

新模块用完整限定名导入，例如 `precompute_pipeline.common.codec`、`precompute_pipeline.train.reader`。既有模型继续按现有方式导入 `model`，libriichi 扩展仍从 `mortal/` 可见。

理由来自现有文件：裸 `import common`、`import train` 可能命中 `mortal/common.py`、`mortal/train.py`，前者还会导入旧配置。限定名称可以避免这类误导入。目录分类本身不要求改成 `python -m`，也不要求将旧项目迁移成包式导入。

`train/train.py` 同时保留命令行入口和核心训练流程，以普通函数及受主模块条件保护的 `main()` 组织。入口导入时不开始训练或清空目录。

## 测试位置

各目录的专属测试就近放在自身 `tests/`；公共格式、共享描述和文件提交工具的测试放在 `common/tests/`；跨筛选、生成、训练的集成测试放在链路根部 `tests/`。

测试位置不影响已经确认的验收内容，也不因此引入额外测试框架依赖。

## Q30–Q32 确认记录

- **Q30 已接受**：采用上述职责树，将私有写盘和共享描述分开；入口目录只依赖自身与 common。
- **Q31 已接受**：直接启动子目录中的同名脚本，新模块使用限定名称导入；入口保留少量定位搜索路径的启动代码。
- **Q32 已接受**：专属测试就近放置，跨阶段集成测试放根 tests。

主实施方案的模块表、固定格式定义位置和运行命令已统一更新。目录设计的确认不构成开始实现代码的授权。
