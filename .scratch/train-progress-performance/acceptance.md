# 主策略优化链路验收记录

日期：2026-10-08。对应正式规格和已确认的 01–08 票据。

## 交付行为

- 独立新训练实现 Conv2d/channels_last、完整 batch 静态 fullgraph 编译/reduce-overhead/CUDA Graph，同权重 eager 尾 batch，FP16 AMP 和原有 Brain+DQN+AuxNet 目标。
- 旧预计算 CUDA checkpoint 显式迁移到新目录；模型、BN、AdamW、AMP、CPU/CUDA RNG 和样本进度保留，只有迁移前未输出指标窗口清空。缺少 CUDA 恢复历史不能作为本期完整 CUDA 续训来源。
- 保持旧 chunk/行乱序与按样本恢复，跨块组 batch；TensorBoard 累计 batch、50/500 周期、固定子集验证与 epoch 全量验证、原子提交。
- 立即刷新终端阶段日志，区分已处理与已保存位置，按本次增量计算整体/近期速度，验证/保存不计入近期训练时间；ETA 仅为当前阶段，统计不进入 checkpoint。
- 普通 mjai 与分批 Mortal 一对三均只加载新格式；推理默认 CPU，评测默认 GPU/AMP，报告独立目录、完整批次更新，可选牌谱。
- 公开操作及命令见[使用说明](../../docs2/optimized-main-model.md)。正式训练配置和源权重未被实验覆盖；未启动新的长期训练任务。

## 验证命令与结果

使用 `/home/tml104/miniconda3/envs/mortal/bin/python`，PWD 为 `~/Mortal`。GPU 验证在能够访问 CUDA 的受控执行环境运行。该仓库没有统一 Python 类型检查配置，本轮执行语法编译、模块/入口导入、TOML 解析与行为测试，不声称已完成独立静态类型检查。

- `python -m unittest discover -s mortal/precompute_pipeline/train_optimize/tests -p test_migration.py`：通过。独立旧模型对照前向、梯度和一次优化更新；核对非零 BN/AdamW 状态、CPU/CUDA RNG、原文件不变及目标保护。
- `python -m unittest discover -s mortal/precompute_pipeline/train_optimize/tests -p test_reader_and_settings.py`：3 项通过。新 reader 等于旧规则在不同样本游标后的后缀，覆盖跨 chunk、尾部、预取开关、独立配置，以及设备不支持时不清理旧输出的约束。
- `python -m unittest discover -s mortal/precompute_pipeline/train_optimize/tests -p test_training.py`：5 项 CUDA 回归通过（最终完整套件也覆盖）。包含真实更新/尾 batch、模型/优化恢复、TensorBoard purge、坏状态拒绝、验证和保存失败保留旧状态、即时日志，以及受控时钟的恢复增量/暂停区间/unknown ETA。
- `python -m unittest discover -s mortal/precompute_pipeline/inference/tests`：通过。匿名化真实开局事件产生合法 JSON 动作，不依赖 chunks 或训练配置；拒绝旧格式和无效玩家 ID。
- `python -m unittest discover -s mortal/precompute_pipeline/evaluation/tests`：2 项通过。控制 arena 结果核对连续 seed、末批、平均顺位/顺位点及失败后部分报告。
- 完整套件以全部 `mortal/precompute_pipeline/**/test_*.py` 的 21 个模块加载运行：**76 项，74 项通过，2 项按显式开关跳过**，耗时 12.821 秒。跳过的是旧三入口真实数据集成与新的手动真实验收；后者已另行显式运行通过。原始输出见[完整回归日志](acceptance/full-tests.log)。
- `MORTAL_RUN_OPTIMIZED_INTEGRATION=1 python -m unittest discover -s mortal/precompute_pipeline/train_optimize/tests -p test_real_pipeline.py`：最终 fullgraph 实现通过，耗时 96.225 秒。真实无损子集使用新生成 ID，原数据只读；192 通道/40 块、batch 2048、3 epoch，消费 12309 样本/9 批，7 样本尾批，固定验证 2051 样本及全量 4103 样本，至少三次验证切换，BN 只推进 9 次；保存恢复、CPU/GPU mjai、两个批次共 8 场 Rust 对局及 8 份牌谱都通过。临时模型/牌谱随测试结束移除，保留[结果摘要](acceptance/real-pipeline.json)及[日志](acceptance/real-pipeline.log)。
- `python -m compileall -q` 针对本轮新增 Python 模块：通过。新入口 `--help`、各示例 TOML 和文档命令/相对链接在最终检查中核对。

完整套件的真实验收开关保持关闭以避免普通回归启动较长对局。没有修改 Rust/PyO3 接口，本轮不运行完整 Rust 构建与 feature 测试矩阵。

## 性能与资源

硬件：Ryzen 7 9700X，8 核/16 线程；RTX 4090，24564 MiB；WSL 可见内存 16316452864 字节，约 15.2 GiB；PyTorch 2.11.0+cu128，驱动 595.97。

同一只读快照起点为 batch 2837000、累计样本 2905085576、epoch 游标 388696064。基线 batch 1024 处理 360 批，新 batch 2048 处理 180 批，均消费同一逻辑后缀的 368640 个真实样本，原模型、原 checkpoint 和 chunks 未写回。对应快照的所有 BN 计数与 optimizer step 都按实际 360/180 次增加，没有隐藏预热训练。

| 测量 | 基线 | 新组合首测 | 最终新组合（编译缓存） |
| --- | ---: | ---: | ---: |
| 稳定样本/秒 | 10568 | 18795 | 19184 |
| 整体样本/秒 | 8073 | 3407 | 12174 |
| 首批秒数 | 3.11 | 62.44 | 4.83 |
| 主机 RSS 峰值/GiB | 2.86 | 3.24 | 2.90 |
| CUDA allocated 峰值/GiB | 4.15 | 5.91 | 5.84 |
| CUDA reserved 峰值/GiB | 4.74 | 7.34 | 7.13 |
| 驱动显存观察/GiB | 6.37 | 9.00 | 8.79 |

稳定段对两侧均排除前 102400 条真实样本及显式验证/保存时间，后续测量 266240 个相同逻辑样本。整体从训练运行函数启动计到返回，包含初始化、验证和保存，不含此前快照/迁移和模块导入。驱动显存使用 CUDA `mem_get_info` 在日志边界观察，不能冒充连续 NVML 采样的严格峰值；allocated/reserved 为 PyTorch 峰值，RSS 为操作系统进程峰值。

最终稳定段约提升 82%，此有限任务整体约提升 51%。首测在启用 fullgraph 严格检查前进行，其冷启动时间只作为观察记录；最终新组合通过完整图的缓存复测和真实形状集成验收。顺序运行会受到文件/编译缓存和系统负载影响，这些结果不是固定吞吐、显存上限或长期效果承诺。

原始记录：[基线](acceptance/benchmark-baseline.json)、[新组合首测](acceptance/benchmark-optimized.json)、[最终缓存复测](acceptance/benchmark-optimized-warm.json)。新进程采用正常主种子/RNG/游标恢复，不沿用历史性能原型的固定 Torch seed 和额外训练预热。

## Standards

按 code-review 对本轮开始时的工作区快照审查新增源码，排除所有预先存在的改动。规范轴未发现硬性违规或需处理的代码异味。关键函数注释、安全权重加载、独立运行模块、临时测试产物和公开行为测试符合仓库约定。新旧 reader 的算法相似是独立链路加样本兼容的明确要求，不强行抽象为共用旧运行模块。

## Spec

初审发现两项 P2：CUDA RNG 缺字段时静默恢复，以及缺少受控时钟数值验收。现已将非空 CUDA RNG 纳入训练/迁移完整状态要求，无条件恢复，并增加坏状态回归；公开训练接口的受控时钟验证恢复增量、10 秒验证/1000 秒保存排除及零时长 unknown。审查代理复查确认两项解决，无遗留规格发现。

两个审查轴均为只读检查；实际测试结果由上述命令提供。

数值恢复对照使用 FP32，保证极短构造任务有实际 AdamW 历史；默认 FP16 由真实生产形状验收及 180 批同源测量验证。后续补充检查曾遇到正常 AMP 溢出造成构造任务全跳步，调整测试后重新完成上述全套回归。另补充设备不支持时先拒绝、后清理输出的顺序，文件保护用例和最终回归通过。

## 一次性代码清理

一次性基准包装脚本只用于本次三组有限测量，位于专用临时验收目录。必要方法、参数、环境、计数和结果已记录于本文件和 JSON；清理它及本轮临时快照/实验模型，保留结果与日志。该脚本不属于生产入口，也不作为正式复测命令依赖。

保留永久迁移工具、行为回归、可手动启用的真实数据验收、配置示例和使用说明。此前其他功能留下的历史实验代码、用户数据、正式模型和 TensorBoard 不纳入本次清理。清理后执行语法、配置、入口用法和引用检查；最终清理与范围检查结果在此记录。

清理已执行：一次性测量包装脚本、只读源快照的临时副本，以及三组实验模型目录均移除，具体见[清理清单](acceptance/cleanup.json)。原始生产 checkpoint 与源快照逐字节一致的核对已通过，见[源保留记录](acceptance/source-preserved.json)。正式运行不依赖这些临时文件。

## 未验证项

- 未运行整个 30 epoch 的长期训练、完整原始 val 的长期扫集或默认 400 场完整评测；已用有限真实子集和 8 场真实对局检验对应路径。
- 未证明新 batch 下的长期收敛、棋力改善或跨设备位级一致；本轮棋力接口验收不等于统计显著的强度比较。
- 未启动正式训练切换；用户可按使用说明显式选择最新源 checkpoint、迁移并启动。

最终清理后语法、三份独立示例配置和四个入口用法检查通过，见[检查日志](acceptance/post-cleanup-checks.log)。本轮实际改动逐项见[文件清单及范围核对](acceptance/scope-check.json)；原有无关源码/配置改动保持不变，8 项任务共 84 条验收标准均已完成。
