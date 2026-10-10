# 优化训练采用独立链路

用户在Q54指定将优化训练尽量独立实现在`mortal/precompute_pipeline/train_optimize/`，与当前`mortal/precompute_pipeline/train/`区分。此前的“保留原模式与优化模式”由两套独立链路承载，而不要求在原训练入口中增加统一模式切换。原链路暂时保留作为现有训练和对照，优化链路跑通后是否弃用原链路另行决定。

这一边界有利于优化链路独立演进，避免其后续维护绑定原训练流程。用户在Q59–Q62进一步确认：在`mortal/precompute_pipeline/inference/`和`evaluation/`放置与train_optimize平级的新入口；新训练不直接导入旧train目录的运行模块，原入口保留现状，本轮不向其回填日志。成熟数据集描述、codec、libriichi和MortalEngine可复用；新模型定义与checkpoint加载器集中于公共模块，由三个新入口共享。

第一期包含普通mjai推理和独立Mortal对Mortal一对三评测。复盘、Akochan、旧训练内置TestPlayer和在线TrainPlayer不纳入本期；训练内继续使用已确认的chunks离线验证。推理采用CPU/GPU可配置的普通eval/inference_mode前向，GPU AMP可配置，普通mjai默认CPU/FP32；不直接套用训练的固定batch CUDA Graph。

Q63将新推理/评测加载范围收敛为新Conv2d checkpoint，原格式只由既有链路或显式迁移入口处理。公共模型加载模块因此不承担旧推理格式兼容。Q64–Q66的评测默认值和结果保存约定记录在设计讨论中。

用户在Q55明确不需要为后台训练另外编写工具，因此本轮不增加专门的后台托管、监视或自动重启程序。迁移与启动采用正常命令；本决策不授权立即停止或切换运行中的训练。当前独立链路已实现并有限验收，见[操作说明](../optimized-main-model.md)和[验收记录](../../.scratch/train-progress-performance/acceptance.md)。原始选择见[设计讨论](../../.scratch/train-progress-performance/design-discussion.md)。
