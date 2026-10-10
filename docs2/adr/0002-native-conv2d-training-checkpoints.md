# 优化训练直接持久化原生二维模型状态

在主策略训练优化的Q45/Q46中，用户决定采用高度为1的Conv2d与channels_last执行表示，并直接保存优化后模型的原生权重及对应优化器状态。保存时不将这些张量还原成Conv1d表示，也不在每个checkpoint中附带一份旧结构模型。这样使持久化结构与实际训练结构一致，代价是原Brain及旧训练入口不能直接加载新二维权重。

旧训练成果通过显式迁移进入新产出目录，保留原目录用于对照；迁移保留已学习参数、BN统计、优化器与AMP状态、必要随机数状态及样本进度。用户在Q53及Q59–Q62确认本轮提供读取新模型的普通mjai推理与独立Mortal对Mortal一对三评测入口，分别放在inference和evaluation目录；不以此推断需要旧格式导出。Q54接受本轮只提供旧→新的迁移，不提供逆向恢复，并要求优化训练实现为独立的train_optimize链路，见[ADR 0003](0003-independent-optimized-training.md)。

用户在Q63进一步限定：新的推理、评测仅加载本优化链路的Conv2d checkpoint，评测双方都使用新格式，不增加旧Conv1d或原通用Mortal格式的运行时加载分支。需要与迁移前模型比较时，可以将已支持的旧预计算checkpoint显式迁移为新格式后作为对手；这复用既定迁移流程，不扩大迁移源范围，也不要求自动生成额外权重副本。

本决策已由用户确认并实现，使用与兼容边界见[优化主策略说明](../optimized-main-model.md)，有限验收见[验收记录](../../.scratch/train-progress-performance/acceptance.md)。它不改变[ADR 0001](0001-frozen-precomputed-samples.md)中的固定六元样本和chunk语义，无需因此重新生成数据。讨论历史见[训练优化设计记录](../../.scratch/train-progress-performance/design-discussion.md)。
