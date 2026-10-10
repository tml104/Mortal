# 05: 使用新 Conv2d checkpoint 完成普通 mjai 推理

**What to build:** 使用者指定新格式 checkpoint、设备与玩家 ID 后，可以从 stdin 输入 mjai 事件并从 stdout 获得正常 JSON 动作响应，无需训练数据集或额外推理权重文件。

**Blocked by:** 01 — 显式迁移旧主策略 checkpoint 为原生 Conv2d 状态。

**Status:** ready-for-agent

实施状态：2026-10-08 已完成。

**规格覆盖：** I1、I5 的公共加载边界及 I9 的普通 mjai 推理。

- [x] 提供独立协议入口及必要 TOML 示例，显式读取配置和玩家 ID 0..3，从 checkpoint 元信息构造 Brain/DQN；只接受新 Conv2d 格式，旧格式或损坏状态直接报错。
- [x] 复用公共新模型/加载能力、MortalEngine 和既有 Rust Bot/PlayerState；不导入旧训练运行模块，不复制麻将规则或改变 v4 观测/动作协议。
- [x] 不实例化训练优化器、AuxNet 或数据集，不依赖训练 chunks、筛选或奖励配置，不要求另存推理权重；完整训练或迁移后的 checkpoint 均可作为模型来源。
- [x] 默认 CPU/FP32，设备与 GPU AMP 可配置，CPU 关闭 AMP；采用 eval/inference_mode 和 eager，根据实际请求动态 batch，不使用训练静态图策略。
- [x] 保持普通决策的贪心、quick_eval 与和牌规则保护；合法动作 mask 正常生效。
- [x] stdout 仅输出 mjai JSON 响应且及时 flush，诊断与错误走 stderr；复盘/GRP 输出不混入普通协议。
- [x] 用隔离的迁移产物和有限真实 mjai 事件流验证完整入口，检查协议可解析、玩家 ID、合法动作及缺少训练数据的运行；覆盖 CPU 路径和可用的 GPU/AMP 路径。
- [x] 提供从仓库根目录调用、设备选择和协议输入输出说明；不承诺旧格式加载或推理编译提速。

**独立验收：** 只依赖 01 生成的新 checkpoint 即可通过协议事件流完成动作响应，不需要等待新训练、终端日志或对局评测。

## Comments

2026-10-08：普通 mjai 在 CPU/GPU 上使用新格式产物通过，协议输出保持 JSON。具体命令、结果与限制见[验收记录](../acceptance.md)。
