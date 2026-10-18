# 原生 Conv2d 主策略优化链路

本链路使用已有预计算 chunks，采用 Conv2d/channels_last、batch 2048、静态编译和 CUDA Graph。代码入口独立于原预计算训练；新模型保存原生二维权重。普通恢复只读取新格式，旧预计算 checkpoint 必须显式迁移。

以下命令的 PWD 均为 `~/Mortal`，并使用安装了 PyTorch 的 `mortal` Conda 环境。配置中的相对路径以 PWD 解析。当前实际训练配置继续由使用者管理，新示例不会覆盖它。

此前的旧训练模型已于 2026-10-08 完成首次配置与正式迁移，记录见[正式迁移结果](../.scratch/train-progress-performance/production-migration.md)。2026-10-14 按用户要求，当前实际配置改为学习率 `1e-5`，在 `artifacts/mortal_precomputed_optimize_261014` 从随机初始化开始新训练，batch 2048、总目标 30 epoch；旧模型保留在原目录。本次新训练的后续启动和恢复直接使用“启动训练”一节，下方首次准备与迁移命令保留为其他初次切换的参考，不用于本次新训练。

## 首次准备配置

进入目录和激活环境是新终端的准备步骤。配置模板只需在首次配置时复制；已有 `config.toml` 时直接使用并按需修改，后续启动或恢复不重复复制。

```bash
cd ~/Mortal
conda activate mortal
cp mortal/precompute_pipeline/train_optimize/config.example.toml mortal/precompute_pipeline/train_optimize/config.toml
```

检查新配置的数据集进度路径及目标产出目录。

## 旧模型的一次性迁移

仅在需要接续旧 Conv1d 模型时执行这一节。对选定的旧 checkpoint，迁移到新输出目录只做一次；成功迁移后，启动、中断恢复和后续 epoch 都直接读取新 checkpoint，不再次运行迁移。从头随机初始化新模型时跳过这一节。

迁移目标必须是 `artifacts` 下新的或空的专用子目录，配置和源权重放在目录之外。模型、损失、AdamW、AMP 和固定验证参数需要与源 checkpoint 对应；只通过迁移改变模型表示和 batch 分组。

```bash
python mortal/precompute_pipeline/train_optimize/migrate.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml" \
  --source-checkpoint "$PWD/artifacts/mortal_precomputed_v2/latest.pth"
```

该命令保留源文件，将完整恢复状态写入新目录的 `latest.pth`，不启动训练。只支持当前旧预计算 CUDA 训练的完整 checkpoint；缺少 CUDA 随机状态的 CPU 来源不能作为完整 CUDA 续训来源。迁移保留 BN、AdamW、AMP、随机数和训练游标，仅清空迁移前未输出的训练展示窗口；不提供逆向迁移。

选择源文件时，应以实际切换时最新完整提交为准。迁移保留样本顺序和已消费位置；batch 变化和浮点算子变化会改变后续优化轨迹，不保证与旧训练逐位相同。

首次开发验收中的迁移只发生在隔离目录；随后用户授权的正式迁移已完成，新产出可加载恢复。如果以后将迁移工具限定为本次切换用途，清理应安排在短期续训及保存后恢复验证完成之后；新配置、模型定义和 checkpoint 恢复能力仍是日常运行所需。

## 启动训练

```bash
python mortal/precompute_pipeline/train_optimize/train.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml"
```

默认设备 `cuda:0`、batch 2048、FP16 AMP、编译线程 2。目标 30 epoch 是累计总目标，迁移前已完成的轮次计入其中。小 batch 使用同一权重的 eager 路径，样本不会填充或丢弃。首次真实 batch 可能承担编译/捕获开销。

存在新 checkpoint 时从其中恢复；不存在时清理新训练专用目录的旧产物并随机初始化，但保留符合 `log-YYMMDD-hhmmss.log` 命名格式的历史日志文件。损坏或不兼容文件直接报错，不会按“从头训练”处理。模型、AdamW、AMP、CPU/CUDA RNG、种子、epoch 和样本游标共同提交。

每次启动自动在模型保存目录中建立日志，同时向原 stdout/stderr 输出。当前运行的日志为 `artifacts/mortal_precomputed_optimize_261014/log-261014-102028.log`；后续启动按新的启动时间命名。前台运行无需额外重定向；后台可使用普通 nohup：

```bash
nohup python mortal/precompute_pipeline/train_optimize/train.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml" &

# 查看当前配置的模型目录中最近更新的自动日志。
tail -f "$(ls -t artifacts/mortal_precomputed_optimize_261014/log-*.log | head -n 1)"
```

nohup 可能把终端副本追加到当前目录的 `nohup.out`，其中也能看到自动日志尚未建立前的启动错误；主要训练日志以模型目录的按运行文件为准。这里使用正常 shell 启动方式，没有额外托管或自动重启程序。中断后最多回退到上次完整 checkpoint；脚本不承诺收到终止信号时立即保存内存进度。

## 周期、指标与终端日志

默认每累计 50 个训练 batch 记录指标，每累计 500 个 batch 验证固定前 131072 条 val 样本并保存。验证 seed 为 0，顺序固定；小验证集使用全部样本。每个 epoch 结束执行全量 val；与周期重合时只做全量验证，不重复验证/保存。

TensorBoard 位于新输出目录的 `tensorboard` 子目录，横轴保持累计 batch，恢复 purge 边界为已保存 batch 加 1。迁移不减半历史 batch，也不把新 batch 大小乘以历史 batch 来估计样本数。比如从 123456 批迁移，下一日志和周期保存都是 123500。

文件名中的日期时间取本次启动时刻。文件和终端的每行以 `[YYMMDD-hh:mm:ss] ` 开头，固定使用北京时间 `Asia/Shanghai`、24 小时制；行时间取实际记录时刻。一次运行跨日仍写原文件，行日期正常更新；同一秒同一目录再次启动时追加同名文件，以 `start/resume` 区分运行。

训练进度仍为立即刷新的单行字段。`event` 包括 `start`/`resume`、`prepare`/`prepared`、`first_batch_start`/`first_batch_end`、`train`、`validation_start`/`validation`/`validation_end`、`checkpoint`、`epoch_complete` 和 `complete`。仅构造编译包装器时，首批编译仍是 pending；实际执行后才报告首批结束。有限验收接口停止时输出 `paused`，正常 CLI 按总目标运行。

```text
[261010-00:03:12] event=resume epoch=4 target_epochs=30 ...
[261010-00:03:20] event=first_batch_start ...
```

日志同时捕获本次训练会话中的 Python stdout/stderr，包括普通输出、Python logging 告警和多行异常堆栈，逐行添加时间；空行和分段文本也保留。绕过 Python 流直接写底层描述符的 Rust/C++ 或编译子进程输出不在捕获保证范围内。

日志创建、写入或必要刷新失败时直接终止，尽可能从原始终端显示错误；不猜测其他日志目录，也不增加重试、异步日志队列或逐行 fsync。日志关闭时还原原始输出流。北京时间仅用于展示，速度和 ETA 仍使用单调时钟；文本日志不进入 checkpoint，也不是恢复依据。

| 字段 | 含义 |
| --- | --- |
| epoch / target_epochs | 当前展示轮次（从 1 开始）与累计总目标 |
| epoch_samples / percent | 本轮已消费样本/总样本及百分比；100% 后仍可能在验证和保存 |
| samples / batches | checkpoint 历史加本次已处理的真实累计数量 |
| saved_batch | 最后成功提交的 batch；可能落后于当前 batches |
| elapsed_s | 本次运行耗时，不包含上次进程的耗时 |
| run_samples_s | 本次新增训练样本/整体耗时，包含准备、编译、验证和保存 |
| train_samples_s | 近期训练窗口速度，包含读取/解码/训练，排除验证/保存；首窗可能包含编译 |
| eta_s | 当前 epoch 数据处理或当前验证的估算剩余时间；无速度依据时为 unknown |
| validation_samples / kind | 本次验证局部进度与 periodic/epoch 类型，不推进训练游标 |
| loss / dqn_loss / cql_loss / rank_loss | 与 TensorBoard 一致的样本加权损失；训练是窗口均值，验证是本次累计均值 |
| rank_accuracy / q_mean / q_target_mean | 辅助顺位准确率、所选动作 Q 均值及目标 Q 均值 |

batch 数代表已处理批数；AMP 遇到溢出可能跳过参数更新，两者不总相等。实时速度和 ETA 不进入恢复文件。保存成功只在原子提交后输出；验证或保存失败会抛错并保留旧 checkpoint。阻塞编译内部没有后台心跳。

## 普通 mjai 推理

```bash
cp mortal/precompute_pipeline/inference/config.example.toml mortal/precompute_pipeline/inference/config.toml
python mortal/precompute_pipeline/inference/inference.py \
  --config "$PWD/mortal/precompute_pipeline/inference/config.toml" 0 < log.json
```

玩家 ID 为 0..3，输入为逐行 mjai JSON。默认 CPU/FP32；可改为 GPU 并启用 FP16 AMP。只从新 checkpoint 构造 Brain+DQN，采用普通 eval/inference_mode；无需训练 chunks、AuxNet 或优化器实例。stdout 只输出协议 JSON，错误走 stderr。第一期不支持复盘/GRP，也不加载旧 Conv1d 或原通用 Mortal 格式。

## 独立一对三评测

```bash
cp mortal/precompute_pipeline/evaluation/config.example.toml mortal/precompute_pipeline/evaluation/config.toml
# 先编辑配置，明确指定 challenger 与 champion 的新格式 checkpoint。
python mortal/precompute_pipeline/evaluation/evaluate.py \
  --config "$PWD/mortal/precompute_pipeline/evaluation/config.toml"
```

两边都必须是新 Conv2d checkpoint。可以比较两个新训练阶段，也可以先将旧预计算模型迁移到另一新目录作为对手；不需要额外导出权重。

默认 400 场半庄、每批 64 场、seed_start=10000、seed_key=0x2000。每个 seed 轮换四席产生四场；总场次和单批场次必须是 4 的正倍数，最后一批处理剩余种子。双方默认 cuda:0、FP16 AMP、贪心、quick_eval 与和牌规则保护，设备、AMP 和保护可分别配置，CPU 自动使用 FP32。

批内显示现有 Rust 进度条，每批打印挑战者排名、比例、平均顺位和按 `[90,45,0,-135]` 计算的平均顺位点。每次运行在配置根目录新建独立 `run-*` 子目录，逐批原子更新 `result.json`。中断只留下完整批次的 incomplete 报告，再次启动是新运行，没有自动断点续评。

`save_logs=false` 默认不写牌谱，开启时保存到本次运行的 `logs` 子目录。关闭写盘仍会有 Rust 内存牌谱成本。第一期不额外统计和率/放铳率；400 场只是有限初评默认，不保证区分细小棋力差异。固定种子也不保证不同设备、精度或分批方式逐局位级相同。

## 当前硬件测量

2026-10-08 在 Ryzen 7 9700X（8 核 16 线程）、RTX 4090（24,564 MiB）、WSL 可见约 15.2 GiB 内存、PyTorch 2.11.0+cu128、驱动 595.97 上有限验收。从同一第 2,837,000 批 checkpoint 快照开始，基线 360 批、新链路 180 批，均消费同一剩余序列的 368640 条真实样本；测试模型更新未写回原训练目录。

| 项目 | 基线 batch 1024 | 新组合 batch 2048（已有编译缓存） |
| --- | ---: | ---: |
| 稳定训练段样本/秒 | 10568 | 19184 |
| 整体样本/秒（含初始化、验证、保存） | 8073 | 12174 |
| 首批执行时间 | 3.11 秒 | 4.83 秒 |
| 主机峰值 RSS | 2.86 GiB | 2.90 GiB |
| PyTorch allocated 峰值 | 4.15 GiB | 5.84 GiB |
| PyTorch reserved 峰值 | 4.74 GiB | 7.13 GiB |
| 驱动显存观察（日志边界） | 6.37 GiB | 8.79 GiB |

稳定段排除相同的前 102400 条样本，以及显式验证/保存时间；整体计时从训练运行函数启动到返回，不含此前的快照与迁移。新链路稳定段约快 82%，此次有限任务的整体速度约快 51%。首次无缓存测量首批约 62 秒、整体约 3407 样本/秒，说明短任务的冷启动成本明显；该首测发生在收紧 fullgraph 检查之前，不能当成所有环境的固定冷启动时间。新组合当前要求完整图，编译错误直接终止。

RSS 是进程主机内存，reserved 是 PyTorch 缓存/图池，驱动观察通过 CUDA mem_get_info 取得，也包含上下文及其他显存使用；它不是连续 NVML 采样的严格峰值。这些不是同一指标，也不是长期上界。正式组合同时通过有限真实数据的训练/验证/尾 batch/恢复，以及 CPU/GPU mjai 和 8 场 Rust 一对三对局；完整长期训练的收敛和棋力仍需观察。

更多验证依据与清理记录见[本轮验收记录](../.scratch/train-progress-performance/acceptance.md)。历史临时实验不是正式运行依赖。
