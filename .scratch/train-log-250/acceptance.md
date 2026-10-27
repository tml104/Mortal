# 优化训练 event=train 每 250 批输出：验收

日期：2026-10-26。依据用户确认的实施计划，代码修改和正式训练切换均已完成。

## 行为

- `event=train` 固定按累计 batch 的 250 倍数输出，不在 epoch 结束或有限运行暂停时补报。
- TensorBoard 与验证进度沿用 `log_every_batches`，当前为 50；周期验证和 checkpoint 保存仍为 500 批。
- 日志复用当前指标窗口，当前配置下对应最近 50 批，与同一步 TensorBoard 数值一致。训练速度覆盖两条日志间的训练时间，并排除验证、保存；恢复后的首个速度窗口只包括本次运行。
- 未新增配置、指标累加器或 checkpoint 字段。现有学习率 `1e-5`、batch 2048、30 epoch 和模型输出目录保持一致。

## 回归与检查

PWD 为 `~/Mortal`，Python 使用 `/home/tml104/miniconda3/envs/mortal/bin/python`，CUDA 回归使用 `TORCHINDUCTOR_CACHE_DIR=/tmp/mortal-optimize-inductor`。

1. 新增公开训练入口用例先在旧实现上失败：第 50、100、150、200 批及第 249 批暂停时出现了不符合新约定的 `event=train`。见 [red-test.log](red-test.log)。
2. 修改后 `python -m unittest discover -s mortal/precompute_pipeline/train_optimize/tests -p test_training.py`：8 项通过，见 [training-tests.log](training-tests.log)。
3. 补充变速时钟断言后，以 unittest 加载 `precompute_pipeline.train_optimize.tests` 下的 `test_training`、`test_run_logging`、`test_reader_and_settings` 三个模块：23 项全部通过，耗时 24.354 秒，见 [final-tests.log](final-tests.log)。
4. 新用例实际完成小模型 502 批训练：249 批暂停、恢复至 500 批、继续完成 epoch 尾批。检查只有 250/500 两条训练日志、TensorBoard 每 50 批和原有尾部记录、50 批验证进度、周期保存及文件/终端一致。变速时钟确认速度窗口没有在中间 TensorBoard 记录点重置；既有受控时钟回归覆盖验证和保存耗时排除。
5. 改动 Python 文件的 `python -m compileall -q` 和 `git diff --check` 通过。仅测试相关模块，未重复全仓库测试或 Rust 构建。

测试产物使用临时目录并自动清理；没有新增一次性工具脚本，保留上述验收日志。

## 正式训练切换

重新核对旧 PID 为 33307，命令、PWD、配置和模型输出目录与当前训练一致。等待下一次第 4,085,000 批 checkpoint 成功提交后，执行：

```bash
kill -TERM 33307
```

确认退出后实际执行：

```bash
cd /home/tml104/Mortal
nohup setsid /home/tml104/miniconda3/envs/mortal/bin/python mortal/precompute_pipeline/train_optimize/train.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml" \
  >> "$PWD/nohup.out" 2>&1 < /dev/null &
printf 'Training PID: %s\n' "$!"
```

新 PID 为 150970；自动日志为 `artifacts/mortal_precomputed_optimize_261014/log-261026-211402.log`。

正式日志从第 4,085,000 批恢复，连续观察到 4,085,250、4,085,500、4,085,750、4,086,000、4,086,250 五个训练日志点，间隔全部为 250，没有中间的 50 批训练日志。已完成周期验证和保存，安全加载的 checkpoint 已推进到 4,086,000 批，学习率仍为 `1e-5`。

直接读取新进程的 TensorBoard 事件文件，确认 4,085,050 到 4,085,500 之间每 50 批均有训练指标；250/500 两个日志点的七项指标与 TensorBoard 在日志显示精度内一致。最终核对新进程仍在运行，保持后台训练。

结构化切换信息见 [restart.json](restart.json)，观察时的运行日志副本见 [runtime-verification.log](runtime-verification.log)。这些 PID 与进度仅是本次检查快照，后续操作须重新核对。

## 修改范围与限制

本次只修改优化训练入口、相邻训练测试和 `docs2/optimized-main-model.md`，新增本验收目录。未修改实际配置、旧训练入口或用户原有 Rust/文档改动。模型目录中的 checkpoint、TensorBoard 和日志由正式训练按原规则持续更新。

日志行为已在隔离测试与正式训练验证；没有为本次日志调整重新验证长期收敛或完整 30 epoch 训练。
