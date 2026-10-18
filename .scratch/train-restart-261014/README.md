# 2026-10-14 优化主模型从头训练

用户要求把当前学习率改为 `1e-5`，从头训练，并保存至 `artifacts/mortal_precomputed_optimize_261014`。本次修改当前实际配置的学习率与输出目录，沿用 batch 2048、30 epoch、原有训练目标以及 Conv2d/channels_last、AMP、编译/CUDA Graph 优化组合。

## 启动与检查

- 启动前工作区干净，未发现其他预计算主模型训练进程；预计算数据状态为 complete，新目标目录尚不存在。
- 北京时间 2026-10-14 10:20:21 通过 nohup 启动独立后台会话，PID 为 6300，PWD 为 `/home/tml104/Mortal`，使用 mortal Conda 环境的 Python。准确命令见 [launch.json](launch.json)。PID 仅代表本次启动，后续操作仍需核对实际进程。
- 配置通过 `load_settings` 和 `validate_output` 检查，输出路径、学习率、batch 与 epoch 均符合要求；`git diff --check` 通过。
- 初始 checkpoint 为 0 batch、0 样本，优化器历史为空，无迁移标记；两组 AdamW 学习率及训练契约均为 `1e-5`。记录见 [initial-check.json](initial-check.json)。
- 已完成 500 个 batch / 1,024,000 个样本，所有优化器参数状态的 step 均为 500。131,072 条固定验证样本处理完成，并在 10:22:06 成功提交周期 checkpoint；使用 `torch.load(weights_only=True, map_location='cpu')` 核对全部模型张量与验证指标均为有限值、学习率正确、TensorBoard 文件已生成。记录见 [verification.json](verification.json)。
- 保存后训练继续超过 650 批，稳定训练窗口约 19,000–20,000 样本/秒；最终再次检查进程仍在运行，保持后台训练。

自动日志位于模型目录，终端副本保留在 [startup.log](startup.log)。旧模型目录和预计算数据未修改。此次采用随机初始化，没有加载旧 checkpoint。

## 查看与恢复

在 `~/Mortal` 查看本次日志：

```bash
tail -f artifacts/mortal_precomputed_optimize_261014/log-261014-102028.log
```

若后续进程退出，在确认没有同配置训练进程后恢复：

```bash
conda activate mortal
nohup python mortal/precompute_pipeline/train_optimize/train.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml" &
```

已有 `latest.pth` 时会恢复本次新训练的已提交进度，不再随机初始化。主 checkpoint、TensorBoard 和按次运行日志都保存在新模型目录。

## 验证范围

本次完成配置加载检查及真实训练、验证、保存和存活检查，未修改训练源码，因此未重复执行完整单元测试套件。新训练尚未完成全部 30 epoch，当前检查不代表长期收敛或最终棋力保证。
