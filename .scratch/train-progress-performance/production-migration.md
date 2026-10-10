# 正式模型首次配置与一次性迁移

日期：2026-10-08。用户明确要求完成首次配置与一次性迁移。

- 已创建实际优化训练配置；以 `~/Mortal` 为 PWD，保持默认 batch 2048、总目标 30 epoch、FP16 AMP、学习率 1e-4、编译线程 2、指标每 50 批、验证保存每 500 批。
- 检查时没有旧训练进程，预计算数据集处于 complete 状态，目标产出目录尚不存在。使用旧训练最新完整 checkpoint，直接迁移到新的优化目录。
- 源：`artifacts/mortal_precomputed_v2/latest.pth`。目标：`artifacts/mortal_precomputed_optimize/latest.pth`。原目录保留。
- 已完成 3 个 epoch，当前为第 4 个 epoch；轮内已消费 388696064 样本，累计 2905085576 样本、2837000 批全部保留。下次日志为 2837050，下次周期验证/保存为 2837500。
- 对照源与目标的全部 653 个模型/BN 张量和 1230 个优化器张量：除卷积增加高度维度外，数值精确相同。参数组、优化器 step、AMP、主种子、CPU/CUDA RNG 和最近验证信息保留，仅未输出展示窗口清空。
- 新配置与数据契约核对通过；严格加载全部新模型及 AdamW，并在 CUDA 上恢复模型、优化器、AMP 和随机状态通过。本次没有前向、反向或优化更新，也没有启动长期训练。

机器可读的详细结果见[production-migration.json](production-migration.json)。该模型后续启动和恢复直接使用新 checkpoint，不再重复运行迁移。

```bash
cd ~/Mortal
conda activate mortal
python mortal/precompute_pipeline/train_optimize/train.py \
  --config "$PWD/mortal/precompute_pipeline/train_optimize/config.toml"
```

迁移入口是否清理，按此前讨论在短期续训、成功保存及保存后重启验证完成后处理。本轮完成首次配置与正式迁移，仍保留该工具。

## 后续正式续训

2026-10-08：按用户继续训练的请求，已启动独立后台进程，PID 132144，运行配置为上述实际配置。确认从第 2837000 批恢复，首次真实 batch 的编译/捕获约 67 秒，随后稳定训练约 19000 样本/秒。至少完成 1000 个新增 batch，两次固定 131072 样本验证与原子 checkpoint 提交成功，TensorBoard 已产生事件文件；进程继续在后台运行。

运行记录见[runtime/launch.json](runtime/launch.json)，终端日志见[runtime/train.log](runtime/train.log)。后续进度会超过本段确认时的快照；当前 checkpoint 位于新优化产出目录。尚未人为停止并重新启动此次正式训练。
