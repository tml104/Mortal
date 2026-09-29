# 新预计算链路使用说明

本文说明新的牌谱筛选、无损样本预计算和主模型训练入口。旧的 Transformer-GRP 实验脚本仍保留，但新链路不导入其缓存、训练或恢复实现。

## 前置条件

- 从仓库根目录运行命令，并使用 mortal Conda 环境。
- 已按项目说明构建可导入的 libriichi 扩展。
- 准备兼容的 Transformer-GRP 权重；默认配置使用 `artifacts/grp_transformer.pth`。
- C、D 盘为缓存准备专用目录和足够空间。默认预算为 C 320 GiB、D 440 GiB；脚本不会突破配置预算。
- 将示例配置复制到产出目录之外。仓库内路径以 `~/Mortal` 为 PWD 使用相对路径，C/D 缓存目录使用绝对路径；根目录现有 `config.toml` 不属于此链路。

## 三个入口

首次运行前，在 `~/Mortal` 下创建一份实际配置。该文件位于训练产出目录之外，不会被从头训练时的目录清理删除：

```bash
cd ~/Mortal
cp mortal/precompute_pipeline/config.example.toml mortal/precompute_pipeline/config.toml
```

根据当前机器修改 `mortal/precompute_pipeline/config.toml` 中的路径、磁盘预算和运行参数。所有相对路径均以启动脚本时的当前工作目录解析；下面固定以 `~/Mortal` 为 PWD。上面的复制命令只需执行一次；已有实际配置时不要再次覆盖。

随后在 `~/Mortal` 下按顺序执行三个入口：

```bash
cd ~/Mortal
python mortal/precompute_pipeline/quarantine_logs/quarantine_logs.py --config "$PWD/mortal/precompute_pipeline/config.toml"
python mortal/precompute_pipeline/precompute/precompute.py --config "$PWD/mortal/precompute_pipeline/config.toml"
python mortal/precompute_pipeline/train/train.py --config "$PWD/mortal/precompute_pipeline/config.toml"
```

这些命令假定已经激活 mortal Conda 环境。三个脚本依次完成牌谱筛选、全量预计算和主模型训练；前一步没有成功完成时，不应启动后一步。

筛选入口检查协议、物理牌与点数规则，并通过 libriichi 实际提取 observation 和动作；它不运行奖励模型。确定不可用的牌谱会保留原相对路径移入 quarantine，原因追加到 JSONL 报告。未知程序、内存或权限错误会停止脚本，不会被当作坏牌谱移动。

## 筛选并行与进度

筛选入口使用 `spawn` 多进程并行校验多份牌谱。worker 只读牌谱并复用自己的 libriichi 校验器；主进程即使乱序收到结果，也只按文件名顺序移动坏谱、追加报告和推进游标。已完成但等待前序牌谱的结果也占用在途窗口，因此内存和 read-ahead 数量有明确上限。

`[quarantine]` 下的 `worker_processes`、`rayon_threads_per_worker` 和 `max_in_flight` 分别控制 worker 数、每个 worker 的 Rust Rayon 线程数和在途上限，默认是 8、2、16。三项必须是正整数，且在途上限不能小于 worker 数。低内存机器可先降低 worker 数；单 worker 仍使用 spawn 路径，便于在相同语义下降级运行。

每顺序提交 `commit_every` 份牌谱时，终端输出 split、已处理/总数、百分比、当前运行累计隔离数、平均速度、ETA、游标，以及 gzip、Python 规则和 libriichi 样本适用性三个阶段的平均毫秒/文件。worker 的阶段可以并行重叠，三个平均耗时不能相加后解释为主进程 wall time。速度和耗时只在终端显示，`progress.json` 仍只有 schema、状态、split 和相对路径游标。

预计算入口按文件名顺序处理筛选后的 train 和 val，使用冻结 GRP 生成六元样本，默认每 16,384 个样本形成一块。观测以 float32 无损保存，连续与二值通道按固定语义编码，再用 Zstd level 1 压缩。train 优先写 D 再写 C，val 写 C；项目目录通过链接提供统一入口。

训练入口从完整数据集随机初始化 Brain、DQN 和 AuxNet，默认在 `cuda:0` 使用 batch 1024 和 AMP。训练先打乱 chunk，再独立打乱每块内部样本；CPU 最多预取一块，完整 observation 只按 batch 在目标设备展开。周期验证使用固定 val 子集，每轮结束验证全部 val。

## 自动恢复

三个入口都以各自的持久化状态恢复，不需要额外 `--resume`：

- 筛选按 split 和最后已处理的相对路径继续；已经移入 quarantine 的游标文件不需要仍在源目录。
- 预计算只承认 `progress.json` 已提交的 chunk。未提交 buffer 会重算，已发布但未登记的下一块会在恢复时清理并重建。
- 训练从唯一的 `latest.pth` 恢复模型、AdamW、AMP、随机状态和已消费样本位置，不从头解码历史 chunk。

状态文件存在但损坏时，脚本直接报错；不会自动当作新运行并清理数据。预计算完成后再次启动会退出。训练完成后，目标 epoch 数不增加会退出；提高 `num_epochs` 后从下一轮继续。

同一次预计算恢复期间，牌谱集合和内容、奖励权重及样本生成语义必须保持不变。训练恢复允许调整线程、日志/保存频率、兼容设备、compile 和 benchmark 开关；网络、batch size、AMP、损失和优化器语义不能改变。

## 重新开始与重扫

筛选完成后需要重扫源目录时使用：

```bash
cd ~/Mortal
python mortal/precompute_pipeline/quarantine_logs/quarantine_logs.py --config "$PWD/mortal/precompute_pipeline/config.toml" --rescan
```

重扫不会清空 quarantine。预计算没有进度文件时视为从头开始，只清理专用输出目录中符合本链路 chunk/临时命名的文件，不删除未知文件。训练没有 checkpoint 时会清空配置指定的训练产出子目录，因此配置和 GRP 权重不能放在该目录内。

## 可调线程与资源

筛选入口使用 `worker_processes`、`rayon_threads_per_worker` 和 `max_in_flight`。预计算入口另有 `rayon_threads` 和 `torch_threads`，分别控制 Rust 数据提取和 PyTorch CPU 奖励计算，默认都是 4；两组参数互不替代。

当前机器对同一批 2,000 份真实牌谱进行只读校验时，单 worker × 2 Rayon 为 2.37 份/秒，8 workers × 2 Rayon 为 9.65 份/秒，提升约 4.07 倍；两者逐份判定完全一致。8×2 的 CPU 利用率约 85.1%，worker RSS 合计峰值约 2.33 GiB，WSL 最低可用内存约 9.31 GiB，未使用 swap。测试中的 2,000 份均判定可用，且测试没有移动牌谱、写报告或更新真实进度。

筛选 worker 不加载 PyTorch，因此当前机器可先按 3–4 GiB 主机内存预算运行默认 8×2；预计算仍按约 6–8 GiB 主机内存预算，训练按约 3–4 GiB 主机内存及 6–8 GiB 显存预算。以上是当前样本上的实测或规划值，不是硬上界；其他硬件可降低 worker 数控制内存，再根据终端吞吐调整。

完整全量筛选、约 629 GiB 缓存生成和正式训练均是长时间、高资源流程，不属于普通冒烟测试。先运行仓库提供的小型测试和有限真实数据验证，再另行启动全量任务。
