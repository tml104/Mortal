# 默认配置的内存与显存预算

日期：2026-09-27；2026-09-28 补充实现后的有限复测。下文区分精确尺寸、合成数据实测、完整链路规划预算和实现后结果。

## 当前硬件

| 项目 | 查询结果 |
| --- | --- |
| CPU | AMD Ryzen 7 9700X，8 核 / 16 线程 |
| Windows 可见物理内存 | 31.14 GiB |
| WSL 可见内存 | 15.19 GiB |
| WSL swap | 4 GiB，查询时未使用 |
| GPU | NVIDIA GeForce RTX 4090 |
| 显存总量 | 24,564 MiB，约 23.99 GiB |
| 驱动 | 595.97 |
| PyTorch | 2.11.0+cu128 |

Windows 查询时可用物理内存约 6.06 GiB；另一次 WSL 查询显示 MemAvailable 约 12.37 GiB。这两个数属于宿主机与虚拟机的不同视角，不能相加，也不能据此断言 WSL 能额外从宿主机取得全部 12.37 GiB。后台负载与 WSL 内存回收会改变可用量。

第一次 nvidia-smi 查询时显存已用 6,557 MiB、空闲 17,582 MiB。微测稍后从 CUDA 获得的可用量不同；这些是不同时间和接口的快照，不作为固定运行条件。

此前沙箱内 GPU/Windows 互操作不可用。本轮在获工具审批后，通过沙箱外的只读查询取得硬件信息，并运行短小的合成数据资源测量。

## 模型的固定存储

| 模型 | 参数数量 | 参数存储 |
| --- | ---: | ---: |
| Brain v4，192 通道、40 块 | 10,787,456 | 43,149,824 B，float32 |
| DQN v4 | 48,175 | 192,700 B，float32 |
| AuxNet，四类顺位 | 4,096 | 16,384 B，float32 |
| GRP Transformer，128/4/4/60 | 841,112 | 6,728,896 B，float64 |

主策略三个网络共 10,839,727 参数，参数约 41.35 MiB。加上梯度、AdamW 两份 moment 和 BatchNorm buffer，固定张量存储约 **165.52 MiB**；另有很少的 optimizer step 标量和框架对象。

AMP 没有把常驻主模型参数和 AdamW moment 改成 float16，其主要节省来自运算及中间张量。GRP 只做 CPU 推理，参数和 buffer 约 **6.42 MiB**，不进入主策略训练的 GPU。

## 缓冲与样本的确定尺寸

| 数据 | 尺寸 |
| --- | ---: |
| 一个完整 observation，`(1012,34)` float32 | 137,632 B |
| 16,384 条完整 observation | 2.1001 GiB |
| 16,384 条紧凑六元样本，Zstd 解压后 | 168.1875 MiB |
| 当前和预取两个紧凑 chunk | 336.375 MiB |
| batch=1024 的完整 observation | 134.40625 MiB |

以上不含 Python 对象、压缩器、文件列表、模型中间张量或框架开销。实现使用预分配的样本 buffer 直接填写，在满块时编码并重置逻辑长度，避免再用整块 `stack` 制造约 2.1 GiB 的额外 observation 副本。

## GPU 合成数据实测

在独立短进程中构造随机初始化的 Brain/DQN/AuxNet，执行六次合成 batch 的前向、反向和 AdamW 更新：

- batch=1024，AMP float16 开启，GradScaler 使用默认初始值。
- compile 关闭，cuDNN benchmark 开启，CPU 线程数为 4。
- 使用已确认的联合损失和优化器参数。
- 额外分配 168.1875 MiB GPU 紧凑 chunk 占位，计入显存峰值。
- 输入在 GPU 上随机生成，未读取真实牌谱或已有主策略 checkpoint。
- 未实际执行新格式的解码、跨盘读取、CPU 预取或 TensorBoard。
- 模型仅存在于该短进程中，统计写入 `/tmp` 后进程结束。

| 实测项目 | 结果 |
| --- | ---: |
| 包含前几步的实际分配峰值 | 4.1819 GiB |
| 包含前几步的 allocator 预留峰值 | 4.4531 GiB |
| 后三步实际分配峰值 | 4.1819 GiB |
| 后三步 allocator 预留峰值 | 4.4473 GiB |
| 该进程主机内存峰值 RSS | 1.8295 GiB |
| 脚本内部总耗时 | 约 7.16 秒 |

六步损失均为有限值，410 份参数的 AdamW state 已建立，GradScaler scale 保持 65,536。这里不是尚未建立优化器状态的单次前向数字。

**allocated 与 reserved 不能相加。** reserved 已包含 allocator 管理的在用张量与缓存；CUDA 上下文、驱动和其他进程的显存不完全包含在这两个 PyTorch 指标中。

原始统计保存在 `/tmp/mortal-resource-probe-20260927.json`。CPU 参数统计及 saved-tensor 分析保存在 `/tmp/mortal_resource_analysis_20260927_agent.json`；本文保留了不依赖临时文件的关键结果。

## 完整新链路的规划预算

| 阶段 | 建议预留主机内存 | 建议预留本任务显存 | 依据 |
| --- | ---: | ---: | --- |
| 独立筛选 | 约 3–4 GiB | 无需 GPU | 默认 8×2 的 2,000 份真实牌谱只读基准中，worker RSS 合计峰值 2.33 GiB；预算另留主进程与波动余量 |
| 样本生成与编码 | 约 6–8 GiB | 无需 GPU | 2.1 GiB 预分配 buffer、四份牌谱批次、编码临时数组、CPU GRP 与框架；估算 |
| 主策略训练 | 约 3–4 GiB | 约 6–8 GiB | 1.83 GiB 微测 RSS 加读取/预取/保存开销；4.45 GiB 显存池峰值加解码、上下文等余量 |
| 离线验证 | 通常不高于同 batch 训练 | 按训练预算保留 | 无反向图，但进程会保留模型、优化状态及 allocator 缓存；未单独实测 |

这些阶段依次运行，表中预算不应全部相加。全量约 629 GiB 的缓存也不会同时加载到内存或显存。

RTX 4090 的容量对 batch=1024 的本次测量有较大余量。主机内存需要同时考虑 WSL 的 15.19 GiB 可见容量和 Windows 当时的后台占用；生成阶段是本方案最需要控制 RAM 峰值的阶段。

预算不是硬上界。异常长但仍兼容的牌谱、不同卷积算法、读取器实现、内存碎片及其他进程都会影响峰值。正式长跑时仍应观察新 buffer、真实 chunk 解码、跨块组批及保存/恢复，不根据有限测量自动增大 batch。

## 实现后的有限复测

筛选入口对同一批 2,000 份真实 train 牌谱完成只读对照：

| 筛选配置 | 吞吐 | CPU 利用率 | worker RSS 峰值合计 | WSL 最低可用内存 | swap 峰值 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 worker × 2 Rayon | 2.37 份/秒 | 10.7% | 0.41 GiB | 11.60 GiB | 0 GiB |
| 8 workers × 2 Rayon | 9.65 份/秒 | 85.1% | 2.33 GiB | 9.31 GiB | 0 GiB |

两种配置对每份牌谱的判定及原因一致，本批 2,000 份均可用。基准没有移动真实牌谱、写报告或更新真实筛选进度。筛选 worker 已延迟 PyTorch 导入，因此峰值明显低于早期带 PyTorch worker 的短基准。

新 reader、设备端按批解包和训练入口完成后，在 RTX 4090 上用新格式临时数据执行了一个默认 `batch_size=1024`、Brain 192 通道/40 块、AMP 开启、compile 关闭的训练 batch，并完成一批全量验证。

| 项目 | 新链路有限实测 |
| --- | ---: |
| PyTorch 实际分配峰值 | 3.9498 GiB |
| PyTorch allocator 预留峰值 | 4.2949 GiB |

该测试走实际 chunk 文件、确定性 reader 和设备端解包，但数据为临时构造样本，仅运行一次更新，不代表长训练峰值。6–8 GiB 显存规划预算仍保留。

另用 9 个临时样本验证 CUDA AMP 在第 2 批 checkpoint 后恢复，第 3 批及轮末验证完成，GradScaler 与模型/优化状态可加载。

在 WSL 的 `/mnt/c`、`/mnt/d` 各使用自动清理的临时目录读写同一个 19.75 MiB 新格式 chunk，单次结果如下：

| 卷 | 写入 | 读取 |
| --- | ---: | ---: |
| C | 180.6 MiB/s | 231.7 MiB/s |
| D | 202.6 MiB/s | 241.9 MiB/s |

这是小文件、单次、缓存可能参与的有界冒烟结果，只能确认真实挂载路径可写、可读并能解码；不能据此外推数百 GiB 全量吞吐。

## 实际检查入口

```bash
/usr/lib/wsl/lib/nvidia-smi --query-gpu=name,memory.total,memory.used,memory.free,driver_version --format=csv,noheader
/home/tml104/miniconda3/envs/mortal/bin/python -B -u -
```

第二条通过内联脚本完成六步测量；CPU 统计也使用同一 Python 的 `-B` 模式。Windows 使用 PowerShell `Get-CimInstance` 只读查询 `Win32_Processor`、`Win32_ComputerSystem` 和 `Win32_OperatingSystem`。未修改 WSL 配置、已有模型、训练产出或数据集。
