# 预计算进度与编码提速验收

日期：2026-10-01。Q39–Q44 已以正式模块代码实现。本次没有启动正式全量预计算或修改原牌谱、既有 chunk 与恢复进度。

## 已交付行为

- 默认每个 chunk 和对应进度提交成功后即时输出日志；日志间隔可配置，启动/恢复、扫描、split 切换、结束和已完成退出有明确反馈。
- 块/样本和完整文件进度按已提交成果计算，支持谱内、谱末及 EOF 旧游标；速度只使用本次增量，ETA 只针对当前 split。
- 五段累计耗时分别为 Rust、GRP、Python 样本整理（含 buffer 复制）、编码压缩、写盘提交；Rust/GRP 不被重复计入 Python。
- 默认参数为 16/1/4，新增 log_every_chunks=1；采用直接标量提取、连续逐牌布局和紧凑二值数组，编码省去重复观测逐值检查。筛选保留完整语义校验。
- float32/float64 精度、样本顺序、chunk 格式和恢复 JSON 保持兼容；本次运行统计不写入恢复状态。

## 验证命令与结果

执行环境为现有 mortal Conda Python；PWD 为 ~/Mortal。

```bash
env MORTAL_RUN_INTEGRATION=1 /home/tml104/miniconda3/envs/mortal/bin/python -B -m unittest discover -s mortal/precompute_pipeline -t mortal
/home/tml104/miniconda3/envs/mortal/bin/python -B -m unittest discover -s mortal/transformer_pipeline/tests
/home/tml104/miniconda3/envs/mortal/bin/python -B mortal/precompute_pipeline/precompute/precompute.py --help
```

- 新链路完整套件：63 项通过，包含隔离真实牌谱的筛选、预计算和 CPU 小模型训练/验证。完整输出见 [test-results.txt](test-results.txt)。
- 旧 Transformer 管线兼容性：5 项通过。
- CLI --help 正常退出，确认直接文件路径启动可用。
- 在 RTX 4090 上对三条构造样本执行公开 CUDA 解码，六元张量与原样本精确相等；没有运行正式 GPU 训练。
- 改动 Python 文件经 AST 检查，定向测试在实现过程中反复执行。仓库没有静态类型检查配置，当前环境也没有 mypy/pyright，本轮未安装新的检查工具。

## 正式实现的性能对照

每轮使用固定 128 份真实牌谱，生成 85,613 条样本、5 个满 chunk 和一个尾 chunk。四份牌谱预热后计时；模型初始化和预热不计入吞吐。基线与正式版在独立新进程中顺序执行，顺序为基线→正式→正式→基线。

基线源代码来自实现前快照，对应 HEAD 8b94ec9718f0e32adea2c365c0088bf3f5bf4682；基线使用 4/4/4，正式版使用 16/1/4。chunk 写入真实 D 盘的独立临时目录，进度 JSON 位于 WSL 临时目录；执行真实 fsync 和原子提交，退出后自动清理。

本次运行正式入口，不用临时 monkey patch 替换生产函数。五段时间直接从正式日志读取。实际执行了归档脚本的 --output 参数，并在基线轮次传入 --baseline-root 指定实现前快照；四轮为顺序子进程。

| 版本 | 秒 | 样本/秒 | 峰值 RSS GiB |
| --- | ---: | ---: | ---: |
| baseline | 55.53 | 1542 | 7.17 |
| implemented | 30.14 | 2841 | 5.14 |
| implemented | 29.43 | 2909 | 5.17 |
| baseline | 60.86 | 1407 | 7.18 |

按总样本数除以总耗时汇总：基线约 **1471 样本/秒**，正式版约 **2875 样本/秒**，本次提升约 **1.95 倍**。正式版峰值 RSS 为 5.14–5.17 GiB，原版约 7.18 GiB。

| 正式版轮次 | Rust 秒 | GRP 秒 | Python 秒 | 编码秒 | 提交秒 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 17.987 | 0.745 | 4.842 | 5.939 | 0.543 |
| 2 | 17.783 | 0.756 | 4.995 | 5.393 | 0.432 |

四次运行的所有解压后 chunk 载荷逐字节一致，校验覆盖格式头、全部字段、样本顺序及 float64 奖励，不使用浮点容差。校验在计时结束后执行；临时摘要仅用于验收，没有成为生产链路的运行前置检查。

原始测量见 [acceptance-results.jsonl](acceptance-results.jsonl)，执行脚本已归档为 [acceptance-benchmark.py](acceptance-benchmark.py)。脚本的 --baseline-root 指向包含旧版 mortal 子目录的实现前快照；不传该参数时验证当前正式代码。

## 数据与资源边界

- 已逐字段核对正式进度与实验前快照一致：train 739 个已提交 chunk、12,107,776 条样本，原生成 ID 和文件内 offset 保留；本轮没有继续正式预计算。
- 正式牌谱只读；临时 chunk 和 checkpoint 已自动清理。任务目录有意保留验收脚本、测量与测试记录，便于复查。
- 性能输入集中于恢复点附近两天，缓存、其他年份和系统负载可能改变结果；约 1.95 倍不是全量承诺，RSS 也不是全量或 Windows 整机内存上界。
- 本次实际盘测量覆盖 D 盘；C/D 路由、配额和切盘恢复由临时双目录回归覆盖，没有重新测量 C 盘全量写入速度。
- 没有修改 Rust/PyO3 接口，未运行 Rust 全矩阵；未执行百万份全量预计算、数百 GiB chunk 生成或正式长训练。

## Standards

独立只读审查：0 findings。未发现实质规范违反或需要阻塞交付的代码异味；配置变更限于授权键，计时与日志职责集中。

## Spec

独立只读审查：0 findings。已核对谱内/谱末/EOF 恢复、成功提交边界、五段计时、旧数据格式、筛选检查保留和配置约束。

审查以本轮开始前工作区快照生成精确差异，覆盖新增文件，排除用户已有的无关文档修改。两条审查均未修改文件或额外运行测试。

