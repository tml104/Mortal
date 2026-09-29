# 并行牌谱筛选

Status: implemented

## Problem Statement

训练数据维护者需要在约 145 万份牌谱中提前隔离不可用牌谱。当前筛选入口一次只处理一份文件，尽管 libriichi 会在单份牌谱内并行处理四家视角，实测仍只有约 3.8 份/秒，完整扫描可能持续四至五天。机器拥有 8 个物理核心、16 个逻辑线程和足够内存，但现有外层串行循环无法同时处理多份牌谱。

直接增加 Python 线程不能稳定解决问题：当前 Python 是常规 GIL 构建，PyO3 的 `load_log` 调用未显式释放 GIL。直接增加 Rayon 线程或无限增加进程也不能线性加速，因为 gzip、Python 规则检查、Rust 视角处理、NumPy 扫描、内存带宽、CPU 缓存和任务同步会共同形成瓶颈。

优化必须保留当前行为：坏谱只能由明确输入错误触发隔离，未知异常必须停止；文件移动、报告和恢复游标必须保持排序提交；现有 incomplete 进度必须可继续使用；性能统计不能污染只负责恢复的进度文件。

## Solution

筛选入口使用有界 `spawn` 进程池并行校验多份牌谱。每个 worker 只读取并完整校验一份文件，复用自己的 libriichi 校验器，只返回小型判定和阶段耗时。主进程仍是唯一提交者：它可以乱序接收结果，但只按文件名顺序移动坏谱、追加报告并推进恢复游标。

默认使用 8 个 worker、每个 worker 2 个 Rayon 线程、最多 16 个在途任务。三项都可配置。筛选 worker 不加载 PyTorch；公共 codec 仅在设备张量解码函数实际调用时才导入 PyTorch。

每完成默认 100 份顺序提交后，终端打印当前 split、已处理数量、总数、隔离数、平均速度、ETA、当前游标，以及 gzip 读取、Python 规则检查、libriichi 样本适用性检查三个阶段的平均耗时。`progress.json` 继续只保存恢复所需的 schema、状态、split 和相对路径游标。

## User Stories

1. As a 训练数据维护者, I want 多份牌谱同时校验, so that 全量筛选不再需要四至五天
2. As a 训练数据维护者, I want 默认并发参数针对当前 9700X 调优, so that 首次使用即可获得实测较高吞吐
3. As a 训练数据维护者, I want worker 数可以配置, so that 我可以适配不同核心数和内存容量的机器
4. As a 训练数据维护者, I want 每个 worker 的 Rayon 线程数可以配置, so that 进程级和谱内并行不会过度竞争 CPU
5. As a 训练数据维护者, I want 在途任务数量有明确上限, so that 百万文件不会产生百万个 Future 或无限积压结果
6. As a 训练数据维护者, I want worker 使用 spawn 启动, so that 不会 fork 已初始化的 Rayon 或 PyTorch 运行时
7. As a 训练数据维护者, I want worker 只读牌谱, so that 并行任务不能乱序移动源文件
8. As a 训练数据维护者, I want 主进程统一移动不可用牌谱, so that 隔离副作用仍按文件名顺序发生
9. As a 训练数据维护者, I want 主进程统一追加隔离报告, so that 多进程不会竞争写同一个 JSONL
10. As a 训练数据维护者, I want 只有连续完成的文件才能推进游标, so that 中断恢复不会跳过尚未提交的牌谱
11. As a 训练数据维护者, I want worker 只返回小型结果, so that observation 和 Gameplay 不会在进程间复制
12. As a 训练数据维护者, I want 明确的数据错误仍然隔离整份牌谱, so that 并行化不改变数据质量标准
13. As a 训练数据维护者, I want 未知异常和 Rust panic 仍然停止运行, so that 程序错误不会被误判为坏牌谱
14. As a 训练数据维护者, I want 未开始的任务在异常后取消, so that 失败后不会继续浪费资源
15. As a 训练数据维护者, I want 已提前读取但未提交的文件保持原样, so that 重启时可以按原游标自然重查
16. As a 训练数据维护者, I want 旧版 incomplete 进度可直接恢复, so that 已完成的串行筛选工作不会丢失
17. As a 训练数据维护者, I want `progress.json` 只保存恢复状态, so that 性能展示字段不会变成新的恢复契约
18. As a 训练数据维护者, I want 终端显示已处理数量和总数, so that 我能直观看到整体进度
19. As a 训练数据维护者, I want 终端显示平均速度和 ETA, so that 我能安排后续预计算时间
20. As a 训练数据维护者, I want 终端显示隔离数量和当前游标, so that 我能快速发现异常数据集中出现的位置
21. As a 性能调优者, I want 看到 gzip、Python 规则和 libriichi 三阶段耗时, so that 后续优化有测量依据
22. As a 机器资源维护者, I want worker 不加载未使用的 PyTorch, so that 多进程筛选占用更少内存并缩短启动时间
23. As a 训练数据维护者, I want worker_processes=1 仍产生相同筛选结果, so that 我可以在低资源机器或诊断场景下降级运行
24. As a 训练数据维护者, I want train 完整提交后才切换 val, so that split 状态和现有恢复语义保持一致
25. As a 训练数据维护者, I want 完成状态仍由主进程在所有 val 结果提交后写入, so that 提前派发任务不会造成假完成
26. As a 开发者, I want 并行实现继续复用现有协议、牌库存、点数和样本适用性校验, so that 不产生第二套数据规则
27. As a 开发者, I want 乱序调度通过公开筛选入口验证, so that 测试关注可观察行为而非 Future 内部细节
28. As a 训练数据维护者, I want 优化代码本身不自动启动全量筛选, so that 部署和长跑仍由我明确控制

## Implementation Decisions

- 保持现有筛选入口及 `run_quarantine` 为主要用户接口；配置、隔离目录、JSONL 报告和恢复游标含义不变。
- 使用 `ProcessPoolExecutor` 和 `spawn` multiprocessing context。进程池 worker 必须是可 pickle 的模块级函数。
- worker initializer 在首次 libriichi 调用前设置每进程 Rayon 线程数，并创建一个长期复用的样本适用性校验器。
- 新增三个正整数配置：worker 进程数默认 8、每 worker Rayon 线程数默认 2、最大在途任务数默认 16。最大在途数不得小于 worker 数。
- 主进程生成排序后的剩余文件流，并限制“最后顺序提交位置之后的全部已派发任务”不超过窗口上限；已完成但等待前序结果的任务同样占用窗口。
- worker 完成 gzip 读取、协议/牌/点数规则检查、libriichi 提取和样本适用性检查。它返回成功或不可用原因，以及三个阶段的 wall-clock 耗时。
- worker 只将既有 `InvalidLogData` 转换为可隔离结果。其他异常必须通过 Future 传播给主进程。
- 主进程可以使用完成通知收集乱序结果，并放入按输入序号索引的短暂结果缓冲。只有从下一个待提交序号开始连续就绪的结果才能产生文件移动、报告追加和内存游标推进。
- 收到任何未知异常、worker 异常退出或用户中断后，主进程停止派发，取消尚未开始的任务，等待或终止进程池清理后重新抛出；未提交结果不产生副作用。
- 继续按 `commit_every` 提交现有进度结构。性能计数、速度、ETA、隔离数和阶段耗时只打印到终端，不加入持久化 JSON。
- 启动时根据源文件、隔离目录和恢复游标计算用于展示的 split 总数及已处理位置。该计算不参与恢复正确性。
- 阶段耗时以累计平均毫秒/文件显示。并行 worker 的阶段时间可以重叠，因此三段时间不能相加后当作主进程 wall time。
- 公共 codec 不再在模块导入时加载 PyTorch；设备端解码函数进行局部导入。CPU 编解码、chunk 格式和训练接口保持不变。
- 现有 incomplete 进度 schema 保持兼容。新版本启动后严格处理大于游标的现存源文件；read-ahead 不修改这一规则。
- train 的所有顺序结果提交完毕后才切换 val；val 的所有结果提交后才写 complete。
- 当前硬件实测默认候选为 8×2。96 份牌谱短基准达到约 10.0 份/秒、平均使用约 13.3 个逻辑核；这不是长期速度保证。

## Testing Decisions

- 主要测试 seam 是公开筛选入口和 `run_quarantine`。测试只观察源文件、quarantine、报告、终端输出和进度游标，不断言内部 Future 数组或调度实现。
- 允许通过依赖注入提供可控制延迟和结果的 validator/executor，以稳定制造乱序完成、前序慢任务和未知异常；生产默认仍使用真实 spawn 进程池。
- 用不同延迟让后方文件先完成，验证坏谱移动、报告顺序和进度游标仍与排序输入一致。
- 在前方任务未完成时让后方结果大量完成，验证在途加等待结果的总数不超过配置窗口。
- 在 worker 返回 `InvalidLogData` 时验证只有主进程移动文件并追加一条报告。
- 在 worker 抛出未知异常时验证后续文件未移动、进度未越过故障文件、read-ahead 文件在恢复后重查。
- 复用现有筛选测试覆盖 gzip/JSON、物理牌、供托、空视角、容量限制、游标恢复和目标冲突，证明并行化没有改变规则。
- 比较 worker 数为 1 与默认并发配置的最终源文件集合、隔离集合和报告原因，忽略允许重复的未提交报告语义。
- 验证旧版 incomplete 进度文件可直接继续，且新提交的 JSON 仍只包含原恢复字段。
- 捕获终端输出，验证每个提交周期包含 split、计数、百分比、速率、ETA、隔离数、阶段平均耗时和游标。
- 验证配置拒绝零或负并发值，以及小于 worker 数的在途窗口。
- 验证导入筛选 worker 所需模块不会加载 PyTorch；设备端 codec 测试继续验证局部导入后的 CPU/CUDA 解码行为。
- 完成功能测试后，对至少数千份真实牌谱运行有界基准，记录吞吐、CPU、worker RSS、WSL 可用内存和 swap。该基准不移动真实牌谱，也不自动继续全量筛选。
- 参考现有 unittest 风格和临时目录隔离方式；测试产物必须自动清理。

## Out of Scope

- 不改变协议、物理牌、点数、供托、动作 mask、空视角或奖励模型容量等筛选规则。
- 不把筛选逻辑迁移到 Rust，也不新增 PyO3 批量校验接口。
- 不使用 Python ThreadPoolExecutor 作为主要并行实现。
- 不采用 Rust 批量加载后对失败批次进行二分或逐文件重试。
- 不改变 quarantine 目录结构、JSONL 报告语义或目标冲突行为。
- 不把速度、ETA、计数或阶段耗时持久化到 `progress.json`。
- 不保证固定倍数加速或固定完成时间。
- 不在实现或测试阶段自动启动百万文件全量筛选。
- 不修改预计算生成和训练阶段的线程模型。

## Further Notes

- 原串行进程已结束，最后提交游标位于 train 的 2019 年数据；新实现应从现有 incomplete 状态继续。
- 短基准使用 fork 以便快速比较组合，正式实现使用 spawn。对百万文件而言启动成本占比较小，但实现后仍需复测。
- 8×3 和 10×2 虽提高 CPU 利用率，却低于 8×2 的吞吐，说明当前硬件已开始受调度、同步、缓存或内存带宽限制。
- 该选择是可配置的运行策略，不需要新增 ADR；未来硬件可以通过配置和有界基准重新选择默认值。
