# 01: 通过 spawn 单 worker 完成轻量筛选

**What to build:** 用户将 worker 数配置为 1 时，可以通过现有筛选入口在一个独立的 `spawn` worker 中完成整套牌谱校验；该路径保留原有隔离、报告和恢复行为，同时避免为筛选进程加载未使用的 PyTorch。

**Blocked by:** None (can start immediately).

**Status:** ready-for-agent

- [x] 筛选配置支持正整数 worker 数、每 worker Rayon 线程数和最大在途任务数，默认分别为 8、2、16，并拒绝最大在途数小于 worker 数的配置。
- [x] `worker_processes=1` 时仍使用 `spawn` 子进程完成 gzip、Python 规则和 libriichi 样本适用性校验，而不是回退到原串行主进程路径。
- [x] worker 在首次使用 libriichi 前应用本进程的 Rayon 线程配置，并在多份任务之间复用已初始化的样本适用性校验器。
- [x] worker 只返回校验判定、不可用原因和三个阶段的耗时，不移动牌谱、不追加报告、不写进度。
- [x] 通过公开筛选入口验证单 worker 的源文件、隔离文件、报告和最终进度与原串行语义一致。
- [x] 导入筛选 worker 所需模块不会把 PyTorch 加载进进程；设备端 chunk 解码仍可在实际调用时正常导入并使用 PyTorch。

## Comments

2026-09-28：完成。配置默认值与非法值测试通过；实际 `spawn` 单 worker 处理损坏 gzip 后父进程未加载 libriichi；codec 独立导入未加载 PyTorch，完整 codec 测试继续通过。
