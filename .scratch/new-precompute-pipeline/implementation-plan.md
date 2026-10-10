# 新预计算链路完整实施方案

状态：Q1–Q38 已实现并完成有限验收；Q39–Q44 的[预计算进度日志与编码提速](../precompute-progress-performance/spec.md)已实现，验收记录见[结果](../precompute-progress-performance/acceptance.md)。此前的[并行筛选优化](../quarantine-parallelism/spec.md)已完成；本轮不启动全量数据处理或正式训练。

讨论记录见 [spec.md](spec.md)，磁盘估算见 [space-estimate.md](space-estimate.md)，当前硬件的内存/显存测量见 [resource-estimate.md](resource-estimate.md)。已确认取舍记录于 [ADR 0001](../../docs2/adr/0001-frozen-precomputed-samples.md)。

## 1. 交付范围与三个独立入口

在 `mortal/precompute_pipeline/` 新写整条链路，继续调用 libriichi 和 `mortal/model.py` 的网络定义。旧训练器、缓存器、奖励计算器和恢复工具函数不作为新链路依赖。

| 入口 | 输入 | 输出 |
| --- | --- | --- |
| `quarantine_logs/quarantine_logs.py` | `data/grp/train`、`data/grp/val` | 保留可用牌谱；隔离问题牌谱；筛选报告与断点 |
| `precompute/precompute.py` | 筛选后的牌谱、冻结 GRP 权重 | 全量 train/val chunks、一个预计算进度 JSON |
| `train/train.py` | 完整 chunks 数据集 | Brain/DQN/AuxNet、一个恢复 checkpoint、TensorBoard |

以下入口和模块路径均相对 `mortal/precompute_pipeline/`。目录结构如下，省略空的包标记文件：

```text
mortal/precompute_pipeline/
  config.example.toml
  common/
    settings.py
    codec.py
    dataset.py
    files.py
    tests/
  quarantine_logs/
    quarantine_logs.py
    log_validation.py
    tests/
  precompute/
    precompute.py
    progress.py
    samples.py
    storage.py
    tests/
  train/
    train.py
    reader.py
    tests/
  tests/                       # 跨阶段集成测试
```

已确认的模块边界：

| 新文件 | 职责 |
| --- | --- |
| `common/settings.py` | 独立 TOML 读取、默认值和必要配置检查，三个入口共用 |
| `common/codec.py` | 固定语义布局、格式与位宽约束、chunk 编码、CPU 读取和设备 batch 解码 |
| `common/dataset.py` | 进度与完成记录的共享结构、生成 ID、chunk 分配区间及路径解析 |
| `common/files.py` | 共用的原子文件提交和有序牌谱路径扫描 |
| `quarantine_logs/log_validation.py` | 基础协议、库存、点数及样本可用性判据；区分数据错误与运行异常 |
| `quarantine_logs/quarantine_logs.py` | 扫描、隔离、报告、筛选断点和 CLI |
| `precompute/samples.py` | Rust 批量提取、CPU GRP 奖励、六元样本构造 |
| `precompute/storage.py` | 两盘配额、目标盘选择、chunk 写入、预计算进度提交及未提交文件清理 |
| `precompute/precompute.py` | 文件排序、固定样本数 buffer、跨文件分块、恢复和 CLI |
| `precompute/progress.py` | 本次运行的提交进度、增量速度、ETA、五段计时与即时终端输出 |
| `train/reader.py` | 两级确定性打乱、直接定位、CPU 单块预取、跨 chunk batch |
| `train/train.py` | 训练 CLI、模型与优化器初始化、训练与验证循环、checkpoint 恢复与保存、指标和 TensorBoard |
| `config.example.toml` | 已确认参数的独立配置示例 |
| 各子目录的 `tests/` | 就近验证对应模块的数据语义、格式、提交或恢复行为 |
| 根部 `tests/` | 跨筛选、预计算和训练的集成验证 |

各入口目录仅依赖自身模块和 `common/`，入口目录之间不直接导入对方私有模块；common 不反向导入入口目录。跨阶段测试可以引用多个入口。

继续按文件路径启动。各入口保留少量按 `__file__` 定位并暴露 `mortal/` 搜索路径的代码；新模块使用 `precompute_pipeline.common.codec` 等完整限定名导入，避免与已有 `mortal/common.py`、`mortal/train.py` 冲突。现有模型保持平铺导入方式。

训练入口与核心训练流程统一放在 `train/train.py`，以普通函数组织。`main()` 负责解析命令行参数、调用 `common/settings.py` 读取配置并启动同文件中的训练流程；通过 `if __name__ == "__main__":` 保护启动入口，导入模块时不启动训练或清空输出目录。测试可以直接调用其中的训练、验证和恢复函数。

不修改根目录实验 `config.toml`，不删除旧链路。新增函数按项目约定写简洁中文注释。模型与 checkpoint 读取继续使用 `weights_only=True`。

## 2. 已确认配置

| 类别 | 默认值 |
| --- | --- |
| 观测与样本 | v4 `(1012,34)`；四家；无 Oracle、无增强；六元样本 |
| GRP | `artifacts/grp_transformer.pth`；128 维、4 头、4 层、60 局容量；CPU float64/eval |
| 名次效用与折扣 | `[6,4,2,0]`；gamma=1，训练时应用折扣 |
| 生成 | 每批四份牌谱；Rust/PyTorch CPU 默认 16/1，均可配置 |
| chunk | 默认 16,384 样本，可配置；各 split 最后余量单独成块 |
| 编码 | 四类固定语义；观测 float32 无损；奖励 float64；Zstd level 1 |
| 主模型 | Brain 192 通道、40 块；DQN v4；AuxNet 四类顺位；BN 正常更新 |
| 损失 | `0.5*MSE + 5*CQL + 0.2*顺位交叉熵` |
| 优化 | AdamW，lr=1e-4 恒定，betas=(0.9,0.999)，eps=1e-8；仅卷积/线性 weight 使用 0.1 decay |
| 更新方式 | 每 batch 一次优化尝试；无梯度累积、无梯度裁剪；尾批正常更新 |
| 训练设备 | cuda:0；batch=1024；AMP 开启；compile 默认关闭；cuDNN benchmark 开启 |
| 轮数与随机 | 默认 1 epoch，可增加；从头时生成随机主 seed，各轮重新确定性打乱 |
| 读取 | 当前 chunk 保持紧凑表示，按 batch 在设备展开；CPU 最多预取一个 chunk |
| 日志与保存 | 每 100 batch 写汇总；每 1,000 batch 周期验证并保存，可配置 |
| 预计算进度 | log_every_chunks=1；提交后即时输出进度、增量速度、split ETA、磁盘用量和五段累计耗时 |
| 验证 | 固定 131,072 样本，不足时全取；validation_seed=0；每轮结束全量 val |
| 隔离 | 保留 split/相对目录；每 100 文件提交筛选进度；JSONL 记录原因 |
| 筛选并行 | spawn 进程池；8 workers × 2 Rayon；最大在途 16；终端输出速度、ETA 与阶段耗时 |
| 训练输出 | `artifacts/mortal_precomputed_v2/latest.pth` 及同目录 TensorBoard |

与样本或训练语义有关的参数在恢复时核对；线程、日志/保存频率等运行参数按 Q27 的规则允许调整。恒定学习率由优化器状态保存，不引入旧调度器代码。

## 3. 独立筛选

递归发现 train、val 中的 `.json.gz`，分别按相对路径的字典顺序处理。规则检查与 libriichi 样本提取都在该阶段执行，GRP 神经网络不参与筛选。

当前实现由只读 worker 并行完整校验文件，主进程按排序后的连续结果执行隔离、报告和游标提交；规则检查内容不变。

并行版继续使用现有进度 JSON。性能统计只输出到终端，不成为新的恢复字段。2,000 份真实牌谱只读验收中，单 worker 与默认 8×2 的判定逐项一致，吞吐分别为 2.37 与 9.65 份/秒。

检查范围：

- gzip、UTF-8、JSON、必要字段以及单场对局和各局的完整边界。
- 按物理牌的来源与移动检查库存；普通五和赤五合计最多四张、每门赤五最多一张。不能把摸牌、弃牌、副露中同一牌的重复出现累加为新牌。
- 点数以 100,000 总点数（含供托）守恒，核对每家分数在局间的衔接；允许负分终局和多家连续荣和。
- 使用 v4、四玩家、非 Oracle、无增强的 libriichi 提取路径，检查非空视角的动作/mask、局索引、形状、有限值及编码所需范围。
- GRP 输入非空、维度和局数符合选定模型容量，不实际计算奖励。

合法空玩家视角贡献零条样本，保留其他玩家样本。四家合计零样本或超过 60 局，以“本链路不可用/模型不兼容”隔离；明确数据错误隔离整谱。未知 Rust panic、程序错误、权限和内存故障直接停止，保留源文件。

隔离目标为 `data/grp/quarantine/<split>/<原相对路径>`，不覆盖已有目标。JSONL 报告记录源、目标和原因，不作为恢复依据，也不承诺跨文件事务意义上的恰好一次记录。

筛选进度保存 split 和最后已处理相对路径。恢复处理严格大于该排序边界的现存文件；该边界路径可能已经被移走，不能要求它仍存在或用列表下标定位。每 100 份提交一次，最后余量也提交。完成后再次启动退出，显式重新扫描只重扫源目录，不清空 quarantine。

并行版每提交 `commit_every` 份时显示 split、已处理/总数、隔离数、速度、ETA、游标，以及 gzip、Python 规则、libriichi 适用性三个阶段的平均耗时。

## 4. 全量样本生成

按 train、val 分别排序生成，两个 split 不混块。一次调用 libriichi 处理四份文件，按输入顺序消费结果；同一文件按 player_id 顺序和各视角原决策顺序展开。

一份牌谱的 GRP 历史前缀共同用于预测四家顺位分布；据此计算每局前后期望效用差，避免为四家重复同一奖励推理。最后边界使用真实终局顺位。六元样本为：

```text
obs, action, mask, steps_to_done, kyoku_reward, player_rank
```

`player_rank` 是本局结束后的顺位。`steps_to_done` 按既有动作折扣含义计算；预计算不把 gamma 折进奖励。生成阶段遇到异常或非有限奖励直接停止，不再隔离或静默丢样本。

使用预分配的定长样本数组作为 buffer，逐段填入；达到 16,384 条即编码提交，重置逻辑长度并继续使用。牌谱可以被 chunk 边界切开。避免将整块 observation 再 `stack` 成第二份同样大的数组。各 split 正常结束时将最后不足一块的余量提交；中断时不额外提交半满 buffer。

Q39–Q41 的日志方案已实现。启动/恢复、目录扫描开始、split 切换和完成均即时输出；默认每成功提交一个 chunk 输出一次，间隔可配置，最后余量仍输出。进度包括当前 split、完整提交牌谱数/总数及百分比、累计已提交 chunks/样本、本次耗时、牌谱/样本速度、当前 split 的 ETA、C/D 用量与预算、游标和样本 offset。内存 buffer 不算已提交成果，文件内断点不提前计作完整牌谱；恢复后的累计成果保留，速度仅统计本次增量。

日志同时显示 Rust/libriichi 提取、GRP 推理、Python 样本整理、编码压缩、写盘与进度提交五段累计耗时。统计只用于终端，及时 flush，不加入进度 JSON；不增加阻塞调用中的定时心跳。

## 5. chunk 格式

每个文件是一个 Zstd 压缩帧，解压后为短版本头部、样本数和固定顺序的数组。采用固定字节序与格式版本；不使用 pickle，不保存每样本来源坐标。

| 字段 | 固定表示 |
| --- | --- |
| 二值整行观测，417 通道 | 每样本 417 bit，按行补齐字节 |
| 二值逐牌观测，514 通道 | 每样本 `514*34` bit，按行补齐字节 |
| 连续整行观测，19 通道 | 每样本 19 个 float32 |
| 连续逐牌观测，62 通道 | 每样本 `62*34` 个 float32 |
| action / player_rank | uint8 / uint8 |
| mask | 每样本 46 bit，补齐为 6 字节 |
| steps_to_done | uint16 |
| kyoku_reward | float64 |

完整六元载荷为 10,764 B/样本，未计短头部和 Zstd 压缩。固定分类表以新 `common/codec.py` 内的 v4 布局为唯一代码定义，经源码核对和无损往返测试确认；生成时不重新按实际值猜测类别。

Q42、Q44 的编码优化已实现：标量通道直接提取第一列，逐牌通道按连续布局选择，二值数组尽早转 uint8，以减少整行副本和 reshape 复制。筛选继续完整检查观测语义；编码信任已筛选且稳定的输入，省去整行一致性、观测二值取值及 NaN/Inf 的重复逐值检查，保留形状、dtype、字段长度、标签/mask、有限奖励等必要检查。固定布局、float32/float64 精度、位序、chunk schema 和训练 reader 保持兼容，不能删除筛选依赖的完整校验函数。

读取时做必要的版本、样本数、长度检查，解压或格式错误直接抛出。完整观测和二值字段在目标设备恢复，CPU 保留紧凑表示。

已从 Rust v4 分支核对的布局规格如下，区间均左闭右开：

- 连续整行：`[7,15)`、`[23,25)`、`{27}`、`[717,723)`、`[889,891)`。
- 连续逐牌：`{131}`、`[324,327)`、`[519,522)`、`[714,717)`、`{835}`、`[961,1012)`。
- 二值整行由以下集合组成：`[4,7)`、`[15,23)`、`[32,35)`；`35+4i+j`（i=0..23，j=2,3）；`b+8i+j`（b=132,327,522，i=0..23，j=4..7）；`723+7i+j`（i=0..3，j=4..6）；`751+5i+4`（i=0..15）；`836+3i+j`（i=0..5，j=1,2）；`[854,860)`、`[861,871)`、`[872,874)`、`[878,885)`、`[887,889)`。
- 二值逐牌为 `0..1011` 中其余通道。四类分别 417、514、19、62 个，互斥且覆盖全部 1,012 个通道。

该规格已在此前 86,409 个观测的空间测量中通过类别检查；实现后仍需做精确往返验证。

## 6. C/D 布局与配额

```text
data/precomputed_chunks/
  progress.json
  train/
    c -> /mnt/c/MortalChunks/train/
    d -> /mnt/d/MortalChunks/train/
  val -> /mnt/c/MortalChunks/val/
```

C 总缓存预算 320 GiB，D 440 GiB；train 优先 D、再 C，val 在 C。按完整 chunk 的压缩字节数分配，单文件不跨盘，不自动突破预算。当前估计总量约 629 GiB，20% 规划余量后约 755 GiB；实际不保证低于此数。

目录或链接与配置冲突时停止，不覆盖未知内容。没有进度 JSON 时，仅清除本链路专用输出位置中的旧 chunk 和临时 chunk，从头初始化新的生成 ID；有 JSON 但损坏时直接报错。

## 7. 预计算提交与恢复

`progress.json` 是唯一提交依据，状态只有 `incomplete` 或 `complete`；运行过程中落盘的是可恢复的 incomplete 状态。

它保存生成 ID、必要配置、当前 split 与源文件/样本游标、各 split 的 chunk/样本计数和尾块数量、各盘已提交字节数、连续 chunk ID 的磁盘分配区间。大小不随每个 chunk 增加一条完整文件路径而线性膨胀；切换存储卷时才增加区间。

每块提交顺序：

1. 在选定盘的目标目录写临时文件并完成文件写入。
2. 在同一文件系统将其替换为正式 chunk。
3. 原子更新 progress，包含下一条未提交源样本的位置和该块配额用量。
4. JSON 成功后才能继续下一块；失败立即停止。

新增的 chunk 成功进度日志也只能出现在第 3 步成功之后。日志周期和线程数属于可调整运行参数，不进入不可变生成契约；旧 schema、生成 ID 和已提交 chunks 可直接继续使用，不因本轮优化要求重建。

恢复时只认可 JSON 已提交区间。先确定下一块 ID，清理该 ID 在两盘留下的未登记正式文件/临时文件，再按已提交字节数重新分配并生成。即使恢复前提高了某盘预算，同一未提交 ID 也不会在两盘留下两份有效副本。

游标落在牌谱中间时，重读该谱、重新计算其样本，再跳过已提交前缀；其后重新生成原 buffer 中尚未提交的样本。已完成 chunk 不回读验证或重算。全部 train、val 提交完成后再标记 complete；再次启动直接退出。

## 8. 训练读取、两级打乱与跨块 batch

从完成状态中的区间构造稳定的 chunk 列表与各块样本数，不扫描 Windows 目录来发现或采纳额外文件。

每次从头训练产生一个主 seed，先确定本轮 chunk 排列，再在访问某块时独立生成内部排列。随机派生以 seed、epoch、split 和 chunk ID 为输入，不依赖此前读取或预取了多少块。内部排列无需全量保存。

当前 chunk 的紧凑数组传到目标设备，每次按本批索引展开完整六元张量。CPU 仅预取下一块。跨 chunk 的 batch 先取当前块余样，再从下一块补齐；保留的是小批余样，不让切片无意持有旧的完整 dense chunk。

训练进度只按真正交给一次优化尝试的样本数推进。读盘、预取、解码或暂存余样都不能提前推进已消费游标。最后不足 batch 的样本照常更新。

恢复时由主 seed、epoch 和已消费样本位置重建顺序，并用各块样本数的前缀累计直接找到对应块及内部位置；不从头读盘解码，也不保存预取状态或余样张量。

## 9. 训练、验证与 checkpoint 提交

新训练循环独立实现已确认的损失和 AdamW 规则。float32 observation 进入模型后由 CUDA autocast 选择运算精度。AMP 若跳过一次权重更新，该批仍视为已消费，GradScaler 状态随 checkpoint 保存。

没有 `latest.pth` 时，按约定清空指定的 artifacts 训练子目录、随机初始化模型和种子，第一步训练前立即保存初始 checkpoint。输出目录必须是 artifacts 下的具体训练子目录，配置与源 GRP 权重位于该目录外。

checkpoint 同时保存三网络、优化器及学习率、AMP 状态、必要 RNG 状态、训练/验证 seed、epoch/样本游标、必要指标累计状态、数据生成 ID 与影响恢复语义的参数。已有文件损坏或与数据/模型不兼容时直接报错，不转成从头训练。

周期验证使用与训练 epoch 无关的固定验证排列，取前 131,072 个样本。全量验证同样按 chunk 消费，不更新网络；所有平均指标按真实样本数加权。

一次完整验证完成后，才原子保存对应训练位置的 checkpoint。epoch 只有在末尾全量验证完成后才标记完成。若周期边界与 epoch 末尾重合，执行末尾验证并合并为一次 checkpoint 提交，避免提前发布完成状态。验证中断会恢复上一份完整 checkpoint，重做未提交训练和必要验证。

TensorBoard 以已提交步数衔接，恢复时隐藏 checkpoint 之后的旧事件。只保留一个 latest checkpoint，不默认保存历史或 best 快照。

## 10. 恢复允许的变更

- 输入牌谱、冻结奖励权重、样本生成语义在同一预计算及恢复期间不变。
- 训练期间 chunks 不改写；生成 ID 区分同路径下重新生成的数据，不使用源码或逐文件哈希。
- 线程数、日志/保存频率、兼容设备、compile/benchmark 开关可调整；网络、batch size、AMP、损失和优化语义不静默改变。
- 设备切换必须支持当前 AMP 模式，否则报错；不承诺跨设备逐位复现。
- 可以增加磁盘预算，不能把预算降到已提交用量以下；磁盘写入错误保留已提交进度并停止。
- 完成训练后目标 epoch 数不增加就退出；增加时接续下一轮，不重新初始化 seed、模型或优化器。
- 同一预计算输出和同一训练输出只由一个写入进程使用；不增加复杂运行锁或源码身份校验。

## 11. 资源与实现顺序

当前 RTX 4090 上的六步合成测量：实际分配峰值约 4.18 GiB，显存池预留峰值约 4.45 GiB，进程 RSS 约 1.83 GiB。完整训练先按 6–8 GiB 显存、3–4 GiB 主机内存预算；生成按 6–8 GiB 主机内存预算。它们是分阶段预算，不相加，也不是完整新链路的实测上界。

Q39–Q44 的隔离实验在固定 128 份牌谱、真实 D 盘临时输出上，测得原配置约 1,556 样本/秒、7.2 GiB 峰值 RSS；16/1/4 配合候选编码优化约 2,890 样本/秒、5.1 GiB。这些是前期临时实验结果；正式版本的有限验收另见[验收记录](../precompute-progress-performance/acceptance.md)，不将 1.86 倍提升或 5.1 GiB 视为全量承诺。方法与原始记录见[性能报告](precompute-performance.md)。

实现顺序：

1. 新配置、固定布局和无损 codec，先验证真实与构造样本的精确往返。
2. 基础规则筛选、空视角边界、隔离及筛选续跑。
3. 六元样本、GRP 目标、定长 buffer、跨盘提交与预计算恢复。
4. 确定性读取、设备 batch 解码、跨块组批和直接定位。
5. 新训练/验证循环、单 checkpoint、TensorBoard 衔接。
6. 少量真实牌谱的端到端验证、GPU/内存与跨盘吞吐测量，补充操作说明。

验收重点包括：错误判据、合法空视角、float32/float64 无损往返、整数与 mask、最后余块/余批、文件中间断点、chunk/JSON 之间中断、两盘切换、损坏状态报错、种子和已消费位置恢复、验证中断与 epoch 完成边界。

测试使用临时目录和小型数据；不把百万份牌谱筛选、约 629 GiB 的全量生成或完整训练作为普通冒烟测试。纯 Python 改动不伪装成运行过 Rust 全矩阵；若实现中发现必须改变 Rust 接口，再按项目要求核对范围并构建验证。

## 12. 运行入口

主体入口已经实现。从 `~/Mortal` 执行，配置使用仓库内实际配置文件：

```bash
cd ~/Mortal
python mortal/precompute_pipeline/quarantine_logs/quarantine_logs.py --config "$PWD/mortal/precompute_pipeline/config.toml"
python mortal/precompute_pipeline/precompute/precompute.py --config "$PWD/mortal/precompute_pipeline/config.toml"
python mortal/precompute_pipeline/train/train.py --config "$PWD/mortal/precompute_pipeline/config.toml"
```

常规启动自动按各自持久化记录恢复；需要重新筛选时使用筛选入口的显式重扫选项。筛选入口默认使用配置中的 8×2 并行参数。

预计算入口已支持周期日志、五段计时、16/1 默认线程和编码优化，启动命令不变。恢复沿用旧进度，只报告新增成功提交的成果；文件名、样本数和磁盘用量以运行时状态为准。
