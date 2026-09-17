# Issue tracker: Local Markdown

本项目的任务与规格保存在仓库内的 `.scratch/`，当前使用本地文件工作流。

## 文件约定

- 每个功能使用一个目录：`.scratch/<feature-slug>/`。
- 规格：`.scratch/<feature-slug>/spec.md`。
- 实现任务：`.scratch/<feature-slug>/issues/<NN>-<slug>.md`；每项任务一个文件，从 `01` 开始编号，新增时使用未占用的编号。
- 在任务文件顶部附近使用 `Status:` 行记录分诊状态；标签含义见同目录的 `triage-labels.md`，新任务默认为 `needs-triage`。
- 评论和讨论按时间追加到文件末尾的 `## Comments` 下。

## 技能操作映射

- “发布到任务跟踪器”：按以上约定创建本地 Markdown 文件及所需目录。
- “获取相关任务”：读取用户提供的文件路径；若只给编号，在对应功能目录定位，编号跨目录重复时先明确功能。
- “添加评论”或“更新状态”：编辑对应任务文件，保留既有正文与讨论历史。

## Wayfinding 操作

供 `/wayfinder` 使用；其探索任务采用下述独立状态约定。

- Map：`.scratch/<effort>/map.md`，包含 Notes / Decisions-so-far / Fog。
- 子任务：`.scratch/<effort>/issues/<NN>-<slug>.md`，从 `01` 编号，正文记录待回答的问题。
- 类型：顶部附近的 `Type:` 为 `research`、`prototype`、`grilling` 或 `task`。
- 状态：顶部附近的 `Status:` 为 `open`、`claimed` 或 `resolved`，新探索任务为 `open`。
- 阻塞关系：`Blocked by: NN, NN` 指向同一 effort 下的任务；全部为 `resolved` 时解除阻塞。
- Frontier：扫描对应 issues 目录，选择编号最小、状态为 `open` 且无阻塞的任务。
- Claim：开始工作前写入并保存 `Status: claimed`。
- Resolve：在 `## Answer` 下追加答案，设置 `Status: resolved`，再向 map 的 Decisions-so-far 追加摘要与相对链接。
