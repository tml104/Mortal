# 评测牌谱可视化回放

从 `~/Mortal` 运行。静态回放只需要 Python 标准库和浏览器，无需启动模型或安装 PyTorch。

## 生成页面

```bash
python log-viewer/build_replays.py \
  --logs "$PWD/artifacts/mortal_epoch8_evaluations_20261012/run-zzi0os8o/logs" \
  --output "$PWD/artifacts/mortal_epoch8_evaluations_20261012/run-zzi0os8o/viewer"
```

重复执行会更新回放目录内的页面和资源，不修改原始牌谱、权重或评测结果。输出包含本地牌面图片，可以整体复制到其它机器。生成器沿用原查看器的图片，新的界面和回放逻辑位于 `log-viewer/replay*`，不依赖原生成的 JavaScript、jQuery 或 CSS。

## 打开回放

```bash
python -m http.server 8000 --bind 127.0.0.1 \
  --directory "$PWD/artifacts/mortal_epoch8_evaluations_20261012/run-zzi0os8o/viewer"
```

在 Windows 浏览器访问 <http://localhost:8000>，选择一个半庄。服务端启动命令与旧回放相同；如果已经打开旧页面，更新后刷新页面即可，必要时按 Ctrl+F5。退出服务用 Ctrl+C。页面不依赖 fetch，也可从复制后的目录直接打开 `index.html`。

## 牌桌与控制

- 中央显示当前场次、四家东南西北座次、实时分数、本场和供托数量；宝牌显示的是**指示牌**。
- P0～P3 是整场固定的玩家编号，座风随庄家变化，切换视角只改变桌面方向。
- 立直成立事件扣除 1,000 点并增加一根供托；和牌和流局累计日志中的 `deltas`，若存在 `scores` 则采用其权威值。和牌清空供托，流局保留供托。
- 下拉框选择小局，时间轴跳转事件，“上一步／下一步”跨小局连续回放。左右键同样可用，空格控制播放或暂停。
- 摸牌及当前打出的牌以黄色边框标记；被吃碰的牌在牌河留空格，保留其原巡目位置。四家手牌均公开，这是复盘视图。
- 布局按牌桌可用宽高自动缩放，右侧文字保持正常字号。支持 1280×720 窗口；最小布局尺寸为 960×580。候选动作和 JSON 较长时，在各自面板内滚动。

## 动作权重

右侧显示当前执行动作、合法候选动作及原始 Q 值，按 Q 值从大到小排序，标记实际执行项。Q 值是模型估计的动作价值，**不是概率**；本次评测使用贪心选择，但和牌规则保护等逻辑可能使实际动作不同于最高 Q 项。

`meta.q_values` 仅包含合法动作，顺序对应 `meta.mask_bits` 中从低到高的置位编号。掩码有 46 位，回放以 BigInt 解析，避免 JavaScript 32 位位运算丢失立直、吃碰杠及跳过等高位动作。动作语义与 `libriichi/src/agent/mortal.rs` 一致。

条形长度仅展示当前候选 Q 值的相对大小，不转换成概率。多候选杠牌的 `meta.kan_select` 单独显示；如果实际动作并未选择杠，则该表不标记“实际”。没有记录 Q 值的步骤会显示说明，不补造权重。日志仅记录实际发生的事件，不能还原未记录的其它玩家跳过副露时的权重。

本次旧 baseline 在原始 JSON 中仍叫 `Mortal-baseline-epoch3.46`，其实际训练进度是完成 3 个 epoch、第 4 个 epoch 的 46.3397%。界面显示明确的进度名称，原始牌谱字段不变。

## 验证

```bash
node --test log-viewer/tests/replay-state.test.js
python -m py_compile log-viewer/build_replays.py
node --check log-viewer/replay-state.js
node --check log-viewer/replay.js
```
