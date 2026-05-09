# 通信必要性科研报告

## 摘要

- 总体 Joint F1 最优的方法是 `full_context_single`，Joint F1 为 0.6721。
- 单位成本效率最优的方法是 `full_context_single`，每千 token 得分为 0.347751。
- 关键对照中提升最大的比较是 `full_context_single - evidence_exchange`，Joint F1 差值为 0.1993。

## 实验概览

- 实验名：`hotpotqa_split_context_communication_necessity`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5`
- 样本数：`100`

## 研究问题与实验设计

- 本实验聚焦 split-context 场景，核心问题是：通信是否真正改善了答案质量与证据质量，而不是只改善其中一部分。
- 主指标包括 Answer EM / F1、Supporting Facts F1 和 Joint F1；成本侧同时记录平均通信 token / 题和平均总 token / 题。
- 由于 split-context 通信存在强设计约束，本报告特别关注关键对照差值，以判断收益来自答案交换、证据交换还是完整消息包交换。

## 总体结果

| 方法 | Answer EM | Answer F1 | Supporting F1 | Joint F1 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `full_context_single` | 0.6900 | 0.8221 | 0.8002 | 0.6721 | 0.00 | 1984.18 | 1.00 |
| `split_no_comm_mv3` | 0.3900 | 0.5093 | 0.4585 | 0.2908 | 0.00 | 2990.28 | 3.00 |
| `answer_only_exchange` | 0.4900 | 0.6159 | 0.5520 | 0.4062 | 59.00 | 6568.61 | 6.00 |
| `evidence_exchange` | 0.5800 | 0.7004 | 0.6230 | 0.4727 | 367.30 | 6868.97 | 6.00 |
| `full_packet_exchange` | 0.5900 | 0.7190 | 0.6184 | 0.4781 | 1008.94 | 7479.70 | 6.00 |

## 关键对照差值

- split 视图数：`300`；full-context 参考视图数：`100`。
- 若 Joint F1 改善主要来自 Supporting F1，而 Answer EM / F1 提升有限，说明通信更像是在修复证据而非修复最终答案。

| 比较 | Answer EM 差值 | Supporting F1 差值 | Joint F1 差值 | 通信 token 差值 |
| --- | --- | --- | --- | --- |
| `evidence_exchange - split_no_comm_mv3` | 0.1900 | 0.1645 | 0.1819 | 367.30 |
| `full_packet_exchange - answer_only_exchange` | 0.1000 | 0.0664 | 0.0719 | 949.94 |
| `full_context_single - evidence_exchange` | 0.1100 | 0.1772 | 0.1993 | -367.30 |

## 分数据集表现

### hotpotqa

| 方法 | Answer EM | Supporting F1 | Joint F1 | 平均通信 token / 题 | 平均总 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `full_context_single` | 0.6900 | 0.8002 | 0.6721 | 0.00 | 1984.18 |
| `split_no_comm_mv3` | 0.3900 | 0.4585 | 0.2908 | 0.00 | 2990.28 |
| `answer_only_exchange` | 0.4900 | 0.5520 | 0.4062 | 59.00 | 6568.61 |
| `evidence_exchange` | 0.5800 | 0.6230 | 0.4727 | 367.30 | 6868.97 |
| `full_packet_exchange` | 0.5900 | 0.6184 | 0.4781 | 1008.94 | 7479.70 |

## 结论与建议

- 正式比较 split-context 通信方法时，应优先以 Joint F1 为主结论，并同步报告 Supporting F1，避免把单纯的答案修正误判为真正的信息整合收益。
- 如果某种交换方式带来更高 Joint F1，但通信成本也明显上升，应结合成本-性能前沿图判断它是否值得成为默认方案。
- 进入更大样本 phase 前，应优先复核关键对照差值是否稳定，而不是只依据单轮 smoke 结论推进。

## 局限性

- 当前报告主要面向当前 phase 的机制验证，不直接等同于更大样本上的最终论文结论。
- split-context 实验高度依赖任务视图切分方式，因此结论应和具体 view 设计一起解读。

## 复现与产物说明

- 运行目录：`runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`、`hotpot_predictions/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。

## 图表资产

### 通信必要性成本-性能前沿

![通信必要性成本-性能前沿](../../runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体 Joint F1 相对于平均总 token 的位置关系。*

### 通信必要性效率排序

![通信必要性效率排序](../../runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 得分的总体效率排序。*

### 通信必要性跨数据集表现

![通信必要性跨数据集表现](../../runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各 split-context 方法在不同数据集上的 Joint F1 分布。*

### 联合指标剖面

![联合指标剖面](../../runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5/figures/joint_metric_panel.svg)

*总体层面 Answer F1、Supporting F1 和 Joint F1 的并列对比。*

### 关键对照差值

![关键对照差值](../../runs/comm_necessary/hotpotqa_split_context_communication_necessity/pilot100/20260509T125909Z-hotpotqa_split_context_communication_necessity-pilot100-xiaomimimo-mimo-v2.5/figures/delta_vs_controls.svg)

*诊断文件中记录的关键控制组差值，重点关注 Joint F1 变化。*
