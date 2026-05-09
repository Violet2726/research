# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7358。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.215709。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.048251, -0.008444]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `True`，原因是 `all_conditions_met`。

## 实验概览

- 实验名：`dala_lite_same_context_main`
- 轨道：`same_context`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 当前轨道为 `same_context`，核心问题是在受限通信预算下，DALA-lite 是否能逼近 `all_to_all_full` 的效果。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；效率指标采用每千 token 准确率。
- 本实验固定比较 `mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence` 和 `dala_lite`，因此可以直接比较预算分配策略本身。

## 预算标定与进入门槛

- `gsm8k`：样本数 5，`p50(all_to_all_full_comm_tokens)`=171`，`round_budget_tokens`=68`。
- `strategyqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=218`，`round_budget_tokens`=87`。
- `hotpotqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=311`，`round_budget_tokens`=124`。
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.5971 | 0.00 | 2768.10 | 3.00 | 0.215709 |
| `all_to_all_full` | 0.7358 | 239.64 | 6091.15 | 6.00 | 0.120802 |
| `budget_random` | 0.7045 | 65.39 | 5643.56 | 6.00 | 0.124826 |
| `budget_confidence` | 0.6972 | 62.81 | 5641.27 | 6.00 | 0.123594 |
| `dala_lite` | 0.7081 | 53.46 | 5618.12 | 6.00 | 0.126035 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 119 | 4 |
| `budget_random` | 1.0060 | 0.6911 | 0.3353 | 0.0000 | 0.0000 | 0.6647 | 94 | 5 |
| `budget_confidence` | 1.0072 | 0.6674 | 0.3357 | 0.0000 | 0.0000 | 0.6643 | 88 | 5 |
| `dala_lite` | 0.8130 | 0.5681 | 0.2425 | 0.0000 | 0.0285 | 0.7290 | 104 | 12 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.4900 | 0.00 | 1359.53 | 0.360418 |
| `all_to_all_full` | 0.8500 | 184.13 | 3323.87 | 0.255726 |
| `budget_random` | 0.7533 | 42.38 | 2835.68 | 0.265663 |
| `budget_confidence` | 0.7300 | 42.18 | 2842.35 | 0.256829 |
| `dala_lite` | 0.7467 | 34.20 | 2806.57 | 0.266043 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6433 | 0.00 | 5463.88 | 0.117743 |
| `all_to_all_full` | 0.6467 | 309.15 | 11508.92 | 0.056188 |
| `budget_random` | 0.6600 | 88.38 | 11037.03 | 0.059799 |
| `budget_confidence` | 0.6500 | 83.42 | 11027.64 | 0.058943 |
| `dala_lite` | 0.6567 | 69.42 | 11004.68 | 0.059672 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6769 | 0.00 | 1081.79 | 0.625679 |
| `all_to_all_full` | 0.7031 | 221.31 | 2618.90 | 0.268455 |
| `budget_random` | 0.6987 | 65.41 | 2256.34 | 0.309656 |
| `budget_confidence` | 0.7162 | 62.83 | 2251.60 | 0.318066 |
| `dala_lite` | 0.7249 | 57.79 | 2244.73 | 0.322930 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00011`
- 问题预览：`Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 ...`
- 金标：`694`
- all_to_all_full：`694 / 1.0`
- dala_lite：`1314 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00067`
- 问题预览：`A treasure hunter found a buried treasure chest filled with gems. There were 175 diamonds, 35 fewer rubies than diamo...`
- 金标：`595`
- all_to_all_full：`595 / 1.0`
- dala_lite：`630 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00100`
- 问题预览：`Jerome had 4 friends who came to visit him on a certain day. The first friend pressed on the doorbell 20 times before...`
- 金标：`175`
- all_to_all_full：`175 / 1.0`
- dala_lite：`175 / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00107`
- 问题预览：`Frankie watches TV after he finishes his homework every night. On Monday and Tuesday, he watched a 1-hour episode of ...`
- 金标：`3`
- all_to_all_full：`3 / 1.0`
- dala_lite：`4 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00153`
- 问题预览：`Dave bought a large pack of french fries and ate fourteen before a hungry seagull stole the pack out of his hand. Whe...`
- 金标：`48`
- all_to_all_full：`42 / 0.0`
- dala_lite：`48 / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_same_context_main/confirmatory300/20260509T125958Z-dala_lite_same_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
