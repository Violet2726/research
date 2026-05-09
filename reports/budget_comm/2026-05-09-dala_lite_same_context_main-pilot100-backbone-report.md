# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7133。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.231380。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.046667, 0.013333]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `True`，原因是 `all_conditions_met`。

## 实验概览

- 实验名：`dala_lite_same_context_main`
- 轨道：`same_context`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5`

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
| `mv_3` | 0.6033 | 0.00 | 2607.54 | 3.00 | 0.231380 |
| `all_to_all_full` | 0.7133 | 236.79 | 5761.93 | 6.00 | 0.123801 |
| `budget_random` | 0.6900 | 66.58 | 5325.38 | 6.00 | 0.129568 |
| `budget_confidence` | 0.6867 | 64.09 | 5321.89 | 6.00 | 0.129027 |
| `dala_lite` | 0.6967 | 54.06 | 5302.74 | 6.00 | 0.131379 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 35 | 2 |
| `budget_random` | 1.0000 | 0.7099 | 0.3333 | 0.0000 | 0.0000 | 0.6667 | 28 | 2 |
| `budget_confidence` | 1.0033 | 0.6861 | 0.3344 | 0.0000 | 0.0000 | 0.6656 | 27 | 2 |
| `dala_lite` | 0.8367 | 0.5830 | 0.2522 | 0.0000 | 0.0267 | 0.7211 | 33 | 5 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5300 | 0.00 | 1353.23 | 0.391656 |
| `all_to_all_full` | 0.8600 | 191.11 | 3340.14 | 0.257474 |
| `budget_random` | 0.7500 | 43.68 | 2828.58 | 0.265151 |
| `budget_confidence` | 0.7400 | 44.10 | 2842.77 | 0.260309 |
| `dala_lite` | 0.7600 | 34.04 | 2806.54 | 0.270796 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6500 | 0.00 | 5397.59 | 0.120424 |
| `all_to_all_full` | 0.6500 | 305.42 | 11361.42 | 0.057211 |
| `budget_random` | 0.6700 | 89.30 | 10904.20 | 0.061444 |
| `budget_confidence` | 0.6600 | 85.59 | 10890.50 | 0.060603 |
| `dala_lite` | 0.6600 | 65.38 | 10860.49 | 0.060771 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6300 | 0.00 | 1071.79 | 0.587802 |
| `all_to_all_full` | 0.6300 | 213.83 | 2584.23 | 0.243786 |
| `budget_random` | 0.6500 | 66.75 | 2243.35 | 0.289745 |
| `budget_confidence` | 0.6600 | 62.59 | 2232.40 | 0.295646 |
| `dala_lite` | 0.6700 | 62.75 | 2241.19 | 0.298948 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00100`
- 问题预览：`Jerome had 4 friends who came to visit him on a certain day. The first friend pressed on the doorbell 20 times before...`
- 金标：`175`
- all_to_all_full：`175 / 1.0`
- dala_lite：`175 / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00107`
- 问题预览：`Frankie watches TV after he finishes his homework every night. On Monday and Tuesday, he watched a 1-hour episode of ...`
- 金标：`3`
- all_to_all_full：`3 / 1.0`
- dala_lite：`4 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00222`
- 问题预览：`Andy plants 90 geraniums and 40 fewer petunias that geraniums. How many flowers does he plant total?`
- 金标：`140`
- all_to_all_full：`140 / 1.0`
- dala_lite：`130 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00303`
- 问题预览：`Elise has been selling her Dad's collection of 250 books for three years. Each book sells at 20$, and she sold twice ...`
- 金标：`1300`
- all_to_all_full：`1300 / 1.0`
- dala_lite：`4000 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00364`
- 问题预览：`There are 96 fourth-graders at Small Tree School. 43 of them are girls. On Friday, 5 fourth-grade girls and 4 fourth-...`
- 金标：`49`
- all_to_all_full：`49 / 1.0`
- dala_lite：`44 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_same_context_main/pilot100/20260509T125906Z-dala_lite_same_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
