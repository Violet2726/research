# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7500。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.254977。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.083333, 0.083333]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `True`，原因是 `all_conditions_met`。

## 实验概览

- 实验名：`dala_lite_same_context_main`
- 轨道：`same_context`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5`

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
| `mv_3` | 0.6667 | 0.00 | 2614.62 | 3.00 | 0.254977 |
| `all_to_all_full` | 0.7500 | 237.80 | 5776.35 | 6.00 | 0.129840 |
| `budget_random` | 0.7000 | 61.98 | 5324.95 | 6.00 | 0.131457 |
| `budget_confidence` | 0.7167 | 59.92 | 5323.62 | 6.00 | 0.134620 |
| `dala_lite` | 0.7500 | 49.47 | 5303.97 | 6.00 | 0.141404 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 5 | 0 |
| `budget_random` | 0.9333 | 0.6579 | 0.3111 | 0.0000 | 0.0000 | 0.6889 | 3 | 1 |
| `budget_confidence` | 0.9333 | 0.6374 | 0.3111 | 0.0000 | 0.0000 | 0.6889 | 3 | 0 |
| `dala_lite` | 0.8000 | 0.5480 | 0.2444 | 0.0000 | 0.0222 | 0.7333 | 7 | 2 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6000 | 0.00 | 1291.65 | 0.464522 |
| `all_to_all_full` | 0.8500 | 181.55 | 3201.25 | 0.265521 |
| `budget_random` | 0.6500 | 38.85 | 2693.70 | 0.241304 |
| `budget_confidence` | 0.7500 | 38.95 | 2709.75 | 0.276778 |
| `dala_lite` | 0.7500 | 38.00 | 2703.35 | 0.277434 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7500 | 0.00 | 5458.30 | 0.137405 |
| `all_to_all_full` | 0.7500 | 312.00 | 11487.10 | 0.065291 |
| `budget_random` | 0.8000 | 84.10 | 11007.75 | 0.072676 |
| `budget_confidence` | 0.7500 | 81.35 | 10993.90 | 0.068220 |
| `dala_lite` | 0.7500 | 53.60 | 10950.50 | 0.068490 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6500 | 0.00 | 1093.90 | 0.594204 |
| `all_to_all_full` | 0.6500 | 219.85 | 2640.70 | 0.246147 |
| `budget_random` | 0.6500 | 63.00 | 2273.40 | 0.285915 |
| `budget_confidence` | 0.6500 | 59.45 | 2267.20 | 0.286697 |
| `dala_lite` | 0.7500 | 56.80 | 2258.05 | 0.332145 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00222`
- 问题预览：`Andy plants 90 geraniums and 40 fewer petunias that geraniums. How many flowers does he plant total?`
- 金标：`140`
- all_to_all_full：`140 / 1.0`
- dala_lite：`130 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00608`
- 问题预览：`Joe has $50 to buy an outfit for his new field trip. There is a 30% off sale at the clothing store. The shirt he pick...`
- 金标：`8`
- all_to_all_full：`8 / 1.0`
- dala_lite：`8 / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00945`
- 问题预览：`James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...`
- 金标：`17`
- all_to_all_full：`17 / 1.0`
- dala_lite：`15.5 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 4

- 数据集：`hotpotqa`
- 样本 ID：`5a89fea655429970aeb701eb`
- 问题预览：`In which film did Emilio Estevez star in in the same year as Nightmares`
- 金标：`The Outsiders`
- all_to_all_full：`nightmares / 0.0`
- dala_lite：`outsiders / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`

### 案例 5

- 数据集：`hotpotqa`
- 样本 ID：`5ae531ee5542990ba0bbb1ff`
- 问题预览：`Tommy's Honour was a drama film that included the actor who found success with what 2016 BBC miniseries?`
- 金标：`War & Peace`
- all_to_all_full：`war peace / 1.0`
- dala_lite：`war and peace / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_same_context_main/smoke20/20260509T125839Z-dala_lite_same_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
