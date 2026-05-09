# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7127。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.291378。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.096408, -0.035870]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `False`，原因是 `gate_not_met`。

## 实验概览

- 实验名：`dala_lite_split_context_main`
- 轨道：`split_context`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 当前轨道为 `split_context`，核心问题是在受限通信预算下，DALA-lite 是否能逼近 `all_to_all_full` 的效果。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；效率指标采用每千 token 准确率。
- 本实验固定比较 `mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence` 和 `dala_lite`，因此可以直接比较预算分配策略本身。

## 预算标定与进入门槛

- `strategyqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=204`，`round_budget_tokens`=81`。
- `hotpotqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=304`，`round_budget_tokens`=121`。
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`False`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.6106 | 0.00 | 2095.51 | 3.00 | 0.291378 |
| `all_to_all_full` | 0.7127 | 281.13 | 4806.61 | 6.00 | 0.148268 |
| `budget_random` | 0.6276 | 75.69 | 4335.14 | 6.00 | 0.144770 |
| `budget_confidence` | 0.6635 | 72.41 | 4328.11 | 6.00 | 0.153304 |
| `dala_lite` | 0.6484 | 61.42 | 4311.73 | 6.00 | 0.150379 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 67 | 13 |
| `budget_random` | 0.9603 | 0.7352 | 0.3201 | 0.0000 | 0.0000 | 0.6799 | 28 | 19 |
| `budget_confidence` | 0.9584 | 0.7026 | 0.3195 | 0.0000 | 0.0000 | 0.6805 | 41 | 13 |
| `dala_lite` | 0.8904 | 0.6005 | 0.2476 | 0.0000 | 0.0491 | 0.7032 | 46 | 26 |

## 分数据集表现

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.4433 | 0.00 | 2782.67 | 0.159319 |
| `all_to_all_full` | 0.5933 | 340.72 | 6304.49 | 0.094113 |
| `budget_random` | 0.4533 | 86.06 | 5733.80 | 0.079063 |
| `budget_confidence` | 0.5033 | 82.67 | 5726.40 | 0.087897 |
| `dala_lite` | 0.4667 | 68.17 | 5705.73 | 0.081789 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8297 | 0.00 | 1195.31 | 0.694127 |
| `all_to_all_full` | 0.8690 | 203.07 | 2844.33 | 0.305519 |
| `budget_random` | 0.8559 | 62.09 | 2502.83 | 0.341971 |
| `budget_confidence` | 0.8734 | 58.97 | 2496.29 | 0.349864 |
| `dala_lite` | 0.8865 | 52.59 | 2485.52 | 0.356652 |

## 典型案例

### 案例 1

- 数据集：`hotpotqa`
- 样本 ID：`5a7129685542994082a3e5fa`
- 问题预览：`Which "Blackzilians" fighter is currently competing in the Middleweight division of Ultimate Fighting Championship?`
- 金标：`Vitor Belfort`
- all_to_all_full：`vitor belfort / 1.0`
- dala_lite：`rashad evans / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 2

- 数据集：`hotpotqa`
- 样本 ID：`5a7349125542994cef4bc505`
- 问题预览：`Baadshah is an Indian action comedy film that was inspired by what Hong Kong action movie starring Jackie Chan and Ri...`
- 金标：`Mr. Nice Guy`
- all_to_all_full：`mr nice guy / 1.0`
- dala_lite：`rush hour / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 3

- 数据集：`hotpotqa`
- 样本 ID：`5a74106b55429979e288289e`
- 问题预览：`Where is the company that Sachin Warrier worked for as a software engineer headquartered?`
- 金标：`Mumbai`
- all_to_all_full：`mumbai / 1.0`
- dala_lite：`kochi / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 4

- 数据集：`hotpotqa`
- 样本 ID：`5a7509175542993748c897a3`
- 问题预览：`What group of Elektra Records recording artists are known to be an indie pop band?`
- 金标：`Saint Motel`
- all_to_all_full：`saint motel / 1.0`
- dala_lite：`fitz and tantrums / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 5

- 数据集：`hotpotqa`
- 样本 ID：`5a76394c5542994ccc918725`
- 问题预览：`When was the band who composited "Discipline" formed?`
- 金标：`1968`
- all_to_all_full：`1968 / 1.0`
- dala_lite：`1987 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_split_context_main/confirmatory300/20260509T130038Z-dala_lite_split_context_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
