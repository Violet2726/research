# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7250。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.276059。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.200000, 0.050000]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `False`，原因是 `gate_not_met`。

## 实验概览

- 实验名：`dala_lite_split_context_main`
- 轨道：`split_context`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 当前轨道为 `split_context`，核心问题是在受限通信预算下，DALA-lite 是否能逼近 `all_to_all_full` 的效果。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；效率指标采用每千 token 准确率。
- 本实验固定比较 `mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence` 和 `dala_lite`，因此可以直接比较预算分配策略本身。

## 预算标定与进入门槛

- `strategyqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=214`，`round_budget_tokens`=85`。
- `hotpotqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=304`，`round_budget_tokens`=121`。
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.5500 | 0.00 | 1992.33 | 3.00 | 0.276059 |
| `all_to_all_full` | 0.7250 | 275.02 | 4585.82 | 6.00 | 0.158096 |
| `budget_random` | 0.5500 | 76.17 | 4136.80 | 6.00 | 0.132953 |
| `budget_confidence` | 0.6500 | 72.75 | 4132.10 | 6.00 | 0.157305 |
| `dala_lite` | 0.6500 | 61.98 | 4101.38 | 6.00 | 0.158483 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| `budget_random` | 0.9750 | 0.7402 | 0.3250 | 0.0000 | 0.0000 | 0.6750 | 3 | 3 |
| `budget_confidence` | 0.9750 | 0.7115 | 0.3250 | 0.0000 | 0.0000 | 0.6750 | 6 | 2 |
| `dala_lite` | 0.9000 | 0.6117 | 0.2583 | 0.0000 | 0.0417 | 0.7000 | 5 | 1 |

## 分数据集表现

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.2500 | 0.00 | 2783.30 | 0.089821 |
| `all_to_all_full` | 0.6000 | 339.45 | 6325.10 | 0.094860 |
| `budget_random` | 0.2500 | 89.10 | 5773.35 | 0.043302 |
| `budget_confidence` | 0.4500 | 82.50 | 5749.65 | 0.078266 |
| `dala_lite` | 0.4500 | 67.10 | 5707.85 | 0.078839 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8500 | 0.00 | 1201.35 | 0.707537 |
| `all_to_all_full` | 0.8500 | 210.60 | 2846.55 | 0.298607 |
| `budget_random` | 0.8500 | 63.25 | 2500.25 | 0.339966 |
| `budget_confidence` | 0.8500 | 63.00 | 2514.55 | 0.338033 |
| `dala_lite` | 0.8500 | 56.85 | 2494.90 | 0.340695 |

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
- 样本 ID：`5a89fea655429970aeb701eb`
- 问题预览：`In which film did Emilio Estevez star in in the same year as Nightmares`
- 金标：`The Outsiders`
- all_to_all_full：`outsiders / 1.0`
- dala_lite：`rated x / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 3

- 数据集：`hotpotqa`
- 样本 ID：`5ae0132d55429925eb1afc00`
- 问题预览：`The Soul of Buddha is a 1918 American silent romance film shot in a borough that is the western terminus of what?`
- 金标：`the George Washington Bridge`
- all_to_all_full：`george washington bridge / 1.0`
- dala_lite：`fort lee new jersey / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 4

- 数据集：`hotpotqa`
- 样本 ID：`5ae4c01e55429913cc2044f3`
- 问题预览：`Which Captain launched the attack which led to more casualties than any other incident in the war fought between the ...`
- 金标：`Captain John Underhill`
- all_to_all_full：`captain john underhill / 1.0`
- dala_lite：`willem kieft / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 5

- 数据集：`hotpotqa`
- 样本 ID：`5ae6b6065542991bbc976168`
- 问题预览：`Out of the actors who have played the role of Luc Deveraux in the Universal Soldier franchise, which actor has also s...`
- 金标：`Scott Adkins`
- all_to_all_full：`scott adkins / 1.0`
- dala_lite：`matt battaglia / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_split_context_main/smoke20/20260509T125849Z-dala_lite_split_context_main-smoke20-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
