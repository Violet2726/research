# 预算通信科研报告

## 摘要

- 总体准确率最高的方法是 `all_to_all_full`，准确率为 0.7400。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.306236。
- `dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.120000, -0.010000]（探索性）`。
- 当前阶段对 Full DALA 的进入判断为 `False`，原因是 `gate_not_met`。

## 实验概览

- 实验名：`dala_lite_split_context_main`
- 轨道：`split_context`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 当前轨道为 `split_context`，核心问题是在受限通信预算下，DALA-lite 是否能逼近 `all_to_all_full` 的效果。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；效率指标采用每千 token 准确率。
- 本实验固定比较 `mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence` 和 `dala_lite`，因此可以直接比较预算分配策略本身。

## 预算标定与进入门槛

- `strategyqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=214`，`round_budget_tokens`=85`。
- `hotpotqa`：样本数 5，`p50(all_to_all_full_comm_tokens)`=304`，`round_budget_tokens`=121`。
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`False`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.6050 | 0.00 | 1975.60 | 3.00 | 0.306236 |
| `all_to_all_full` | 0.7400 | 272.94 | 4541.28 | 6.00 | 0.162949 |
| `budget_random` | 0.6150 | 73.88 | 4087.36 | 6.00 | 0.150464 |
| `budget_confidence` | 0.6900 | 70.50 | 4083.96 | 6.00 | 0.168954 |
| `dala_lite` | 0.6750 | 63.82 | 4073.14 | 6.00 | 0.165720 |

## 机制诊断

- 如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。
- 若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。

| 方法 | 平均胜者集合大小 | 预算利用率 | Full 比例 | Summary 比例 | Keywords 比例 | Silence 比例 | 纠正题数 | 伤害题数 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 31 | 4 |
| `budget_random` | 0.9500 | 0.7209 | 0.3167 | 0.0000 | 0.0000 | 0.6833 | 12 | 10 |
| `budget_confidence` | 0.9500 | 0.6887 | 0.3167 | 0.0000 | 0.0000 | 0.6833 | 22 | 5 |
| `dala_lite` | 0.9450 | 0.6274 | 0.2650 | 0.0000 | 0.0500 | 0.6850 | 23 | 9 |

## 分数据集表现

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.4000 | 0.00 | 2760.02 | 0.144926 |
| `all_to_all_full` | 0.6100 | 343.02 | 6249.54 | 0.097607 |
| `budget_random` | 0.3900 | 84.75 | 5682.16 | 0.068636 |
| `budget_confidence` | 0.5000 | 80.41 | 5669.19 | 0.088196 |
| `dala_lite` | 0.4800 | 70.50 | 5663.79 | 0.084749 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8100 | 0.00 | 1191.18 | 0.679998 |
| `all_to_all_full` | 0.8700 | 202.86 | 2833.03 | 0.307092 |
| `budget_random` | 0.8400 | 63.01 | 2492.55 | 0.337004 |
| `budget_confidence` | 0.8800 | 60.59 | 2498.73 | 0.352179 |
| `dala_lite` | 0.8700 | 57.14 | 2482.50 | 0.350453 |

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
- 样本 ID：`5a76394c5542994ccc918725`
- 问题预览：`When was the band who composited "Discipline" formed?`
- 金标：`1968`
- all_to_all_full：`1968 / 1.0`
- dala_lite：`1987 / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 3

- 数据集：`hotpotqa`
- 样本 ID：`5a7d1765554299452d57bade`
- 问题预览：`The 1919 Mississippi gubernatorial election Incumbent Democrat was a master of what?`
- 金标：`filibuster`
- all_to_all_full：`filibuster / 1.0`
- dala_lite：`filibuster / 1.0`
- 解释：`dala_lite 在同预算下优于 budget_confidence。`

### 案例 4

- 数据集：`hotpotqa`
- 样本 ID：`5a7f244255429934daa2fcec`
- 问题预览：`St. John's College, Belize offers an education in a tradition in which what three subjects were the core?`
- 金标：`Grammar, logic, and rhetoric`
- all_to_all_full：`grammar logic and rhetoric / 1.0`
- dala_lite：`liberal arts and science / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`

### 案例 5

- 数据集：`hotpotqa`
- 样本 ID：`5a8051265542992bc0c4a6f8`
- 问题预览：`Tommy Swerdlow co-wrote the screenplay of what film directed by Jon Turteltaub?`
- 金标：`Cool Runnings`
- all_to_all_full：`cool runnings / 1.0`
- dala_lite：`snow dogs / 0.0`
- 解释：`dala_lite 在该题上弱于 all_to_all_full。`


## 结论与建议

- 若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。
- 如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。
- 正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。

## 局限性

- 当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。
- 当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。

## 复现与产物说明

- 运行目录：`runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。

## 图表资产

### 预算通信成本-性能前沿

![预算通信成本-性能前沿](../../runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。*

### 预算通信效率排序

![预算通信效率排序](../../runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 预算通信跨数据集表现

![预算通信跨数据集表现](../../runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各预算通信方法在不同数据集上的准确率分布。*

### 消息包模式构成

![消息包模式构成](../../runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/packet_mode_mix.svg)

*总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。*

### 预算利用率权衡

![预算利用率权衡](../../runs/budget_comm/dala_lite_split_context_main/pilot100/20260509T125924Z-dala_lite_split_context_main-pilot100-xiaomimimo-mimo-v2.5/figures/budget_utilization_tradeoff.svg)

*总体准确率相对于平均预算利用率的变化。*
