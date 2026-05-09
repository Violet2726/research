# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `mv_6`，准确率为 0.8468。
- 触发策略中表现最佳的是 `disagreement`。
- 共享前缀机制的平均 token 节省比例约为 0.5885。
- 当前推荐的下一轮默认策略为 `hybrid_trigger`。

## 实验概览

- 实验名：`trigger_early_exit_main`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`895834.00`，独立重跑 token=`1927824.00`，节省比例=`0.5353`。
- `hotpotqa`：共享执行实际 token=`3369101.00`，独立重跑 token=`8989572.00`，节省比例=`0.6252`。
- `overall`：共享执行实际 token=`4784715.00`，独立重跑 token=`12174858.00`，节省比例=`0.6070`。
- `strategyqa`：共享执行实际 token=`519780.00`，独立重跑 token=`1257462.00`，节省比例=`0.5866`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8335 | 0.00 | 2523.53 | 0.330305 |
| `always` | 0.8347 | 3248.14 | 5771.67 | 0.144627 |
| `disagreement` | 0.8359 | 602.27 | 3125.80 | 0.267435 |
| `confidence` | 0.8323 | 107.13 | 2630.66 | 0.316396 |
| `hybrid` | 0.8359 | 634.55 | 3158.07 | 0.264701 |
| `mv_6` | 0.8468 | 0.00 | 5046.90 | 0.167787 |
| `sc_6` | 0.8468 | 0.00 | 5046.90 | 0.167787 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.0084 | 1.0000 | 0.9916 | 0.0000 |
| `disagreement` | 0.1448 | 0.8552 | 0.0583 | 1.0000 | 0.1363 | 0.0000 |
| `confidence` | 0.0277 | 0.9723 | 0.0435 | 0.1429 | 0.0265 | 0.8571 |
| `hybrid` | 0.1592 | 0.8408 | 0.0530 | 1.0000 | 0.1508 | 0.0000 |

## VoC 诊断

- 推荐默认策略：`hybrid_trigger`。
- 相对 `always_communicate` 的准确率下降：`-0.001206`；总 token 降低比例：`0.452832`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 1.0000 | 0.9843 | 1.0000 | 3248.14 |
| `disagreement` | 1.0000 | 0.8333 | 0.9000 | 0.1448 | 602.27 |
| `confidence` | 0.1429 | 0.3333 | 0.8696 | 0.0277 | 107.13 |
| `hybrid` | 1.0000 | 0.8333 | 0.9091 | 0.1592 | 634.55 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.9667 | 0.00 | 1080.21 | 0.894885 |
| `always` | 0.9667 | 1905.90 | 2986.11 | 0.323721 |
| `disagreement` | 0.9667 | 99.66 | 1179.88 | 0.819295 |
| `confidence` | 0.9667 | 0.00 | 1080.21 | 0.894885 |
| `hybrid` | 0.9667 | 99.66 | 1179.88 | 0.819295 |
| `mv_6` | 0.9733 | 0.00 | 2159.41 | 0.450740 |
| `sc_6` | 0.9733 | 0.00 | 2159.41 | 0.450740 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7100 | 0.00 | 5252.10 | 0.135184 |
| `always` | 0.7033 | 5978.23 | 11230.34 | 0.062628 |
| `disagreement` | 0.7067 | 1350.65 | 6602.76 | 0.107026 |
| `confidence` | 0.7067 | 236.01 | 5488.11 | 0.128763 |
| `hybrid` | 0.7067 | 1391.93 | 6644.04 | 0.106361 |
| `mv_6` | 0.7200 | 0.00 | 10505.86 | 0.068533 |
| `sc_6` | 0.7200 | 0.00 | 10505.86 | 0.068533 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8210 | 0.00 | 839.78 | 0.977594 |
| `always` | 0.8341 | 1430.00 | 2269.78 | 0.367463 |
| `disagreement` | 0.8341 | 280.29 | 1120.07 | 0.744654 |
| `confidence` | 0.8210 | 78.64 | 918.41 | 0.893889 |
| `hybrid` | 0.8341 | 343.06 | 1182.84 | 0.705135 |
| `mv_6` | 0.8472 | 0.00 | 1678.15 | 0.504819 |
| `sc_6` | 0.8472 | 0.00 | 1678.15 | 0.504819 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00062`
- 问题预览：`If Marcy works for the same company for 40 years, she gets an annual pension of $50,000/year. Starting after 20 years...`
- 金标：`25000`
- mv_3：`25000 / 1.0`
- always_communicate：`25000 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00066`
- 问题预览：`There are four schools competing at a basketball tournament. Each school has sent a girls’ basketball team and a boys...`
- 金标：`48`
- mv_3：`48 / 1.0`
- always_communicate：`48 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00187`
- 问题预览：`Mandy owes Benedict $100. They agreed to have monthly interest of 2%. If Mandy was able to pay it after 3 months, how...`
- 金标：`106`
- mv_3：`106.12 / 0.0`
- always_communicate：`106.12 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00371`
- 问题预览：`A shoe store was having a weekend sale on a brand of popular tennis shoes. On Friday the store sold 14 pairs of tenni...`
- 金标：`50`
- mv_3：`100 / 0.0`
- always_communicate：`100 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00423`
- 问题预览：`Cole wanted to buy new jeans for a dance contest. At the store, he couldn't decide between tattered jeans and jogger ...`
- 金标：`8`
- mv_3：`-8 / 0.0`
- always_communicate：`-8 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/trigger_early_exit_main/confirmatory300/20260509T130017Z-trigger_early_exit_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
