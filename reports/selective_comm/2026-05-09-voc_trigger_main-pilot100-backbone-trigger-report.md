# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.7267。
- 触发策略中表现最佳的是 `always`。
- 共享前缀机制的平均 token 节省比例约为 0.7378。
- 当前推荐的下一轮默认策略为 `voc_trigger_v2`。

## 实验概览

- 实验名：`voc_trigger_main`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`290870.00`，独立重跑 token=`1095342.00`，节省比例=`0.7344`。
- `hotpotqa`：共享执行实际 token=`671290.00`，独立重跑 token=`2892539.00`，节省比例=`0.7679`。
- `overall`：共享执行实际 token=`1188665.00`，独立重跑 token=`4741425.00`，节省比例=`0.7493`。
- `strategyqa`：共享执行实际 token=`226505.00`，独立重跑 token=`753544.00`，节省比例=`0.6994`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6467 | 0.00 | 2483.37 | 0.260399 |
| `always` | 0.7267 | 1478.85 | 3962.22 | 0.183399 |
| `disagreement` | 0.7167 | 550.34 | 3033.71 | 0.236235 |
| `confidence` | 0.6467 | 13.36 | 2496.73 | 0.259006 |
| `hybrid` | 0.7167 | 563.86 | 3047.23 | 0.235186 |
| `voc_v2` | 0.7200 | 781.49 | 3264.86 | 0.220530 |
| `mv_6` | 0.6433 | 0.00 | 4972.00 | 0.129391 |
| `sc_6` | 0.6433 | 0.00 | 4972.00 | 0.129391 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.0900 | 1.0000 | 0.9100 | 0.0000 |
| `disagreement` | 0.3433 | 0.6567 | 0.2330 | 0.8889 | 0.2633 | 0.1111 |
| `confidence` | 0.0100 | 0.9900 | 0.0000 | 0.0000 | 0.0100 | 1.0000 |
| `hybrid` | 0.3533 | 0.6467 | 0.2264 | 0.8889 | 0.2733 | 0.1111 |
| `voc_v2` | 0.5133 | 0.4867 | 0.1623 | 0.9259 | 0.4300 | 0.0741 |

## VoC 诊断

- 推荐默认策略：`voc_trigger_v2`。
- 相对 `always_communicate` 的准确率下降：`0.006667`；总 token 降低比例：`0.176001`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 1.0000 | 0.9000 | 1.0000 | 1478.85 |
| `disagreement` | 0.8889 | 1.0000 | 0.7379 | 0.3433 | 550.34 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0100 | 13.36 |
| `hybrid` | 0.8889 | 1.0000 | 0.7453 | 0.3533 | 563.86 |
| `voc_v2` | 0.9259 | 1.0000 | 0.8182 | 0.5133 | 781.49 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5400 | 0.00 | 1218.97 | 0.442997 |
| `always` | 0.7800 | 1689.73 | 2908.70 | 0.268161 |
| `disagreement` | 0.7600 | 1039.61 | 2258.58 | 0.336495 |
| `confidence` | 0.5400 | 0.00 | 1218.97 | 0.442997 |
| `hybrid` | 0.7600 | 1039.61 | 2258.58 | 0.336495 |
| `voc_v2` | 0.7700 | 1089.62 | 2308.59 | 0.333537 |
| `mv_6` | 0.5100 | 0.00 | 2444.68 | 0.208616 |
| `sc_6` | 0.5100 | 0.00 | 2444.68 | 0.208616 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6900 | 0.00 | 5266.92 | 0.131006 |
| `always` | 0.6800 | 1445.98 | 6712.90 | 0.101298 |
| `disagreement` | 0.6800 | 338.80 | 5605.72 | 0.121305 |
| `confidence` | 0.6900 | 0.00 | 5266.92 | 0.131006 |
| `hybrid` | 0.6800 | 338.80 | 5605.72 | 0.121305 |
| `voc_v2` | 0.6800 | 467.21 | 5734.13 | 0.118588 |
| `mv_6` | 0.7200 | 0.00 | 10539.41 | 0.068315 |
| `sc_6` | 0.7200 | 0.00 | 10539.41 | 0.068315 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7100 | 0.00 | 964.22 | 0.736346 |
| `always` | 0.7200 | 1300.83 | 2265.05 | 0.317874 |
| `disagreement` | 0.7100 | 272.61 | 1236.83 | 0.574048 |
| `confidence` | 0.7100 | 40.08 | 1004.30 | 0.706960 |
| `hybrid` | 0.7100 | 313.17 | 1277.39 | 0.555821 |
| `voc_v2` | 0.7100 | 787.65 | 1751.87 | 0.405281 |
| `mv_6` | 0.7000 | 0.00 | 1931.92 | 0.362334 |
| `sc_6` | 0.7000 | 0.00 | 1931.92 | 0.362334 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00002`
- 问题预览：`Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased...`
- 金标：`70000`
- mv_3：`90000 / 0.0`
- always_communicate：`70000 / 1.0`
- 解释：`always_communicate 能纠错，但 hybrid_trigger 在该题 early exit，漏掉了有益通信。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00015`
- 问题预览：`A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets wo...`
- 金标：`125`
- mv_3：`125 / 1.0`
- always_communicate：`125 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00021`
- 问题预览：`Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Sa...`
- 金标：`14`
- mv_3：`14 / 1.0`
- always_communicate：`14 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00035`
- 问题预览：`Mike plays ping pong for 40 minutes. In the first 20 minutes, he scores 4 points. In the second 20 minutes, he scores...`
- 金标：`9`
- mv_3：`9 / 1.0`
- always_communicate：`9 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00057`
- 问题预览：`A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 po...`
- 金标：`83`
- mv_3：`82 / 0.0`
- always_communicate：`82 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/voc_trigger_main/pilot100/20260509T125937Z-voc_trigger_main-pilot100-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
