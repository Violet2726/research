# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.7667。
- 触发策略中表现最佳的是 `always`。
- 共享前缀机制的平均 token 节省比例约为 0.7358。
- 当前推荐的下一轮默认策略为 `voc_trigger_v2`。

## 实验概览

- 实验名：`voc_trigger_main`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`55917.00`，独立重跑 token=`206643.00`，节省比例=`0.7294`。
- `hotpotqa`：共享执行实际 token=`135986.00`，独立重跑 token=`581154.00`，节省比例=`0.7660`。
- `overall`：共享执行实际 token=`237323.00`，独立重跑 token=`939378.00`，节省比例=`0.7474`。
- `strategyqa`：共享执行实际 token=`45420.00`，独立重跑 token=`151581.00`，节省比例=`0.7004`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7167 | 0.00 | 2479.05 | 0.289089 |
| `always` | 0.7667 | 1476.33 | 3955.38 | 0.193829 |
| `disagreement` | 0.7667 | 513.40 | 2992.45 | 0.256200 |
| `confidence` | 0.7167 | 0.00 | 2479.05 | 0.289089 |
| `hybrid` | 0.7667 | 513.40 | 2992.45 | 0.256200 |
| `voc_v2` | 0.7667 | 757.92 | 3236.97 | 0.236847 |
| `mv_6` | 0.6833 | 0.00 | 4980.77 | 0.137194 |
| `sc_6` | 0.6833 | 0.00 | 4980.77 | 0.137194 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.0667 | 1.0000 | 0.9333 | 0.0000 |
| `disagreement` | 0.3167 | 0.6833 | 0.2105 | 1.0000 | 0.2500 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 |
| `hybrid` | 0.3167 | 0.6833 | 0.2105 | 1.0000 | 0.2500 | 0.0000 |
| `voc_v2` | 0.5000 | 0.5000 | 0.1333 | 1.0000 | 0.4333 | 0.0000 |

## VoC 诊断

- 推荐默认策略：`voc_trigger_v2`。
- 相对 `always_communicate` 的准确率下降：`0.0`；总 token 降低比例：`0.18163`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 1.0000 | 0.9167 | 1.0000 | 1476.33 |
| `disagreement` | 1.0000 | 1.0000 | 0.7368 | 0.3167 | 513.40 |
| `confidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 |
| `hybrid` | 1.0000 | 1.0000 | 0.7368 | 0.3167 | 513.40 |
| `voc_v2` | 1.0000 | 1.0000 | 0.8333 | 0.5000 | 757.92 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6000 | 0.00 | 1152.90 | 0.520427 |
| `always` | 0.7000 | 1642.95 | 2795.85 | 0.250371 |
| `disagreement` | 0.7000 | 974.90 | 2127.80 | 0.328978 |
| `confidence` | 0.6000 | 0.00 | 1152.90 | 0.520427 |
| `hybrid` | 0.7000 | 974.90 | 2127.80 | 0.328978 |
| `voc_v2` | 0.7000 | 974.90 | 2127.80 | 0.328978 |
| `mv_6` | 0.5500 | 0.00 | 2374.70 | 0.231608 |
| `sc_6` | 0.5500 | 0.00 | 2374.70 | 0.231608 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7500 | 0.00 | 5318.50 | 0.141017 |
| `always` | 0.7500 | 1480.80 | 6799.30 | 0.110305 |
| `disagreement` | 0.7500 | 304.65 | 5623.15 | 0.133377 |
| `confidence` | 0.7500 | 0.00 | 5318.50 | 0.141017 |
| `hybrid` | 0.7500 | 304.65 | 5623.15 | 0.133377 |
| `voc_v2` | 0.7500 | 375.10 | 5693.60 | 0.131727 |
| `mv_6` | 0.8000 | 0.00 | 10636.15 | 0.075215 |
| `sc_6` | 0.8000 | 0.00 | 10636.15 | 0.075215 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8000 | 0.00 | 965.75 | 0.828372 |
| `always` | 0.8500 | 1305.25 | 2271.00 | 0.374284 |
| `disagreement` | 0.8500 | 260.65 | 1226.40 | 0.693085 |
| `confidence` | 0.8000 | 0.00 | 965.75 | 0.828372 |
| `hybrid` | 0.8500 | 260.65 | 1226.40 | 0.693085 |
| `voc_v2` | 0.8500 | 923.75 | 1889.50 | 0.449854 |
| `mv_6` | 0.7000 | 0.00 | 1931.45 | 0.362422 |
| `sc_6` | 0.7000 | 0.00 | 1931.45 | 0.362422 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00057`
- 问题预览：`A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 po...`
- 金标：`83`
- mv_3：`82 / 0.0`
- always_communicate：`82 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00222`
- 问题预览：`Andy plants 90 geraniums and 40 fewer petunias that geraniums. How many flowers does he plant total?`
- 金标：`140`
- mv_3：`140 / 1.0`
- always_communicate：`140 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00599`
- 问题预览：`A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?`
- 金标：`48`
- mv_3：`44 / 0.0`
- always_communicate：`44 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00618`
- 问题预览：`The school auditorium has 4 rows of seats. There are 18 seats in each row. One-fourth of the seats were occupied by t...`
- 金标：`36`
- mv_3：`45 / 0.0`
- always_communicate：`45 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00672`
- 问题预览：`Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...`
- 金标：`2`
- mv_3：`5 / 0.0`
- always_communicate：`5 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/voc_trigger_main/smoke20/20260509T125855Z-voc_trigger_main-smoke20-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
