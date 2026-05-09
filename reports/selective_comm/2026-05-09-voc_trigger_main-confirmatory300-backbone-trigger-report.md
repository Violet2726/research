# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.7346。
- 触发策略中表现最佳的是 `always`。
- 共享前缀机制的平均 token 节省比例约为 0.7371。
- 当前推荐的下一轮默认策略为 `voc_trigger_v2`。

## 实验概览

- 实验名：`voc_trigger_main`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`869983.00`，独立重跑 token=`3233141.00`，节省比例=`0.7309`。
- `hotpotqa`：共享执行实际 token=`2037592.00`，独立重跑 token=`8779717.00`，节省比例=`0.7679`。
- `overall`：共享执行实际 token=`3426749.00`，独立重跑 token=`13738015.00`，节省比例=`0.7506`。
- `strategyqa`：共享执行实际 token=`519174.00`，独立重跑 token=`1725157.00`，节省比例=`0.6991`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6393 | 0.00 | 2636.88 | 0.242454 |
| `always` | 0.7346 | 1496.71 | 4133.59 | 0.177719 |
| `disagreement` | 0.7250 | 556.56 | 3193.44 | 0.227018 |
| `confidence` | 0.6393 | 9.63 | 2646.52 | 0.241572 |
| `hybrid` | 0.7262 | 568.32 | 3205.21 | 0.226561 |
| `voc_v2` | 0.7274 | 756.15 | 3393.03 | 0.214375 |
| `mv_6` | 0.6285 | 0.00 | 5272.69 | 0.119193 |
| `sc_6` | 0.6285 | 0.00 | 5272.69 | 0.119193 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.1098 | 1.0000 | 0.8902 | 0.0000 |
| `disagreement` | 0.3438 | 0.6562 | 0.2877 | 0.9011 | 0.2449 | 0.0989 |
| `confidence` | 0.0072 | 0.9928 | 0.0000 | 0.0000 | 0.0072 | 1.0000 |
| `hybrid` | 0.3522 | 0.6478 | 0.2842 | 0.9121 | 0.2521 | 0.0879 |
| `voc_v2` | 0.4885 | 0.5115 | 0.2074 | 0.9231 | 0.3872 | 0.0769 |

## VoC 诊断

- 推荐默认策略：`voc_trigger_v2`。
- 相对 `always_communicate` 的准确率下降：`0.007238`；总 token 降低比例：`0.179157`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 1.0000 | 0.8758 | 1.0000 | 1496.71 |
| `disagreement` | 0.9011 | 0.9167 | 0.6737 | 0.3438 | 556.56 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0072 | 9.63 |
| `hybrid` | 0.9121 | 0.9167 | 0.6781 | 0.3522 | 568.32 |
| `voc_v2` | 0.9231 | 0.9167 | 0.7654 | 0.4885 | 756.15 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5233 | 0.00 | 1220.51 | 0.428783 |
| `always` | 0.7933 | 1679.44 | 2899.94 | 0.273568 |
| `disagreement` | 0.7667 | 972.66 | 2193.17 | 0.349570 |
| `confidence` | 0.5233 | 0.00 | 1220.51 | 0.428783 |
| `hybrid` | 0.7700 | 978.18 | 2198.68 | 0.350210 |
| `voc_v2` | 0.7733 | 1044.33 | 2264.83 | 0.341452 |
| `mv_6` | 0.4900 | 0.00 | 2433.56 | 0.201351 |
| `sc_6` | 0.4900 | 0.00 | 2433.56 | 0.201351 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6567 | 0.00 | 5329.15 | 0.123222 |
| `always` | 0.6533 | 1462.82 | 6791.97 | 0.096192 |
| `disagreement` | 0.6567 | 340.26 | 5669.42 | 0.115826 |
| `confidence` | 0.6567 | 0.00 | 5329.15 | 0.123222 |
| `hybrid` | 0.6567 | 344.94 | 5674.09 | 0.115731 |
| `voc_v2` | 0.6567 | 471.94 | 5801.09 | 0.113197 |
| `mv_6` | 0.6800 | 0.00 | 10659.55 | 0.063793 |
| `sc_6` | 0.6800 | 0.00 | 10659.55 | 0.063793 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7686 | 0.00 | 965.41 | 0.796099 |
| `always` | 0.7642 | 1301.73 | 2267.14 | 0.337074 |
| `disagreement` | 0.7598 | 294.80 | 1260.21 | 0.602935 |
| `confidence` | 0.7686 | 34.88 | 1000.28 | 0.768341 |
| `hybrid` | 0.7598 | 324.05 | 1289.45 | 0.589261 |
| `voc_v2` | 0.7598 | 750.95 | 1716.35 | 0.442697 |
| `mv_6` | 0.7424 | 0.00 | 1935.03 | 0.383641 |
| `sc_6` | 0.7424 | 0.00 | 1935.03 | 0.383641 |

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
- 样本 ID：`gsm8k-00011`
- 问题预览：`Toula went to the bakery and bought various types of pastries. She bought 3 dozen donuts which cost $68 per dozen, 2 ...`
- 金标：`694`
- mv_3：`732 / 0.0`
- always_communicate：`732 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00015`
- 问题预览：`A merchant wants to make a choice of purchase between 2 purchase plans: jewelry worth $5,000 or electronic gadgets wo...`
- 金标：`125`
- mv_3：`125 / 1.0`
- always_communicate：`125 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00021`
- 问题预览：`Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Sa...`
- 金标：`14`
- mv_3：`14 / 1.0`
- always_communicate：`14 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00034`
- 问题预览：`Siobhan has 2 fewer jewels than Aaron. Aaron has 5 more jewels than half of Raymond's jewels. If Raymond has 40 jewel...`
- 金标：`23`
- mv_3：`23 / 1.0`
- always_communicate：`23 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/voc_trigger_main/confirmatory300/20260509T130107Z-voc_trigger_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
