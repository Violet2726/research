# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `mv_6`，准确率为 0.8567。
- 触发策略中表现最佳的是 `always`。
- 共享前缀机制的平均 token 节省比例约为 0.5960。
- 当前推荐的下一轮默认策略为 `hybrid_trigger`。

## 实验概览

- 实验名：`trigger_early_exit_main`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`296995.00`，独立重跑 token=`645988.00`，节省比例=`0.5402`。
- `hotpotqa`：共享执行实际 token=`1105628.00`，独立重跑 token=`3017206.00`，节省比例=`0.6336`。
- `overall`：共享执行实际 token=`1629031.00`，独立重跑 token=`4223452.00`，节省比例=`0.6143`。
- `strategyqa`：共享执行实际 token=`226408.00`，独立重跑 token=`560258.00`，节省比例=`0.5959`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8400 | 0.00 | 2363.06 | 0.355471 |
| `always` | 0.8467 | 3067.04 | 5430.10 | 0.155921 |
| `disagreement` | 0.8467 | 703.57 | 3066.64 | 0.276090 |
| `confidence` | 0.8433 | 98.77 | 2461.83 | 0.342563 |
| `hybrid` | 0.8467 | 756.54 | 3119.60 | 0.271402 |
| `mv_6` | 0.8567 | 0.00 | 4728.56 | 0.181169 |
| `sc_6` | 0.8567 | 0.00 | 4728.56 | 0.181169 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.0133 | 1.0000 | 0.9867 | 0.0000 |
| `disagreement` | 0.1800 | 0.8200 | 0.0741 | 1.0000 | 0.1667 | 0.0000 |
| `confidence` | 0.0333 | 0.9667 | 0.1000 | 0.2500 | 0.0300 | 0.7500 |
| `hybrid` | 0.2033 | 0.7967 | 0.0656 | 1.0000 | 0.1900 | 0.0000 |

## VoC 诊断

- 推荐默认策略：`hybrid_trigger`。
- 相对 `always_communicate` 的准确率下降：`0.0`；总 token 降低比例：`0.425498`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 1.0000 | 0.9800 | 1.0000 | 3067.04 |
| `disagreement` | 1.0000 | 1.0000 | 0.8889 | 0.1800 | 703.57 |
| `confidence` | 0.2500 | 0.0000 | 0.9000 | 0.0333 | 98.77 |
| `hybrid` | 1.0000 | 1.0000 | 0.9016 | 0.2033 | 756.54 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.9700 | 0.00 | 1076.85 | 0.900775 |
| `always` | 0.9700 | 1893.10 | 2969.95 | 0.326605 |
| `disagreement` | 0.9700 | 129.69 | 1206.54 | 0.803952 |
| `confidence` | 0.9700 | 0.00 | 1076.85 | 0.900775 |
| `hybrid` | 0.9700 | 129.69 | 1206.54 | 0.803952 |
| `mv_6` | 0.9700 | 0.00 | 2154.72 | 0.450175 |
| `sc_6` | 0.9700 | 0.00 | 2154.72 | 0.450175 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7500 | 0.00 | 5176.54 | 0.144884 |
| `always` | 0.7400 | 5879.74 | 11056.28 | 0.066930 |
| `disagreement` | 0.7400 | 1659.48 | 6836.02 | 0.108250 |
| `confidence` | 0.7500 | 192.87 | 5369.41 | 0.139680 |
| `hybrid` | 0.7400 | 1733.81 | 6910.35 | 0.107086 |
| `mv_6` | 0.7700 | 0.00 | 10362.42 | 0.074307 |
| `sc_6` | 0.7700 | 0.00 | 10362.42 | 0.074307 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8000 | 0.00 | 835.80 | 0.957167 |
| `always` | 0.8300 | 1428.28 | 2264.08 | 0.366595 |
| `disagreement` | 0.8300 | 321.55 | 1157.35 | 0.717156 |
| `confidence` | 0.8100 | 103.43 | 939.23 | 0.862409 |
| `hybrid` | 0.8300 | 406.12 | 1241.92 | 0.668320 |
| `mv_6` | 0.8300 | 0.00 | 1668.55 | 0.497438 |
| `sc_6` | 0.8300 | 0.00 | 1668.55 | 0.497438 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00066`
- 问题预览：`There are four schools competing at a basketball tournament. Each school has sent a girls’ basketball team and a boys...`
- 金标：`48`
- mv_3：`48 / 1.0`
- always_communicate：`48 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 2

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00187`
- 问题预览：`Mandy owes Benedict $100. They agreed to have monthly interest of 2%. If Mandy was able to pay it after 3 months, how...`
- 金标：`106`
- mv_3：`106.12 / 0.0`
- always_communicate：`106.12 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00371`
- 问题预览：`A shoe store was having a weekend sale on a brand of popular tennis shoes. On Friday the store sold 14 pairs of tenni...`
- 金标：`50`
- mv_3：`100 / 0.0`
- always_communicate：`100 / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00672`
- 问题预览：`Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...`
- 金标：`2`
- mv_3：`2 / 1.0`
- always_communicate：`2 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00969`
- 问题预览：`John decides to buy new phones for him, his 2 kids, and his wife. Each phone after the first 2 is half price. If the ...`
- 金标：`1800`
- mv_3：`1800 / 1.0`
- always_communicate：`1800 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/trigger_early_exit_main/pilot100/20260509T125913Z-trigger_early_exit_main-pilot100-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
