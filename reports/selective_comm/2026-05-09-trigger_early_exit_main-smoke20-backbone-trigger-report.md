# 选择性通信科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.8667。
- 触发策略中表现最佳的是 `always`。
- 共享前缀机制的平均 token 节省比例约为 0.6008。
- 当前推荐的下一轮默认策略为 `hybrid_trigger`。

## 实验概览

- 实验名：`trigger_early_exit_main`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。
- 主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。
- 所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。

## 共享前缀节省情况

- `gsm8k`：共享执行实际 token=`56017.00`，独立重跑 token=`122079.00`，节省比例=`0.5411`。
- `hotpotqa`：共享执行实际 token=`224188.00`，独立重跑 token=`633920.00`，节省比例=`0.6463`。
- `overall`：共享执行实际 token=`325437.00`，独立重跑 token=`866641.00`，节省比例=`0.6245`。
- `strategyqa`：共享执行实际 token=`45232.00`，独立重跑 token=`110642.00`，节省比例=`0.5912`。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8500 | 0.00 | 2369.27 | 0.358761 |
| `always` | 0.8667 | 3054.68 | 5423.95 | 0.159785 |
| `disagreement` | 0.8667 | 877.00 | 3246.27 | 0.266973 |
| `confidence` | 0.8500 | 133.48 | 2502.75 | 0.339626 |
| `hybrid` | 0.8667 | 901.78 | 3271.05 | 0.264951 |
| `mv_6` | 0.8667 | 0.00 | 4732.97 | 0.183113 |
| `sc_6` | 0.8667 | 0.00 | 4732.97 | 0.183113 |

## Trigger 诊断

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.0167 | 1.0000 | 0.9833 | 0.0000 |
| `disagreement` | 0.2000 | 0.8000 | 0.0833 | 1.0000 | 0.1833 | 0.0000 |
| `confidence` | 0.0333 | 0.9667 | 0.0000 | 0.0000 | 0.0333 | 1.0000 |
| `hybrid` | 0.2167 | 0.7833 | 0.0769 | 1.0000 | 0.2000 | 0.0000 |

## VoC 诊断

- 推荐默认策略：`hybrid_trigger`。
- 相对 `always_communicate` 的准确率下降：`0.0`；总 token 降低比例：`0.396925`。

| 策略 | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | 触发率 | 平均通信 token / 题 |
| --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.9833 | 1.0000 | 3054.68 |
| `disagreement` | 1.0000 | 0.0000 | 0.9167 | 0.2000 | 877.00 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0333 | 133.48 |
| `hybrid` | 1.0000 | 0.0000 | 0.9231 | 0.2167 | 901.78 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 1.0000 | 0.00 | 1033.80 | 0.967305 |
| `always` | 1.0000 | 1767.05 | 2800.85 | 0.357034 |
| `disagreement` | 1.0000 | 100.85 | 1134.65 | 0.881329 |
| `confidence` | 1.0000 | 0.00 | 1033.80 | 0.967305 |
| `hybrid` | 1.0000 | 100.85 | 1134.65 | 0.881329 |
| `mv_6` | 1.0000 | 0.00 | 2039.15 | 0.490400 |
| `sc_6` | 1.0000 | 0.00 | 2039.15 | 0.490400 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7500 | 0.00 | 5235.10 | 0.143264 |
| `always` | 0.7500 | 5974.30 | 11209.40 | 0.066908 |
| `disagreement` | 0.7500 | 2227.60 | 7462.70 | 0.100500 |
| `confidence` | 0.7500 | 326.10 | 5561.20 | 0.134863 |
| `hybrid` | 0.7500 | 2227.60 | 7462.70 | 0.100500 |
| `mv_6` | 0.8500 | 0.00 | 10468.15 | 0.081199 |
| `sc_6` | 0.8500 | 0.00 | 10468.15 | 0.081199 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.8000 | 0.00 | 838.90 | 0.953630 |
| `always` | 0.8500 | 1422.70 | 2261.60 | 0.375840 |
| `disagreement` | 0.8500 | 302.55 | 1141.45 | 0.744667 |
| `confidence` | 0.8000 | 74.35 | 913.25 | 0.875992 |
| `hybrid` | 0.8500 | 376.90 | 1215.80 | 0.699128 |
| `mv_6` | 0.7500 | 0.00 | 1691.60 | 0.443367 |
| `sc_6` | 0.7500 | 0.00 | 1691.60 | 0.443367 |

## 典型案例

### 案例 1

- 数据集：`gsm8k`
- 样本 ID：`gsm8k-00672`
- 问题预览：`Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...`
- 金标：`2`
- mv_3：`2 / 1.0`
- always_communicate：`2 / 1.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 2

- 数据集：`hotpotqa`
- 样本 ID：`5a73977d554299623ed4ac08`
- 问题预览：`What is the shared country of ancestry between Art Laboe and Scout Tufankjian?`
- 金标：`Armenian`
- mv_3：`armenia / 0.0`
- always_communicate：`armenia / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 3

- 数据集：`hotpotqa`
- 样本 ID：`5a7a567255429941d65f25bd`
- 问题预览：`What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okham...`
- 金标：`flotilla`
- mv_3：`part of flotilla / 0.0`
- always_communicate：`part of flotilla / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 4

- 数据集：`hotpotqa`
- 样本 ID：`5a80cf4c55429938b61421f6`
- 问题预览：`What was the concept of the business Eric S .Pistorius worked for after being an attorney?`
- 金标：`to ensure wide visibility and understanding of cases in a region`
- mv_3：`his law firm / 0.0`
- always_communicate：`his law firm / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`

### 案例 5

- 数据集：`hotpotqa`
- 样本 ID：`5ab514c05542991779162d72`
- 问题预览：`The school in which the Wilmslow Show is held is designated as what?`
- 金标：`Centre of Excellence`
- mv_3：`designated centre of excellence / 0.0`
- always_communicate：`designated centre of excellence / 0.0`
- 解释：`通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。`


## 结论与建议

- 若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。
- 若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。
- 默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。

## 局限性

- 当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。

## 复现与产物说明

- 运行目录：`runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。

## 图表资产

### 选择性通信成本-性能前沿

![选择性通信成本-性能前沿](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### 选择性通信效率排序

![选择性通信效率排序](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 选择性通信跨数据集表现

![选择性通信跨数据集表现](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 触发率权衡

![触发率权衡](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/trigger_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### 共享前缀节省比

![共享前缀节省比](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/shared_prefix_savings.svg)

*共享前缀执行相对于独立重跑的 token 节省比例。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/selective_comm/trigger_early_exit_main/smoke20/20260509T125843Z-trigger_early_exit_main-smoke20-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
