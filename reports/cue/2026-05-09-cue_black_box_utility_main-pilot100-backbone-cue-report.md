# CUE 科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.6360。
- 在通信策略中，`always` 的总体表现最佳。
- 当前推荐的下一轮默认策略为 `disagreement_triggered`。

## 实验概览

- 实验名：`cue_black_box_utility_main`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与方法结构

- CUE 的核心问题是：在黑盒条件下，能否先估计通信效用，再把通信机会定向分配给真正需要协同的样本。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；策略诊断侧重触发率、误触发率与漏掉有益通信率。
- 报告中特别关注 `cue_v1` 相对于 `always_communicate` 是否能够在显著降低成本时保持可接受的性能损失。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.5480 | 0.00 | 2135.59 | 3.00 | 0.256603 |
| `always` | 0.6360 | 3220.89 | 5356.49 | 6.26 | 0.118735 |
| `disagreement` | 0.6240 | 1237.01 | 3372.60 | 4.45 | 0.185020 |
| `consensus_freeze` | 0.6340 | 3090.62 | 5226.21 | 6.18 | 0.121312 |
| `cue_v1` | 0.5780 | 663.37 | 2798.96 | 3.79 | 0.206505 |

## 触发与 Oracle 诊断

- 推荐策略：`disagreement_triggered`。
- 相对 `always_communicate` 的准确率下降：`0.058`；总 token 降低比例：`0.477463`。

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.1040 | 1.0000 | 0.8960 | 0.0000 |
| `disagreement` | 0.4060 | 0.5940 | 0.2266 | 0.8846 | 0.3140 | 0.1154 |
| `consensus_freeze` | 0.9760 | 0.0240 | 0.1045 | 0.9808 | 0.8740 | 0.0192 |
| `cue_v1` | 0.2220 | 0.7780 | 0.1802 | 0.3846 | 0.1820 | 0.6154 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.4900 | 0.00 | 1346.64 | 0.363869 |
| `always` | 0.7200 | 2398.96 | 3745.60 | 0.192226 |
| `disagreement` | 0.6900 | 1356.09 | 2702.73 | 0.255297 |
| `consensus_freeze` | 0.7100 | 2346.00 | 3692.64 | 0.192274 |
| `cue_v1` | 0.6000 | 699.10 | 2045.74 | 0.293292 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6600 | 0.00 | 5295.33 | 0.124638 |
| `always` | 0.6800 | 6547.12 | 11842.45 | 0.057421 |
| `disagreement` | 0.6800 | 1300.81 | 6596.14 | 0.103091 |
| `consensus_freeze` | 0.6800 | 5948.71 | 11244.04 | 0.060476 |
| `cue_v1` | 0.6600 | 626.37 | 5921.70 | 0.111454 |

### math500

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.3000 | 0.00 | 1361.67 | 0.220318 |
| `always` | 0.4300 | 2584.22 | 3945.89 | 0.108974 |
| `disagreement` | 0.4000 | 1662.89 | 3024.56 | 0.132251 |
| `consensus_freeze` | 0.4300 | 2584.22 | 3945.89 | 0.108974 |
| `cue_v1` | 0.3300 | 964.35 | 2326.02 | 0.141873 |

### mmlu_pro

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6200 | 0.00 | 1612.78 | 0.384429 |
| `always` | 0.6600 | 2641.26 | 4254.04 | 0.155147 |
| `disagreement` | 0.6600 | 1303.07 | 2915.85 | 0.226349 |
| `consensus_freeze` | 0.6600 | 2641.26 | 4254.04 | 0.155147 |
| `cue_v1` | 0.6100 | 690.66 | 2303.44 | 0.264821 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6700 | 0.00 | 1061.54 | 0.631159 |
| `always` | 0.6900 | 1932.91 | 2994.45 | 0.230426 |
| `disagreement` | 0.6900 | 562.20 | 1623.74 | 0.424945 |
| `consensus_freeze` | 0.6900 | 1932.91 | 2994.45 | 0.230426 |
| `cue_v1` | 0.6900 | 336.37 | 1397.91 | 0.493594 |

## 结论与建议

- 若 CUE 的 Oracle 精确率和召回率同时偏低，则应优先改进 utility 估计，而不是直接增加通信预算。
- 若准确率接近 `always_communicate`，但总 token 明显下降，则说明 CUE 已具备作为默认策略的工程价值。
- 进入更大样本 phase 前，建议优先复核误触发率与漏掉有益通信率是否同时可控，避免只依赖总体准确率决策。

## 局限性

- 当前 Oracle 是基于现有控制组构造的近似标签，因此更适合做策略筛选，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒实现效果，不能直接等同于可访问内部不确定性信号时的理想上界。

## 复现与产物说明

- 运行目录：`runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`oracle_trigger_eval.json`。
- 本地报告与发布报告共享同一套 run 内图资产，便于后续引用与审稿期复核。

## 图表资产

### CUE 成本-性能前沿

![CUE 成本-性能前沿](../../runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### CUE 效率排序

![CUE 效率排序](../../runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### CUE 跨数据集表现

![CUE 跨数据集表现](../../runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 策略触发率权衡

![策略触发率权衡](../../runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5/figures/policy_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/cue/cue_black_box_utility_main/pilot100/20260509T125932Z-cue_black_box_utility_main-pilot100-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
