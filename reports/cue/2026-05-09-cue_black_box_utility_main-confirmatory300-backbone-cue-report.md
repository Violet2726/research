# CUE 科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.6284。
- 在通信策略中，`always` 的总体表现最佳。
- 当前推荐的下一轮默认策略为 `disagreement_triggered`。

## 实验概览

- 实验名：`cue_black_box_utility_main`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与方法结构

- CUE 的核心问题是：在黑盒条件下，能否先估计通信效用，再把通信机会定向分配给真正需要协同的样本。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；策略诊断侧重触发率、误触发率与漏掉有益通信率。
- 报告中特别关注 `cue_v1` 相对于 `always_communicate` 是否能够在显著降低成本时保持可接受的性能损失。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.5248 | 0.00 | 2226.98 | 3.00 | 0.235675 |
| `always` | 0.6284 | 3349.45 | 5576.42 | 6.28 | 0.112691 |
| `disagreement` | 0.6130 | 1360.24 | 3587.22 | 4.51 | 0.170889 |
| `consensus_freeze` | 0.6277 | 3140.62 | 5367.60 | 6.16 | 0.116945 |
| `cue_v1` | 0.5724 | 754.82 | 2981.79 | 3.85 | 0.191974 |

## 触发与 Oracle 诊断

- 推荐策略：`disagreement_triggered`。
- 相对 `always_communicate` 的准确率下降：`0.055983`；总 token 降低比例：`0.465286`。

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.1155 | 1.0000 | 0.8845 | 0.0000 |
| `disagreement` | 0.4206 | 0.5794 | 0.2379 | 0.8667 | 0.3205 | 0.1333 |
| `consensus_freeze` | 0.9629 | 0.0371 | 0.1192 | 0.9939 | 0.8481 | 0.0061 |
| `cue_v1` | 0.2358 | 0.7642 | 0.2226 | 0.4545 | 0.1833 | 0.5455 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.4767 | 0.00 | 1350.38 | 0.352988 |
| `always` | 0.7433 | 2448.49 | 3798.87 | 0.195672 |
| `disagreement` | 0.7067 | 1467.67 | 2818.04 | 0.250765 |
| `consensus_freeze` | 0.7400 | 2424.30 | 3774.68 | 0.196043 |
| `cue_v1` | 0.6100 | 857.21 | 2207.59 | 0.276319 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6433 | 0.00 | 5361.20 | 0.119998 |
| `always` | 0.6500 | 6664.68 | 12025.88 | 0.054050 |
| `disagreement` | 0.6500 | 1523.03 | 6884.22 | 0.094419 |
| `consensus_freeze` | 0.6500 | 5724.42 | 11085.62 | 0.058635 |
| `cue_v1` | 0.6433 | 797.86 | 6159.06 | 0.104453 |

### math500

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.3033 | 0.00 | 1361.60 | 0.222777 |
| `always` | 0.4200 | 2594.45 | 3956.04 | 0.106167 |
| `disagreement` | 0.4000 | 1851.97 | 3213.57 | 0.124472 |
| `consensus_freeze` | 0.4200 | 2581.68 | 3943.28 | 0.106510 |
| `cue_v1` | 0.3400 | 1023.11 | 2384.71 | 0.142575 |

### mmlu_pro

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5200 | 0.00 | 1720.92 | 0.302164 |
| `always` | 0.6133 | 2759.85 | 4480.77 | 0.136881 |
| `disagreement` | 0.6000 | 1294.62 | 3015.54 | 0.198969 |
| `consensus_freeze` | 0.6133 | 2742.36 | 4463.28 | 0.137417 |
| `cue_v1` | 0.5600 | 716.33 | 2437.25 | 0.229767 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7293 | 0.00 | 1066.03 | 0.684085 |
| `always` | 0.7424 | 1948.12 | 3014.15 | 0.246291 |
| `disagreement` | 0.7380 | 448.03 | 1514.06 | 0.487425 |
| `consensus_freeze` | 0.7424 | 1948.12 | 3014.15 | 0.246291 |
| `cue_v1` | 0.7511 | 263.22 | 1329.26 | 0.565046 |

## 结论与建议

- 若 CUE 的 Oracle 精确率和召回率同时偏低，则应优先改进 utility 估计，而不是直接增加通信预算。
- 若准确率接近 `always_communicate`，但总 token 明显下降，则说明 CUE 已具备作为默认策略的工程价值。
- 进入更大样本 phase 前，建议优先复核误触发率与漏掉有益通信率是否同时可控，避免只依赖总体准确率决策。

## 局限性

- 当前 Oracle 是基于现有控制组构造的近似标签，因此更适合做策略筛选，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒实现效果，不能直接等同于可访问内部不确定性信号时的理想上界。

## 复现与产物说明

- 运行目录：`runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`oracle_trigger_eval.json`。
- 本地报告与发布报告共享同一套 run 内图资产，便于后续引用与审稿期复核。

## 图表资产

### CUE 成本-性能前沿

![CUE 成本-性能前沿](../../runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### CUE 效率排序

![CUE 效率排序](../../runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### CUE 跨数据集表现

![CUE 跨数据集表现](../../runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 策略触发率权衡

![策略触发率权衡](../../runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/policy_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/cue/cue_black_box_utility_main/confirmatory300/20260509T130058Z-cue_black_box_utility_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
