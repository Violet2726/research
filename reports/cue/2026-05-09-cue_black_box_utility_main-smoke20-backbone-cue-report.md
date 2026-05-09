# CUE 科研报告

## 摘要

- 总体准确率最高的方法是 `always`，准确率为 0.7000。
- 在通信策略中，`always` 的总体表现最佳。
- 当前推荐的下一轮默认策略为 `disagreement_triggered`。

## 实验概览

- 实验名：`cue_black_box_utility_main`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与方法结构

- CUE 的核心问题是：在黑盒条件下，能否先估计通信效用，再把通信机会定向分配给真正需要协同的样本。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；策略诊断侧重触发率、误触发率与漏掉有益通信率。
- 报告中特别关注 `cue_v1` 相对于 `always_communicate` 是否能够在显著降低成本时保持可接受的性能损失。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.5700 | 0.00 | 2136.82 | 3.00 | 0.266752 |
| `always` | 0.7000 | 3260.95 | 5397.77 | 6.27 | 0.129683 |
| `disagreement` | 0.7000 | 1436.78 | 3573.60 | 4.43 | 0.195881 |
| `consensus_freeze` | 0.7000 | 3099.74 | 5236.56 | 6.18 | 0.133676 |
| `cue_v1` | 0.6200 | 626.83 | 2763.65 | 3.64 | 0.224341 |

## 触发与 Oracle 诊断

- 推荐策略：`disagreement_triggered`。
- 相对 `always_communicate` 的准确率下降：`0.08`；总 token 降低比例：`0.488002`。

| 策略 | 触发率 | 早退率 | Oracle 精确率 | Oracle 召回率 | 误触发率 | 漏掉有益通信率 |
| --- | --- | --- | --- | --- | --- | --- |
| `always` | 1.0000 | 0.0000 | 0.1400 | 1.0000 | 0.8600 | 0.0000 |
| `disagreement` | 0.4000 | 0.6000 | 0.3500 | 1.0000 | 0.2600 | 0.0000 |
| `consensus_freeze` | 0.9700 | 0.0300 | 0.1443 | 1.0000 | 0.8300 | 0.0000 |
| `cue_v1` | 0.1800 | 0.8200 | 0.3333 | 0.4286 | 0.1200 | 0.5714 |

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5500 | 0.00 | 1294.00 | 0.425039 |
| `always` | 0.8000 | 2353.20 | 3647.20 | 0.219346 |
| `disagreement` | 0.8000 | 1303.10 | 2597.10 | 0.308036 |
| `consensus_freeze` | 0.8000 | 2353.20 | 3647.20 | 0.219346 |
| `cue_v1` | 0.7500 | 647.95 | 1941.95 | 0.386210 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 5341.35 | 0.131053 |
| `always` | 0.7500 | 6956.10 | 12297.45 | 0.060988 |
| `disagreement` | 0.7500 | 2931.85 | 8273.20 | 0.090654 |
| `consensus_freeze` | 0.7500 | 6150.05 | 11491.40 | 0.065266 |
| `cue_v1` | 0.7000 | 1201.15 | 6542.50 | 0.106993 |

### math500

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.3500 | 0.00 | 1335.60 | 0.262055 |
| `always` | 0.5500 | 2472.35 | 3807.95 | 0.144435 |
| `disagreement` | 0.5500 | 1440.70 | 2776.30 | 0.198105 |
| `consensus_freeze` | 0.5500 | 2472.35 | 3807.95 | 0.144435 |
| `cue_v1` | 0.4000 | 532.75 | 1868.35 | 0.214093 |

### mmlu_pro

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5500 | 0.00 | 1649.30 | 0.333475 |
| `always` | 0.7000 | 2631.80 | 4281.10 | 0.163509 |
| `disagreement` | 0.7000 | 1239.30 | 2888.60 | 0.242332 |
| `consensus_freeze` | 0.7000 | 2631.80 | 4281.10 | 0.163509 |
| `cue_v1` | 0.5500 | 568.70 | 2218.00 | 0.247971 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 1063.85 | 0.657987 |
| `always` | 0.7000 | 1891.30 | 2955.15 | 0.236875 |
| `disagreement` | 0.7000 | 268.95 | 1332.80 | 0.525210 |
| `consensus_freeze` | 0.7000 | 1891.30 | 2955.15 | 0.236875 |
| `cue_v1` | 0.7000 | 183.60 | 1247.45 | 0.561145 |

## 结论与建议

- 若 CUE 的 Oracle 精确率和召回率同时偏低，则应优先改进 utility 估计，而不是直接增加通信预算。
- 若准确率接近 `always_communicate`，但总 token 明显下降，则说明 CUE 已具备作为默认策略的工程价值。
- 进入更大样本 phase 前，建议优先复核误触发率与漏掉有益通信率是否同时可控，避免只依赖总体准确率决策。

## 局限性

- 当前 Oracle 是基于现有控制组构造的近似标签，因此更适合做策略筛选，而非最终机制证明。
- 本报告反映的是当前 phase 的黑盒实现效果，不能直接等同于可访问内部不确定性信号时的理想上界。

## 复现与产物说明

- 运行目录：`runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`oracle_trigger_eval.json`。
- 本地报告与发布报告共享同一套 run 内图资产，便于后续引用与审稿期复核。

## 图表资产

### CUE 成本-性能前沿

![CUE 成本-性能前沿](../../runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各策略的准确率相对于平均总 token 的位置关系。*

### CUE 效率排序

![CUE 效率排序](../../runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### CUE 跨数据集表现

![CUE 跨数据集表现](../../runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各策略在不同数据集上的准确率分布。*

### 策略触发率权衡

![策略触发率权衡](../../runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5/figures/policy_tradeoff.svg)

*总体触发率相对于准确率的变化。*

### Oracle 对齐情况

![Oracle 对齐情况](../../runs/cue/cue_black_box_utility_main/smoke20/20260509T125852Z-cue_black_box_utility_main-smoke20-xiaomimimo-mimo-v2.5/figures/oracle_alignment.svg)

*总体 Oracle 精确率与召回率对比。*
