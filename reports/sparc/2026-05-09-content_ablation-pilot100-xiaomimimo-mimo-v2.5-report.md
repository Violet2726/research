# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6400 | 0.00 | 2445.76 | 3.00 | 0.261677 | 1.0000 |
| `full_cot` | 0.7300 | 3346.04 | 5791.80 | 6.00 | 0.126040 | 0.0000 |
| `answer_only` | 0.6300 | 2529.57 | 4975.34 | 6.00 | 0.126625 | 0.2440 |
| `answer_confidence` | 0.6400 | 2610.26 | 5056.02 | 6.00 | 0.126582 | 0.2199 |
| `disagreement_step_only` | 0.6567 | 2711.27 | 5157.03 | 6.00 | 0.127334 | 0.1897 |
| `critical_evidence_only` | 0.6600 | 2796.77 | 5242.54 | 6.00 | 0.125893 | 0.1642 |
| `task_adaptive` | 0.6400 | 2752.93 | 5198.69 | 6.00 | 0.123108 | 0.1773 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.4386 | 0.00 | 2083.60 |
| `full_cot` | 0.6404 | 3098.02 | 5181.61 |
| `answer_only` | 0.4211 | 2229.38 | 4312.97 |
| `answer_confidence` | 0.4386 | 2306.94 | 4390.54 |
| `disagreement_step_only` | 0.4825 | 2390.43 | 4474.03 |
| `critical_evidence_only` | 0.4737 | 2463.36 | 4546.96 |
| `task_adaptive` | 0.4386 | 2424.68 | 4508.28 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1704.61 |
| `full_cot` | 1.0000 | 2732.39 | 4437.00 |
| `answer_only` | 0.0000 | 1849.35 | 3553.97 |
| `answer_confidence` | 0.0968 | 1914.65 | 3619.26 |
| `disagreement_step_only` | 0.2258 | 1986.52 | 3691.13 |
| `critical_evidence_only` | 0.3226 | 2027.68 | 3732.29 |
| `task_adaptive` | 0.1613 | 1983.13 | 3687.74 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.7300`，total tokens=`5791.80`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Compression tradeoff

![Compression tradeoff](../../runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5/figures/compression_tradeoff.svg)

*Overall compression ratio versus accuracy across content-ablation variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/content_ablation/pilot100/20260509T125926Z-content_ablation-pilot100-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
