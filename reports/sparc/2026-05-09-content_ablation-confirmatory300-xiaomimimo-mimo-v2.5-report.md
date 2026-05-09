# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6128 | 0.00 | 2602.72 | 3.00 | 0.235440 | 1.0000 |
| `full_cot` | 0.7382 | 3501.25 | 6103.97 | 6.00 | 0.120944 | 0.0000 |
| `answer_only` | 0.6104 | 2681.86 | 5284.59 | 6.00 | 0.115501 | 0.2340 |
| `answer_confidence` | 0.6188 | 2761.39 | 5364.12 | 6.00 | 0.115362 | 0.2113 |
| `disagreement_step_only` | 0.6333 | 2861.10 | 5463.82 | 6.00 | 0.115907 | 0.1828 |
| `critical_evidence_only` | 0.6478 | 2949.01 | 5551.73 | 6.00 | 0.116679 | 0.1577 |
| `task_adaptive` | 0.6261 | 2906.37 | 5509.09 | 6.00 | 0.113640 | 0.1699 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.3345 | 0.00 | 2081.18 |
| `full_cot` | 0.6586 | 3070.96 | 5152.14 |
| `answer_only` | 0.3310 | 2231.44 | 4312.62 |
| `answer_confidence` | 0.3517 | 2309.61 | 4390.80 |
| `disagreement_step_only` | 0.3862 | 2389.32 | 4470.50 |
| `critical_evidence_only` | 0.4172 | 2450.68 | 4531.87 |
| `task_adaptive` | 0.3655 | 2418.32 | 4499.50 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1539.01 |
| `full_cot` | 1.0000 | 2541.39 | 4080.40 |
| `answer_only` | 0.0090 | 1691.99 | 3231.00 |
| `answer_confidence` | 0.0721 | 1759.41 | 3298.41 |
| `disagreement_step_only` | 0.1892 | 1818.26 | 3357.27 |
| `critical_evidence_only` | 0.3153 | 1859.94 | 3398.95 |
| `task_adaptive` | 0.1982 | 1833.21 | 3372.22 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.7382`，total tokens=`6103.97`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Compression tradeoff

![Compression tradeoff](../../runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/compression_tradeoff.svg)

*Overall compression ratio versus accuracy across content-ablation variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/content_ablation/confirmatory300/20260509T130044Z-content_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
