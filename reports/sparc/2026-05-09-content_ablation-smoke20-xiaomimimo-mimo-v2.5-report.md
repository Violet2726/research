# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6667 | 0.00 | 2453.77 | 3.00 | 0.271691 | 1.0000 |
| `full_cot` | 0.7333 | 3372.53 | 5826.30 | 6.00 | 0.125866 | 0.0000 |
| `answer_only` | 0.6500 | 2535.65 | 4989.42 | 6.00 | 0.130276 | 0.2481 |
| `answer_confidence` | 0.6833 | 2614.83 | 5068.60 | 6.00 | 0.134817 | 0.2247 |
| `disagreement_step_only` | 0.6833 | 2722.18 | 5175.95 | 6.00 | 0.132021 | 0.1928 |
| `critical_evidence_only` | 0.7000 | 2814.18 | 5267.95 | 6.00 | 0.132879 | 0.1656 |
| `task_adaptive` | 0.6833 | 2764.05 | 5217.82 | 6.00 | 0.130961 | 0.1804 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.4000 | 0.00 | 2197.56 |
| `full_cot` | 0.5600 | 3249.88 | 5447.44 |
| `answer_only` | 0.4000 | 2340.36 | 4537.92 |
| `answer_confidence` | 0.4400 | 2407.48 | 4605.04 |
| `disagreement_step_only` | 0.4400 | 2495.92 | 4693.48 |
| `critical_evidence_only` | 0.4800 | 2604.48 | 4802.04 |
| `task_adaptive` | 0.4400 | 2562.08 | 4759.64 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 2681.17 |
| `full_cot` | 1.0000 | 3887.50 | 6568.67 |
| `answer_only` | 0.0000 | 2761.33 | 5442.50 |
| `answer_confidence` | 0.3333 | 2888.00 | 5569.17 |
| `disagreement_step_only` | 0.5000 | 2980.33 | 5661.50 |
| `critical_evidence_only` | 0.5000 | 3055.33 | 5736.50 |
| `task_adaptive` | 0.3333 | 3038.67 | 5719.83 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.7333`，total tokens=`5826.30`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Compression tradeoff

![Compression tradeoff](../../runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5/figures/compression_tradeoff.svg)

*Overall compression ratio versus accuracy across content-ablation variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/content_ablation/smoke20/20260509T125850Z-content_ablation-smoke20-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
