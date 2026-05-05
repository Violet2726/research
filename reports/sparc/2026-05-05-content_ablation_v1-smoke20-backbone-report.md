# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation_v1`
- Phase：`smoke20`
- Backbone：`None`
- 运行目录：`runs/sparc/content_ablation_v1/smoke20/20260505T052741Z-content_ablation_v1-smoke20-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 2429.68 | 3.00 | 0.246946 | 1.0000 |
| `full_cot` | 0.7667 | 3322.90 | 5752.58 | 6.00 | 0.133274 | 0.0000 |
| `answer_only` | 0.6000 | 2533.17 | 4962.85 | 6.00 | 0.120898 | 0.2377 |
| `answer_confidence` | 0.6000 | 2621.13 | 5050.82 | 6.00 | 0.118793 | 0.2112 |
| `disagreement_step_only` | 0.6500 | 2720.63 | 5150.32 | 6.00 | 0.126206 | 0.1812 |
| `critical_evidence_only` | 0.6500 | 2799.12 | 5228.80 | 6.00 | 0.124312 | 0.1576 |
| `task_adaptive` | 0.6500 | 2747.43 | 5177.12 | 6.00 | 0.125553 | 0.1732 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.2692 | 0.00 | 2060.58 |
| `full_cot` | 0.5769 | 3034.69 | 5095.27 |
| `answer_only` | 0.2692 | 2222.38 | 4282.96 |
| `answer_confidence` | 0.2692 | 2296.96 | 4357.54 |
| `disagreement_step_only` | 0.3846 | 2397.69 | 4458.27 |
| `critical_evidence_only` | 0.3846 | 2464.15 | 4524.73 |
| `task_adaptive` | 0.3846 | 2404.88 | 4465.46 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1763.70 |
| `full_cot` | 1.0000 | 2787.80 | 4551.50 |
| `answer_only` | 0.0000 | 1966.50 | 3730.20 |
| `answer_confidence` | 0.0000 | 2027.20 | 3790.90 |
| `disagreement_step_only` | 0.2000 | 2097.60 | 3861.30 |
| `critical_evidence_only` | 0.2000 | 2174.10 | 3937.80 |
| `task_adaptive` | 0.2000 | 2112.10 | 3875.80 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.7667`，total tokens=`5752.58`。

