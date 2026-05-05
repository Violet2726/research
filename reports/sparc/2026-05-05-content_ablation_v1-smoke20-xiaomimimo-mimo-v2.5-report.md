# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/content_ablation_v1/smoke20/20260505T095502Z-content_ablation_v1-smoke20-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 2432.85 | 3.00 | 0.267176 | 1.0000 |
| `full_cot` | 0.8167 | 3318.17 | 5751.02 | 6.00 | 0.142004 | 0.0000 |
| `answer_only` | 0.6333 | 2532.23 | 4965.08 | 6.00 | 0.127557 | 0.2369 |
| `answer_confidence` | 0.6333 | 2618.88 | 5051.73 | 6.00 | 0.125369 | 0.2107 |
| `disagreement_step_only` | 0.6667 | 2712.07 | 5144.92 | 6.00 | 0.129578 | 0.1827 |
| `critical_evidence_only` | 0.6833 | 2799.37 | 5232.22 | 6.00 | 0.130601 | 0.1564 |
| `task_adaptive` | 0.6667 | 2753.78 | 5186.63 | 6.00 | 0.128536 | 0.1701 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.4000 | 0.00 | 1878.00 |
| `full_cot` | 0.8000 | 2877.75 | 4755.75 |
| `answer_only` | 0.3500 | 2052.05 | 3930.05 |
| `answer_confidence` | 0.3500 | 2143.45 | 4021.45 |
| `disagreement_step_only` | 0.4500 | 2199.60 | 4077.60 |
| `critical_evidence_only` | 0.4500 | 2279.65 | 4157.65 |
| `task_adaptive` | 0.4500 | 2208.80 | 4086.80 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1574.10 |
| `full_cot` | 1.0000 | 2632.60 | 4206.70 |
| `answer_only` | 0.0000 | 1730.70 | 3304.80 |
| `answer_confidence` | 0.0000 | 1864.90 | 3439.00 |
| `disagreement_step_only` | 0.0000 | 1916.00 | 3490.10 |
| `critical_evidence_only` | 0.2000 | 2003.90 | 3578.00 |
| `task_adaptive` | 0.0000 | 1884.90 | 3459.00 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.8167`，total tokens=`5751.02`。

