# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation_v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/sparc/content_ablation/20260421T083855Z-content_ablation_v1-smoke20-dashscope-qwen-turbo-1101`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5333 | 0.00 | 2262.87 | 3.00 | 0.235689 | 1.0000 |
| `full_cot` | 0.6000 | 2885.87 | 5148.73 | 6.00 | 0.116534 | 0.0000 |
| `answer_only` | 0.5500 | 2417.57 | 4680.43 | 6.00 | 0.117510 | 0.1623 |
| `answer_confidence` | 0.5833 | 2520.30 | 4783.17 | 6.00 | 0.121955 | 0.1267 |
| `disagreement_step_only` | 0.5833 | 2532.50 | 4795.37 | 6.00 | 0.121645 | 0.1224 |
| `critical_evidence_only` | 0.5500 | 2533.87 | 4796.73 | 6.00 | 0.114661 | 0.1220 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.2000 | 0.00 | 2039.67 |
| `full_cot` | 0.4667 | 2720.60 | 4760.27 |
| `answer_only` | 0.2667 | 2210.73 | 4250.40 |
| `answer_confidence` | 0.4000 | 2305.60 | 4345.27 |
| `disagreement_step_only` | 0.4000 | 2319.67 | 4359.33 |
| `critical_evidence_only` | 0.2667 | 2308.20 | 4347.87 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1609.80 |
| `full_cot` | 1.0000 | 2387.40 | 3997.20 |
| `answer_only` | 0.2000 | 1737.80 | 3347.60 |
| `answer_confidence` | 0.6000 | 1832.00 | 3441.80 |
| `disagreement_step_only` | 0.4000 | 1859.60 | 3469.40 |
| `critical_evidence_only` | 0.4000 | 1849.80 | 3459.60 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.6000`，total tokens=`5148.73`。

