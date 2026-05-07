# SPARC 内容消融报告

## 1. 实验范围与公平性说明

- 实验名：`content_ablation_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/content_ablation_v1/pilot100/20260507T121725Z-content_ablation_v1-pilot100-xiaomimimo-mimo-v2.5`
- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。
- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6267 | 0.00 | 2443.48 | 3.00 | 0.256465 | 1.0000 |
| `full_cot` | 0.7200 | 3330.89 | 5774.37 | 6.00 | 0.124689 | 0.0000 |
| `answer_only` | 0.6133 | 2525.39 | 4968.87 | 6.00 | 0.123435 | 0.2418 |
| `answer_confidence` | 0.6167 | 2612.86 | 5056.34 | 6.00 | 0.121959 | 0.2156 |
| `disagreement_step_only` | 0.6200 | 2714.93 | 5158.41 | 6.00 | 0.120192 | 0.1849 |
| `critical_evidence_only` | 0.6567 | 2795.47 | 5238.95 | 6.00 | 0.125343 | 0.1607 |
| `task_adaptive` | 0.6233 | 2747.92 | 5191.40 | 6.00 | 0.120070 | 0.1750 |

## 3. 条件子集

### initial_disagreement = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.4118 | 0.00 | 2054.35 |
| `full_cot` | 0.6569 | 3015.10 | 5069.45 |
| `answer_only` | 0.3922 | 2214.75 | 4269.10 |
| `answer_confidence` | 0.3824 | 2308.92 | 4363.27 |
| `disagreement_step_only` | 0.3922 | 2387.46 | 4441.81 |
| `critical_evidence_only` | 0.4804 | 2446.17 | 4500.52 |
| `task_adaptive` | 0.4020 | 2398.30 | 4452.66 |

### oracle_positive = true

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 1611.97 |
| `full_cot` | 1.0000 | 2684.26 | 4296.23 |
| `answer_only` | 0.0323 | 1768.48 | 3380.45 |
| `answer_confidence` | 0.0000 | 1842.81 | 3454.77 |
| `disagreement_step_only` | 0.0968 | 1951.42 | 3563.39 |
| `critical_evidence_only` | 0.2581 | 1972.32 | 3584.29 |
| `task_adaptive` | 0.1290 | 1938.58 | 3550.55 |

## 4. 探索性区间

- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 6. 下一轮建议

- 推荐默认消息模式：`full_cot`
- 推荐依据：overall accuracy=`0.7200`，total tokens=`5774.37`。

