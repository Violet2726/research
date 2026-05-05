# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_same_context_v1`
- 轨道：`same_context`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_same_context_v1/smoke20/20260505T090136Z-dala_lite_same_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `gsm8k`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`170`，`round_budget_tokens`=`68`。
- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`201`，`round_budget_tokens`=`80`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`315`，`round_budget_tokens`=`126`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6167 | 0.00 | 2618.38 | 3.00 | 0.235514 |
| `all_to_all_full` | 0.6833 | 229.27 | 5753.30 | 6.00 | 0.118772 |
| `budget_random` | 0.7000 | 66.77 | 5350.07 | 6.00 | 0.130839 |
| `budget_confidence` | 0.7000 | 65.47 | 5343.37 | 6.00 | 0.131004 |
| `dala_lite` | 0.6833 | 59.30 | 5322.45 | 6.00 | 0.128387 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 4 | 0 |
| `budget_random` | 1.0833 | 0.7245 | 0.3611 | 0.0000 | 0.0000 | 0.6389 | 5 | 0 |
| `budget_confidence` | 1.0833 | 0.7067 | 0.3611 | 0.0000 | 0.0000 | 0.6389 | 5 | 0 |
| `dala_lite` | 0.9333 | 0.6398 | 0.2611 | 0.0000 | 0.0500 | 0.6889 | 4 | 0 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 1303.95 | 0.421795 |
| `all_to_all_full` | 0.7500 | 156.40 | 3123.05 | 0.240150 |
| `budget_random` | 0.7500 | 45.70 | 2751.95 | 0.272534 |
| `budget_confidence` | 0.7500 | 43.20 | 2727.95 | 0.274932 |
| `dala_lite` | 0.6500 | 35.25 | 2691.75 | 0.241479 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 5467.30 | 0.128034 |
| `all_to_all_full` | 0.7000 | 314.75 | 11513.05 | 0.060801 |
| `budget_random` | 0.7000 | 94.45 | 11052.95 | 0.063332 |
| `budget_confidence` | 0.7000 | 94.30 | 11052.45 | 0.063334 |
| `dala_lite` | 0.7000 | 83.70 | 11018.50 | 0.063530 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1083.90 | 0.553557 |
| `all_to_all_full` | 0.6000 | 216.65 | 2623.80 | 0.228676 |
| `budget_random` | 0.6500 | 60.15 | 2245.30 | 0.289494 |
| `budget_confidence` | 0.6500 | 58.90 | 2249.70 | 0.288927 |
| `dala_lite` | 0.7000 | 58.95 | 2257.10 | 0.310132 |

## 6. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00618`
- 问题预览：The school auditorium has 4 rows of seats. There are 18 seats in each row. One-fourth of the seats were occupied by t...
- 金标：`36`
- `all_to_all_full`：`36` / score=`1.0`
- `dala_lite`：`24` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00945`
- 问题预览：James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...
- 金标：`17`
- `all_to_all_full`：`17` / score=`1.0`
- `dala_lite`：`18` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 3

- 数据集：`strategyqa`
- 样本：`a5f8af1dd0e9c46c47be`
- 问题预览：Does Orange County, California require airplanes to be quiet?
- 金标：`yes`
- `all_to_all_full`：`no` / score=`0.0`
- `dala_lite`：`yes` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.066667, 0.066667]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`False`
- `dala_beats_budget_random_on_acc_per_1k`：`False`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.0`
- `communication_ratio_vs_all_to_all_full`：`0.258651`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

