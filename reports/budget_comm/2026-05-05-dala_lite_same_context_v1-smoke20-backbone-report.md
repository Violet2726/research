# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_same_context_v1`
- 轨道：`same_context`
- Phase：`smoke20`
- Backbone：`None`
- 运行目录：`runs/budget_comm/dala_lite_same_context_v1/smoke20/20260505T050230Z-dala_lite_same_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `gsm8k`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`167`，`round_budget_tokens`=`66`。
- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`201`，`round_budget_tokens`=`80`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`315`，`round_budget_tokens`=`126`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 2617.27 | 3.00 | 0.229247 |
| `all_to_all_full` | 0.6667 | 232.83 | 5768.98 | 6.00 | 0.115561 |
| `budget_random` | 0.7167 | 65.77 | 5331.42 | 6.00 | 0.134423 |
| `budget_confidence` | 0.7167 | 64.52 | 5333.07 | 6.00 | 0.134382 |
| `dala_lite` | 0.7167 | 60.93 | 5316.88 | 6.00 | 0.134791 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 5 | 1 |
| `budget_random` | 1.0333 | 0.7161 | 0.3444 | 0.0000 | 0.0000 | 0.6556 | 7 | 0 |
| `budget_confidence` | 1.0333 | 0.6987 | 0.3444 | 0.0000 | 0.0000 | 0.6556 | 7 | 0 |
| `dala_lite` | 1.0000 | 0.6698 | 0.2778 | 0.0000 | 0.0556 | 0.6667 | 8 | 1 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5000 | 0.00 | 1300.60 | 0.384438 |
| `all_to_all_full` | 0.7000 | 167.10 | 3170.10 | 0.220813 |
| `budget_random` | 0.8000 | 42.70 | 2696.00 | 0.296736 |
| `budget_confidence` | 0.8000 | 40.35 | 2697.05 | 0.296620 |
| `dala_lite` | 0.7500 | 40.15 | 2675.05 | 0.280369 |

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

- 数据集：`strategyqa`
- 样本：`a5f8af1dd0e9c46c47be`
- 问题预览：Does Orange County, California require airplanes to be quiet?
- 金标：`yes`
- `all_to_all_full`：`no` / score=`0.0`
- `dala_lite`：`yes` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.116667]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`True`
- 原因：`all_conditions_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`-0.05`
- `communication_ratio_vs_all_to_all_full`：`0.261704`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

