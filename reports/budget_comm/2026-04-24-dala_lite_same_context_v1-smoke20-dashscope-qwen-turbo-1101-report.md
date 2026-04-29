# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_same_context_v1`
- 轨道：`same_context`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/budget_comm/dala_lite_same_context_v1/smoke20/20260424T040321Z-dala_lite_same_context_v1-smoke20-dashscope-qwen-turbo-1101`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `gsm8k`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`145`，`round_budget_tokens`=`58`。
- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`92`，`round_budget_tokens`=`36`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`202`，`round_budget_tokens`=`80`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5333 | 0.00 | 2443.62 | 3.00 | 0.218256 |
| `all_to_all_full` | 0.5667 | 149.08 | 5227.85 | 6.00 | 0.108394 |
| `budget_random` | 0.5833 | 43.25 | 4939.47 | 6.00 | 0.118096 |
| `budget_confidence` | 0.5667 | 41.00 | 4937.85 | 6.00 | 0.114760 |
| `dala_lite` | 0.5667 | 34.10 | 4919.50 | 6.00 | 0.115188 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 2 | 0 |
| `budget_random` | 1.0167 | 0.7544 | 0.3389 | 0.0000 | 0.0000 | 0.6611 | 3 | 0 |
| `budget_confidence` | 1.0167 | 0.7176 | 0.3389 | 0.0000 | 0.0000 | 0.6611 | 2 | 0 |
| `dala_lite` | 0.8167 | 0.6109 | 0.2556 | 0.0000 | 0.0167 | 0.7278 | 2 | 0 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4500 | 0.00 | 1129.90 | 0.398265 |
| `all_to_all_full` | 0.5500 | 131.35 | 2662.70 | 0.206557 |
| `budget_random` | 0.6000 | 42.15 | 2359.65 | 0.254275 |
| `budget_confidence` | 0.5500 | 41.00 | 2358.05 | 0.233244 |
| `dala_lite` | 0.5500 | 41.90 | 2353.95 | 0.233650 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 5308.80 | 0.103602 |
| `all_to_all_full` | 0.5500 | 216.20 | 10998.60 | 0.050006 |
| `budget_random` | 0.5500 | 58.70 | 10618.80 | 0.051795 |
| `budget_confidence` | 0.5500 | 54.45 | 10614.55 | 0.051816 |
| `dala_lite` | 0.5500 | 37.15 | 10577.10 | 0.051999 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 892.15 | 0.672533 |
| `all_to_all_full` | 0.6000 | 99.70 | 2022.25 | 0.296699 |
| `budget_random` | 0.6000 | 28.90 | 1839.95 | 0.326096 |
| `budget_confidence` | 0.6000 | 27.55 | 1840.95 | 0.325919 |
| `dala_lite` | 0.6000 | 23.25 | 1827.45 | 0.328326 |

## 6. 失败案例

- 当前 smoke20 下没有收集到稳定失败案例。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.000000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`False`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.0`
- `communication_ratio_vs_all_to_all_full`：`0.228731`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

