# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_same_context_v1`
- 轨道：`same_context`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_same_context_v1/pilot100/20260506T140645Z-dala_lite_same_context_v1-pilot100-xiaomimimo-mimo-v2.5`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `gsm8k`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`167`，`round_budget_tokens`=`66`。
- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`214`，`round_budget_tokens`=`85`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`297`，`round_budget_tokens`=`118`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5967 | 0.00 | 2609.69 | 3.00 | 0.228635 |
| `all_to_all_full` | 0.7233 | 232.78 | 5746.42 | 6.00 | 0.125875 |
| `budget_random` | 0.6900 | 63.89 | 5327.40 | 6.00 | 0.129519 |
| `budget_confidence` | 0.6700 | 60.65 | 5322.85 | 6.00 | 0.125872 |
| `dala_lite` | 0.7100 | 46.71 | 5280.35 | 6.00 | 0.134461 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 38 | 0 |
| `budget_random` | 1.0133 | 0.7036 | 0.3378 | 0.0000 | 0.0000 | 0.6622 | 32 | 4 |
| `budget_confidence` | 1.0133 | 0.6713 | 0.3378 | 0.0000 | 0.0000 | 0.6622 | 25 | 3 |
| `dala_lite` | 0.7300 | 0.5244 | 0.2189 | 0.0000 | 0.0244 | 0.7567 | 37 | 3 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4900 | 0.00 | 1357.10 | 0.361064 |
| `all_to_all_full` | 0.8100 | 177.63 | 3271.83 | 0.247568 |
| `budget_random` | 0.7100 | 41.04 | 2836.83 | 0.250279 |
| `budget_confidence` | 0.7000 | 39.89 | 2838.41 | 0.246617 |
| `dala_lite` | 0.7700 | 31.02 | 2771.30 | 0.277848 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6400 | 0.00 | 5395.92 | 0.118608 |
| `all_to_all_full` | 0.6700 | 303.69 | 11361.90 | 0.058969 |
| `budget_random` | 0.6800 | 86.10 | 10897.34 | 0.062401 |
| `budget_confidence` | 0.6600 | 79.57 | 10886.21 | 0.060627 |
| `dala_lite` | 0.6700 | 54.80 | 10839.49 | 0.061811 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6600 | 0.00 | 1076.06 | 0.613349 |
| `all_to_all_full` | 0.6900 | 217.03 | 2605.54 | 0.264820 |
| `budget_random` | 0.6800 | 64.54 | 2248.04 | 0.302486 |
| `budget_confidence` | 0.6500 | 62.50 | 2243.93 | 0.289670 |
| `dala_lite` | 0.6900 | 54.31 | 2230.25 | 0.309382 |

## 6. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00066`
- 问题预览：There are four schools competing at a basketball tournament. Each school has sent a girls’ basketball team and a boys...
- 金标：`48`
- `all_to_all_full`：`48` / score=`1.0`
- `dala_lite`：`56` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00100`
- 问题预览：Jerome had 4 friends who came to visit him on a certain day. The first friend pressed on the doorbell 20 times before...
- 金标：`175`
- `all_to_all_full`：`175` / score=`1.0`
- `dala_lite`：`175` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00132`
- 问题预览：Pam and Fred went to a carnival. Pam rode the roller coaster 2 times while Fred rode it 4 times. After that, each of ...
- 金标：`60`
- `all_to_all_full`：`60` / score=`1.0`
- `dala_lite`：`72` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00238`
- 问题预览：Mary is two years younger than Joan, who is five years older than Jessa. If Jessa is 20 years old, what is the sum of...
- 金标：`68`
- `all_to_all_full`：`68` / score=`1.0`
- `dala_lite`：`58` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00364`
- 问题预览：There are 96 fourth-graders at Small Tree School. 43 of them are girls. On Friday, 5 fourth-grade girls and 4 fourth-...
- 金标：`49`
- `all_to_all_full`：`49` / score=`1.0`
- `dala_lite`：`49` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.046667, 0.020000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`True`
- 原因：`all_conditions_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.013333`
- `communication_ratio_vs_all_to_all_full`：`0.200659`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

