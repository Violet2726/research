# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_split_context_v1`
- 轨道：`split_context`
- Phase：`smoke20`
- Backbone：`None`
- 运行目录：`runs/budget_comm/dala_lite_split_context_v1/smoke20/20260505T054055Z-dala_lite_split_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`190`，`round_budget_tokens`=`76`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`360`，`round_budget_tokens`=`144`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 2000.67 | 3.00 | 0.274907 |
| `all_to_all_full` | 0.7500 | 284.90 | 4636.55 | 6.00 | 0.161758 |
| `budget_random` | 0.6250 | 84.17 | 4172.20 | 6.00 | 0.149801 |
| `budget_confidence` | 0.7500 | 78.25 | 4152.38 | 6.00 | 0.180620 |
| `dala_lite` | 0.7500 | 75.83 | 4141.43 | 6.00 | 0.181097 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 8 | 0 |
| `budget_random` | 0.9750 | 0.7567 | 0.3250 | 0.0000 | 0.0000 | 0.6750 | 4 | 1 |
| `budget_confidence` | 0.9750 | 0.7068 | 0.3250 | 0.0000 | 0.0000 | 0.6750 | 8 | 0 |
| `dala_lite` | 0.9500 | 0.6838 | 0.2833 | 0.0000 | 0.0333 | 0.6833 | 8 | 0 |

## 5. 数据集分表

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.2500 | 0.00 | 2809.65 | 0.088979 |
| `all_to_all_full` | 0.6500 | 370.35 | 6445.45 | 0.100846 |
| `budget_random` | 0.4000 | 112.95 | 5870.10 | 0.068142 |
| `budget_confidence` | 0.6500 | 103.90 | 5834.20 | 0.111412 |
| `dala_lite` | 0.6000 | 101.05 | 5822.05 | 0.103056 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8500 | 0.00 | 1191.70 | 0.713267 |
| `all_to_all_full` | 0.8500 | 199.45 | 2827.65 | 0.300603 |
| `budget_random` | 0.8500 | 55.40 | 2474.30 | 0.343532 |
| `budget_confidence` | 0.8500 | 52.60 | 2470.55 | 0.344053 |
| `dala_lite` | 0.9000 | 50.60 | 2460.80 | 0.365735 |

## 6. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a77897f55429949eeb29edc`
- 问题预览：Jason Regler, stated that he had the idea for the flashing wristbands during a song built around which instrument ?
- 金标：`an organ`
- `all_to_all_full`：`organ` / score=`1.0`
- `dala_lite`：`fix you` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5ab514c05542991779162d72`
- 问题预览：The school in which the Wilmslow Show is held is designated as what?
- 金标：`Centre of Excellence`
- `all_to_all_full`：`centre of excellence` / score=`1.0`
- `dala_lite`：`centre of excellence` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

### Case 3

- 数据集：`strategyqa`
- 样本：`66b3cfaa499773bdf513`
- 问题预览：Would it be difficult for Will Ferrell to win Empire Award for Best Newcomer?
- 金标：`yes`
- `all_to_all_full`：`no` / score=`0.0`
- `dala_lite`：`yes` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.075000, 0.075000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`True`
- 原因：`all_conditions_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`True`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.0`
- `communication_ratio_vs_all_to_all_full`：`0.266146`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

