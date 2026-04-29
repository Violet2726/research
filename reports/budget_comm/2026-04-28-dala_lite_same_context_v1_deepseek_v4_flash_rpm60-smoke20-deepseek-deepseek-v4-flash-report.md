# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_same_context_v1_deepseek_v4_flash_rpm60`
- 轨道：`same_context`
- Phase：`smoke20`
- Backbone：`deepseek/deepseek-v4-flash`
- 运行目录：`runs/budget_comm/dala_lite_same_context_v1_deepseek_v4_flash_rpm60/smoke20/20260428T021337Z-dala_lite_same_context_v1_deepseek_v4_flash_rpm60-smoke20-deepseek-deepseek-v4-flash`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `gsm8k`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`68`，`round_budget_tokens`=`27`。
- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`177`，`round_budget_tokens`=`70`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`231`，`round_budget_tokens`=`92`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 2543.88 | 3.00 | 0.275170 |
| `all_to_all_full` | 0.8167 | 198.37 | 5465.42 | 6.00 | 0.149424 |
| `budget_random` | 0.7333 | 36.45 | 5043.82 | 6.00 | 0.145392 |
| `budget_confidence` | 0.7333 | 34.32 | 5039.12 | 6.00 | 0.145528 |
| `dala_lite` | 0.7667 | 27.38 | 5033.00 | 6.00 | 0.152328 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 7 | 0 |
| `budget_random` | 0.7500 | 0.4788 | 0.2500 | 0.0000 | 0.0000 | 0.7500 | 2 | 0 |
| `budget_confidence` | 0.7500 | 0.4534 | 0.2500 | 0.0000 | 0.0000 | 0.7500 | 3 | 1 |
| `dala_lite` | 0.4833 | 0.3713 | 0.1444 | 0.0000 | 0.0167 | 0.8389 | 4 | 0 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1286.55 | 0.466364 |
| `all_to_all_full` | 0.9000 | 145.95 | 2930.75 | 0.307089 |
| `budget_random` | 0.7000 | 2.60 | 2456.80 | 0.284923 |
| `budget_confidence` | 0.7500 | 2.60 | 2456.50 | 0.305312 |
| `dala_lite` | 0.7500 | 3.60 | 2475.65 | 0.302951 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5271.10 | 0.142285 |
| `all_to_all_full` | 0.8000 | 266.25 | 10975.40 | 0.072890 |
| `budget_random` | 0.7500 | 54.10 | 10499.85 | 0.071430 |
| `budget_confidence` | 0.7500 | 49.70 | 10490.00 | 0.071497 |
| `dala_lite` | 0.7500 | 41.40 | 10477.60 | 0.071581 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 1074.00 | 0.698324 |
| `all_to_all_full` | 0.7500 | 182.90 | 2490.10 | 0.301193 |
| `budget_random` | 0.7500 | 52.65 | 2174.80 | 0.344859 |
| `budget_confidence` | 0.7000 | 50.65 | 2170.85 | 0.322454 |
| `dala_lite` | 0.8000 | 37.15 | 2145.75 | 0.372830 |

## 6. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00588`
- 问题预览：Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next t...
- 金标：`52`
- `all_to_all_full`：`52` / score=`1.0`
- `dala_lite`：`50.67` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00906`
- 问题预览：Tim has a box with 7 blue shoe boxes and 9 red shoe boxes. If he uses 3 blue shoeboxes and 1/3 red of his shoeboxes t...
- 金标：`10`
- `all_to_all_full`：`10` / score=`1.0`
- `dala_lite`：`4` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-01252`
- 问题预览：Dominick went to his team's changing room and saw half as many robots as helmets and half as many helmets as football...
- 金标：`70`
- `all_to_all_full`：`70` / score=`1.0`
- `dala_lite`：`140` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5ab514c05542991779162d72`
- 问题预览：The school in which the Wilmslow Show is held is designated as what?
- 金标：`Centre of Excellence`
- `all_to_all_full`：`centre of excellence` / score=`1.0`
- `dala_lite`：`wilmslow high school` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 5

- 数据集：`strategyqa`
- 样本：`62e8d2b87e80f78b152c`
- 问题预览：Does an individual oceanographer study many sciences?
- 金标：`yes`
- `all_to_all_full`：`no` / score=`0.0`
- `dala_lite`：`yes` / score=`1.0`
- 说明：dala_lite 在同预算下优于 budget_confidence。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.133333, 0.016667]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.05`
- `communication_ratio_vs_all_to_all_full`：`0.138044`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

