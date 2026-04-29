# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_split_context_v1_deepseek_v4_flash_rpm60`
- 轨道：`split_context`
- Phase：`smoke20`
- Backbone：`deepseek/deepseek-v4-flash`
- 运行目录：`runs/budget_comm/dala_lite_split_context_v1_deepseek_v4_flash_rpm60/smoke20/20260428T022136Z-dala_lite_split_context_v1_deepseek_v4_flash_rpm60-smoke20-deepseek-deepseek-v4-flash`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`165`，`round_budget_tokens`=`66`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`256`，`round_budget_tokens`=`102`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5750 | 0.00 | 1972.17 | 3.00 | 0.291556 |
| `all_to_all_full` | 0.8250 | 239.10 | 4383.73 | 6.00 | 0.188196 |
| `budget_random` | 0.7250 | 64.33 | 3974.28 | 6.00 | 0.182423 |
| `budget_confidence` | 0.6750 | 60.15 | 3967.43 | 6.00 | 0.170136 |
| `dala_lite` | 0.6250 | 35.67 | 3917.07 | 6.00 | 0.159558 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 10 | 0 |
| `budget_random` | 1.0250 | 0.7666 | 0.3417 | 0.0000 | 0.0000 | 0.6583 | 6 | 0 |
| `budget_confidence` | 1.0000 | 0.7238 | 0.3333 | 0.0000 | 0.0000 | 0.6667 | 4 | 0 |
| `dala_lite` | 0.7000 | 0.4215 | 0.1833 | 0.0000 | 0.0500 | 0.7667 | 4 | 2 |

## 5. 数据集分表

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.3500 | 0.00 | 2736.55 | 0.127898 |
| `all_to_all_full` | 0.8000 | 293.75 | 6022.70 | 0.132831 |
| `budget_random` | 0.5500 | 77.80 | 5513.55 | 0.099754 |
| `budget_confidence` | 0.5500 | 70.15 | 5500.20 | 0.099996 |
| `dala_lite` | 0.4500 | 44.50 | 5449.00 | 0.082584 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 1207.80 | 0.662361 |
| `all_to_all_full` | 0.8500 | 184.45 | 2744.75 | 0.309682 |
| `budget_random` | 0.9000 | 50.85 | 2435.00 | 0.369610 |
| `budget_confidence` | 0.8000 | 50.15 | 2434.65 | 0.328589 |
| `dala_lite` | 0.8000 | 26.85 | 2385.15 | 0.335409 |

## 6. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a7129685542994082a3e5fa`
- 问题预览：Which "Blackzilians" fighter is currently competing in the Middleweight division of Ultimate Fighting Championship?
- 金标：`Vitor Belfort`
- `all_to_all_full`：`vitor belfort` / score=`1.0`
- `dala_lite`：`rashad evans` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5a73977d554299623ed4ac08`
- 问题预览：What is the shared country of ancestry between Art Laboe and Scout Tufankjian?
- 金标：`Armenian`
- `all_to_all_full`：`armenian` / score=`1.0`
- `dala_lite`：`armenia` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a88d6df554299206df2b377`
- 问题预览：What animated movie, starring Danny Devito, featured music written and produced by Kool Kojak?
- 金标：`The Lorax`
- `all_to_all_full`：`lorax` / score=`1.0`
- `dala_lite`：`dr seusss lorax` / score=`0.0`
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

- 数据集：`hotpotqa`
- 样本：`5ae0132d55429925eb1afc00`
- 问题预览：The Soul of Buddha is a 1918 American silent romance film shot in a borough that is the western terminus of what?
- 金标：`the George Washington Bridge`
- `all_to_all_full`：`george washington bridge` / score=`1.0`
- `dala_lite`：`fort lee new jersey` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.325000, -0.075000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`False`
- `dala_beats_budget_random_on_acc_per_1k`：`False`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.2`
- `communication_ratio_vs_all_to_all_full`：`0.149205`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

