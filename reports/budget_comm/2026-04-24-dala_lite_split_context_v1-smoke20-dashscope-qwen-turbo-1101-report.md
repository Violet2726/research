# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_split_context_v1`
- 轨道：`split_context`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/budget_comm/dala_lite_split_context_v1/smoke20/20260424T043649Z-dala_lite_split_context_v1-smoke20-dashscope-qwen-turbo-1101`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`98`，`round_budget_tokens`=`39`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`175`，`round_budget_tokens`=`70`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4250 | 0.00 | 1793.45 | 3.00 | 0.236973 |
| `all_to_all_full` | 0.5000 | 134.50 | 3882.00 | 6.00 | 0.128800 |
| `budget_random` | 0.4250 | 35.62 | 3623.93 | 6.00 | 0.117276 |
| `budget_confidence` | 0.4500 | 34.23 | 3623.65 | 6.00 | 0.124184 |
| `dala_lite` | 0.4500 | 25.98 | 3605.88 | 6.00 | 0.124796 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 4 | 1 |
| `budget_random` | 0.9500 | 0.6812 | 0.3167 | 0.0000 | 0.0000 | 0.6833 | 1 | 1 |
| `budget_confidence` | 0.9250 | 0.6581 | 0.3083 | 0.0000 | 0.0000 | 0.6917 | 2 | 1 |
| `dala_lite` | 0.7500 | 0.5034 | 0.2083 | 0.0000 | 0.0417 | 0.7500 | 2 | 1 |

## 5. 数据集分表

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.2500 | 0.00 | 2560.70 | 0.097630 |
| `all_to_all_full` | 0.3500 | 165.80 | 5469.10 | 0.063996 |
| `budget_random` | 0.2000 | 40.90 | 5141.05 | 0.038903 |
| `budget_confidence` | 0.2000 | 38.65 | 5142.25 | 0.038893 |
| `dala_lite` | 0.2000 | 28.65 | 5121.65 | 0.039050 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1026.20 | 0.584681 |
| `all_to_all_full` | 0.6500 | 103.20 | 2294.90 | 0.283237 |
| `budget_random` | 0.6500 | 30.35 | 2106.80 | 0.308525 |
| `budget_confidence` | 0.7000 | 29.80 | 2105.05 | 0.332534 |
| `dala_lite` | 0.7000 | 23.30 | 2090.10 | 0.334912 |

## 6. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a89b1de5542992e4fca8378`
- 问题预览：Which port city lies approximately 25 km north of the Lingnan Fine Arts Museum?
- 金标：`Keelung`
- `all_to_all_full`：`keelung` / score=`1.0`
- `dala_lite`：`unknown` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5ab82d095542990e739ec853`
- 问题预览："Tunak", is a bhangra/pop love song by an artist born in which year ?
- 金标：`1967`
- `all_to_all_full`：`1967` / score=`1.0`
- `dala_lite`：`1975` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5abca1a55542993a06baf937`
- 问题预览：When did the park at which Tivolis Koncertsal is located open?
- 金标：`15 August 1843`
- `all_to_all_full`：`15 august 1843` / score=`1.0`
- `dala_lite`：`1956` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5ae6b6065542991bbc976168`
- 问题预览：Out of the actors who have played the role of Luc Deveraux in the Universal Soldier franchise, which actor has also s...
- 金标：`Scott Adkins`
- `all_to_all_full`：`scott adkins` / score=`1.0`
- `dala_lite`：`jeanclaude van damme` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.175000, 0.075000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.05`
- `communication_ratio_vs_all_to_all_full`：`0.193123`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

