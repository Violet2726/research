# DALA-lite Smoke20 报告

## 1. 实验概览

- 实验名：`dala_lite_split_context_v1`
- 轨道：`split_context`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/budget_comm/dala_lite_split_context_v1/pilot100/20260506T140706Z-dala_lite_split_context_v1-pilot100-xiaomimimo-mimo-v2.5`
- 方法固定为：`mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence`、`dala_lite`。

## 2. 预算冻结

- `strategyqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`209`，`round_budget_tokens`=`83`。
- `hotpotqa`：校准样本数=`5`，`p50(all_to_all_full_comm_tokens)`=`291`，`round_budget_tokens`=`116`。

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6100 | 0.00 | 1972.35 | 3.00 | 0.309277 |
| `all_to_all_full` | 0.7500 | 269.14 | 4524.57 | 6.00 | 0.165762 |
| `budget_random` | 0.6450 | 73.64 | 4086.80 | 6.00 | 0.157825 |
| `budget_confidence` | 0.6800 | 70.07 | 4080.07 | 6.00 | 0.166664 |
| `dala_lite` | 0.6950 | 57.01 | 4055.80 | 6.00 | 0.171360 |

## 4. 机制表

| Method | Winner Set Size | Budget Utilization | Full Ratio | Summary Ratio | Keywords Ratio | Silence Ratio | Corrected | Harmed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | - | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0 | 0 |
| `all_to_all_full` | 3.0000 | - | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 30 | 2 |
| `budget_random` | 0.9850 | 0.7424 | 0.3283 | 0.0000 | 0.0000 | 0.6717 | 11 | 4 |
| `budget_confidence` | 0.9850 | 0.7055 | 0.3283 | 0.0000 | 0.0000 | 0.6717 | 21 | 7 |
| `dala_lite` | 0.8950 | 0.5775 | 0.2433 | 0.0000 | 0.0550 | 0.7017 | 23 | 6 |

## 5. 数据集分表

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4100 | 0.00 | 2756.11 | 0.148760 |
| `all_to_all_full` | 0.6600 | 339.57 | 6232.58 | 0.105895 |
| `budget_random` | 0.4500 | 84.52 | 5682.09 | 0.079196 |
| `budget_confidence` | 0.5200 | 80.97 | 5675.57 | 0.091621 |
| `dala_lite` | 0.5200 | 63.77 | 5647.24 | 0.092080 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8100 | 0.00 | 1188.58 | 0.681485 |
| `all_to_all_full` | 0.8400 | 198.71 | 2816.56 | 0.298236 |
| `budget_random` | 0.8400 | 62.77 | 2491.52 | 0.337144 |
| `budget_confidence` | 0.8400 | 59.17 | 2484.58 | 0.338085 |
| `dala_lite` | 0.8700 | 50.24 | 2464.35 | 0.353034 |

## 6. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a738fe855429908901be2fb`
- 问题预览：What film was written and directed by Joby Harold with music written by Samuel Sim?
- 金标：`Awake`
- `all_to_all_full`：`awake` / score=`1.0`
- `dala_lite`：`robin hood` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5a76394c5542994ccc918725`
- 问题预览：When was the band who composited "Discipline" formed?
- 金标：`1968`
- `all_to_all_full`：`1968` / score=`1.0`
- `dala_lite`：`1987` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a77897f55429949eeb29edc`
- 问题预览：Jason Regler, stated that he had the idea for the flashing wristbands during a song built around which instrument ?
- 金标：`an organ`
- `all_to_all_full`：`organ` / score=`1.0`
- `dala_lite`：`fix you` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5a7f244255429934daa2fcec`
- 问题预览：St. John's College, Belize offers an education in a tradition in which what three subjects were the core?
- 金标：`Grammar, logic, and rhetoric`
- `all_to_all_full`：`grammar logic and rhetoric` / score=`1.0`
- `dala_lite`：`liberal arts and science` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5a89b1de5542992e4fca8378`
- 问题预览：Which port city lies approximately 25 km north of the Lingnan Fine Arts Museum?
- 金标：`Keelung`
- `all_to_all_full`：`keelung` / score=`1.0`
- `dala_lite`：`sokółka` / score=`0.0`
- 说明：dala_lite 在该题上弱于 all_to_all_full。

## 7. 探索性区间

- `dala_lite` 相对 `all_to_all_full` 的 overall accuracy delta 95% bootstrap CI：[-0.100000, -0.010000]（探索性）

## 8. Full DALA 进入门槛

- 是否满足进入条件：`False`
- 原因：`gate_not_met`
- `dala_accuracy_gap_vs_all_to_all_full_le_3pp`：`False`
- `dala_beats_budget_confidence_on_acc_per_1k`：`True`
- `dala_beats_budget_random_on_acc_per_1k`：`True`
- `dala_communication_le_60pct_of_all_to_all_full`：`True`
- `accuracy_gap_vs_all_to_all_full`：`0.055`
- `communication_ratio_vs_all_to_all_full`：`0.211804`

## 9. 局限

- 当前只覆盖 smoke20，小样本结果仅用于机制验证与工程联调。
- split-context 轨道与 same-context 轨道分开报告，不直接合并成统一 overall 结论。
- 当前只实现 DALA-lite 的无训练近似，不包含 MAPPO/value network 学习版。

