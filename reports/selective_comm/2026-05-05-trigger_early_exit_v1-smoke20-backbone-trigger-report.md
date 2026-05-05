# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_early_exit_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_v1/smoke20/20260505T091439Z-trigger_early_exit_v1-smoke20-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`55326.00`；若按 4 套 trigger 独立重跑则为 `120546.00`；共享前缀节省比例=`0.5410`
- `hotpotqa`：共享实际 token=`223518.00`；若按 4 套 trigger 独立重跑则为 `615301.00`；共享前缀节省比例=`0.6367`
- `overall`：共享实际 token=`324714.00`；若按 4 套 trigger 独立重跑则为 `848336.00`；共享前缀节省比例=`0.6172`
- `strategyqa`：共享实际 token=`45870.00`；若按 4 套 trigger 独立重跑则为 `112489.00`；共享前缀节省比例=`0.5922`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8500 | 0.00 | 2355.18 | 0.360906 |
| `always` | 0.8833 | 3056.72 | 5411.90 | 0.163220 |
| `disagreement` | 0.8667 | 818.78 | 3173.97 | 0.273055 |
| `confidence` | 0.8500 | 23.92 | 2379.10 | 0.357278 |
| `hybrid` | 0.8667 | 818.78 | 3173.97 | 0.273055 |
| `mv_6` | 0.8833 | 0.00 | 4726.60 | 0.186885 |
| `sc_6` | 0.8667 | 0.00 | 4750.83 | 0.182424 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 | 0.9500 | 0.0000 | 3056.72 | 5411.90 | 0.163220 |
| `disagreement` | 0.2000 | 0.8000 | 0.1667 | 0.6667 | 0.1667 | 0.3333 | 818.78 | 3173.97 | 0.273055 |
| `confidence` | 0.0167 | 0.9833 | 0.0000 | 0.0000 | 0.0167 | 1.0000 | 23.92 | 2379.10 | 0.357278 |
| `hybrid` | 0.2000 | 0.8000 | 0.1667 | 0.6667 | 0.1667 | 0.3333 | 818.78 | 3173.97 | 0.273055 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 3056.72 | 5411.90 |
| `disagreement` | 0.6667 | 1.0000 | 0.7500 | 0.2000 | 818.78 | 3173.97 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0167 | 23.92 | 2379.10 |
| `hybrid` | 0.6667 | 1.0000 | 0.7500 | 0.2000 | 818.78 | 3173.97 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.9500 | 0.00 | 1016.90 | 0.934212 |
| `always` | 1.0000 | 1749.40 | 2766.30 | 0.361494 |
| `disagreement` | 1.0000 | 105.15 | 1122.05 | 0.891226 |
| `confidence` | 0.9500 | 0.00 | 1016.90 | 0.934212 |
| `hybrid` | 1.0000 | 105.15 | 1122.05 | 0.891226 |
| `mv_6` | 1.0000 | 0.00 | 2041.85 | 0.489752 |
| `sc_6` | 1.0000 | 0.00 | 2048.00 | 0.488281 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.0500 | 0.9500 | 1.0000 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.0500 | 0.9500 | 1.0000 | 1.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 5209.05 | 0.153579 |
| `always` | 0.8000 | 5966.85 | 11175.90 | 0.071583 |
| `disagreement` | 0.7500 | 1981.00 | 7190.05 | 0.104311 |
| `confidence` | 0.8000 | 0.00 | 5209.05 | 0.153579 |
| `hybrid` | 0.7500 | 1981.00 | 7190.05 | 0.104311 |
| `mv_6` | 0.9000 | 0.00 | 10460.45 | 0.086038 |
| `sc_6` | 0.8000 | 0.00 | 10494.65 | 0.076229 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 839.60 | 0.952835 |
| `always` | 0.8500 | 1453.90 | 2293.50 | 0.370613 |
| `disagreement` | 0.8500 | 370.20 | 1209.80 | 0.702595 |
| `confidence` | 0.8000 | 71.75 | 911.35 | 0.877819 |
| `hybrid` | 0.8500 | 370.20 | 1209.80 | 0.702595 |
| `mv_6` | 0.7500 | 0.00 | 1677.50 | 0.447094 |
| `sc_6` | 0.8000 | 0.00 | 1709.85 | 0.467877 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.2500 | 0.7500 | 0.2000 | 1.0000 |
| `confidence` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `hybrid` | 0.2500 | 0.7500 | 0.2000 | 1.0000 |

## 7. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a7a567255429941d65f25bd`
- 问题预览：What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okham...
- 金标：`flotilla`
- `mv_3`：`flotilla` / score=`1.0`
- `always`：`flotilla` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5a80cf4c55429938b61421f6`
- 问题预览：What was the concept of the business Eric S .Pistorius worked for after being an attorney?
- 金标：`to ensure wide visibility and understanding of cases in a region`
- `mv_3`：`his law firm` / score=`0.0`
- `always`：`his law firm` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a89b1de5542992e4fca8378`
- 问题预览：Which port city lies approximately 25 km north of the Lingnan Fine Arts Museum?
- 金标：`Keelung`
- `mv_3`：`request was rejected because it was considered high risk` / score=`0.0`
- `always`：`keelung` / score=`1.0`
- 说明：always_communicate 能纠错，但 hybrid_trigger 在该题 early exit，漏掉了有益通信。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5ab514c05542991779162d72`
- 问题预览：The school in which the Wilmslow Show is held is designated as what?
- 金标：`Centre of Excellence`
- `mv_3`：`designated centre of excellence` / score=`0.0`
- `always`：`designated centre of excellence` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5ae2f5b955429928c423957e`
- 问题预览：What language, traditionally written with the ancient Libyco-Berber script, is closely related to the Tumzabt and Teg...
- 金标：`The Tugurt language`
- `mv_3`：`tugurt language` / score=`1.0`
- `always`：`tugurt language` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`hybrid_trigger`
- 相对 `always_communicate` 的准确率下降：`0.016666`
- 相对 `always_communicate` 的总 token 下降比例：`0.413521`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

