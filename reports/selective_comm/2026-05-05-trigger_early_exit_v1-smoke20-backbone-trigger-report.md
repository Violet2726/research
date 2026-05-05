# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_early_exit_v1`
- Phase：`smoke20`
- Backbone：`None`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_v1/smoke20/20260505T050233Z-trigger_early_exit_v1-smoke20-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`54359.00`；若按 4 套 trigger 独立重跑则为 `118460.00`；共享前缀节省比例=`0.5411`
- `hotpotqa`：共享实际 token=`219980.00`；若按 4 套 trigger 独立重跑则为 `594758.00`；共享前缀节省比例=`0.6301`
- `overall`：共享实际 token=`318469.00`；若按 4 套 trigger 独立重跑则为 `817218.00`；共享前缀节省比例=`0.6103`
- `strategyqa`：共享实际 token=`44130.00`；若按 4 套 trigger 独立重跑则为 `104000.00`；共享前缀节省比例=`0.5757`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 2313.52 | 0.345794 |
| `always` | 0.8333 | 2994.30 | 5307.82 | 0.157001 |
| `disagreement` | 0.8167 | 685.97 | 2999.48 | 0.272269 |
| `confidence` | 0.8000 | 0.00 | 2313.52 | 0.345794 |
| `hybrid` | 0.8167 | 685.97 | 2999.48 | 0.272269 |
| `mv_6` | 0.8333 | 0.00 | 4663.95 | 0.178675 |
| `sc_6` | 0.8667 | 0.00 | 4648.68 | 0.186433 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 | 0.9500 | 0.0000 | 2994.30 | 5307.82 | 0.157001 |
| `disagreement` | 0.1667 | 0.8333 | 0.2000 | 0.6667 | 0.1333 | 0.3333 | 685.97 | 2999.48 | 0.272269 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 2313.52 | 0.345794 |
| `hybrid` | 0.1667 | 0.8333 | 0.2000 | 0.6667 | 0.1333 | 0.3333 | 685.97 | 2999.48 | 0.272269 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.9333 | 1.0000 | 2994.30 | 5307.82 |
| `disagreement` | 0.6667 | 1.0000 | 0.7000 | 0.1667 | 685.97 | 2999.48 |
| `confidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 2313.52 |
| `hybrid` | 0.6667 | 1.0000 | 0.7000 | 0.1667 | 685.97 | 2999.48 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.9500 | 0.00 | 996.55 | 0.953289 |
| `always` | 0.9500 | 1721.40 | 2717.95 | 0.349528 |
| `disagreement` | 0.9500 | 107.70 | 1104.25 | 0.860312 |
| `confidence` | 0.9500 | 0.00 | 996.55 | 0.953289 |
| `hybrid` | 0.9500 | 107.70 | 1104.25 | 0.860312 |
| `mv_6` | 0.9500 | 0.00 | 1977.65 | 0.480368 |
| `sc_6` | 1.0000 | 0.00 | 1981.15 | 0.504757 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5137.00 | 0.146000 |
| `always` | 0.8000 | 5862.00 | 10999.00 | 0.072734 |
| `disagreement` | 0.7500 | 1663.95 | 6800.95 | 0.110279 |
| `confidence` | 0.7500 | 0.00 | 5137.00 | 0.146000 |
| `hybrid` | 0.7500 | 1663.95 | 6800.95 | 0.110279 |
| `mv_6` | 0.8000 | 0.00 | 10360.10 | 0.077219 |
| `sc_6` | 0.8000 | 0.00 | 10320.75 | 0.077514 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.2500 | 0.7500 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.2500 | 0.7500 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 807.00 | 0.867410 |
| `always` | 0.7500 | 1399.50 | 2206.50 | 0.339905 |
| `disagreement` | 0.7500 | 286.25 | 1093.25 | 0.686028 |
| `confidence` | 0.7000 | 0.00 | 807.00 | 0.867410 |
| `hybrid` | 0.7500 | 286.25 | 1093.25 | 0.686028 |
| `mv_6` | 0.7500 | 0.00 | 1654.10 | 0.453419 |
| `sc_6` | 0.8000 | 0.00 | 1644.15 | 0.486574 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1000 | 1.0000 |
| `disagreement` | 0.2000 | 0.8000 | 0.5000 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.2000 | 0.8000 | 0.5000 | 1.0000 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00672`
- 问题预览：Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...
- 金标：`2`
- `mv_3`：`0` / score=`0.0`
- `always`：`14` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`hotpotqa`
- 样本：`5a7a567255429941d65f25bd`
- 问题预览：What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okham...
- 金标：`flotilla`
- `mv_3`：`flotilla` / score=`1.0`
- `always`：`flotilla` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a80cf4c55429938b61421f6`
- 问题预览：What was the concept of the business Eric S .Pistorius worked for after being an attorney?
- 金标：`to ensure wide visibility and understanding of cases in a region`
- `mv_3`：`not mentioned in context` / score=`0.0`
- `always`：`law firm` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5a89b1de5542992e4fca8378`
- 问题预览：Which port city lies approximately 25 km north of the Lingnan Fine Arts Museum?
- 金标：`Keelung`
- `mv_3`：`request was rejected because it was considered high risk` / score=`0.0`
- `always`：`keelung` / score=`1.0`
- 说明：always_communicate 能纠错，但 hybrid_trigger 在该题 early exit，漏掉了有益通信。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5ab514c05542991779162d72`
- 问题预览：The school in which the Wilmslow Show is held is designated as what?
- 金标：`Centre of Excellence`
- `mv_3`：`designated centre of excellence` / score=`0.0`
- `always`：`designated centre of excellence` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`hybrid_trigger`
- 相对 `always_communicate` 的准确率下降：`0.016666`
- 相对 `always_communicate` 的总 token 下降比例：`0.434893`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

