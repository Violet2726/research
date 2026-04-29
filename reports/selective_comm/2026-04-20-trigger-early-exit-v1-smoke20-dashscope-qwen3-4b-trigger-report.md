# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger-early-exit-v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen3-4b`
- Prompt Version：`v1-trigger-json-short-reasoning`
- 运行目录：`local/runs/selective_comm/trigger-early-exit-v1/smoke20/20260420T131621Z-trigger-early-exit-v1-smoke20-dashscope-qwen3-4b`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`54375.00`；若按 4 套 trigger 独立重跑则为 `120671.00`；共享前缀节省比例=`0.5494`
- `hotpotqa`：共享实际 token=`225899.00`；若按 4 套 trigger 独立重跑则为 `627278.00`；共享前缀节省比例=`0.6399`
- `overall`：共享实际 token=`321442.00`；若按 4 套 trigger 独立重跑则为 `844869.00`；共享前缀节省比例=`0.6195`
- `strategyqa`：共享实际 token=`41168.00`；若按 4 套 trigger 独立重跑则为 `96920.00`；共享前缀节省比例=`0.5752`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 2342.47 | 0.298830 |
| `always` | 0.7000 | 3014.90 | 5357.37 | 0.130661 |
| `disagreement` | 0.7000 | 705.97 | 3048.43 | 0.229626 |
| `confidence` | 0.7000 | 238.50 | 2580.97 | 0.271216 |
| `hybrid` | 0.7000 | 751.92 | 3094.38 | 0.226216 |
| `mv_6` | 0.6833 | 0.00 | 4683.07 | 0.145916 |
| `sc_6` | 0.6667 | 0.00 | 4680.60 | 0.142432 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0167 | 1.0000 | 0.9833 | 0.0000 | 3014.90 | 5357.37 | 0.130661 |
| `disagreement` | 0.1500 | 0.8500 | 0.1111 | 1.0000 | 0.1333 | 0.0000 | 705.97 | 3048.43 | 0.229626 |
| `confidence` | 0.0667 | 0.9333 | 0.0000 | 0.0000 | 0.0667 | 1.0000 | 238.50 | 2580.97 | 0.271216 |
| `hybrid` | 0.1833 | 0.8167 | 0.0909 | 1.0000 | 0.1667 | 0.0000 | 751.92 | 3094.38 | 0.226216 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.9500 | 0.00 | 984.60 | 0.964859 |
| `always` | 0.9500 | 1734.15 | 2718.75 | 0.349425 |
| `disagreement` | 0.9500 | 180.50 | 1165.10 | 0.815381 |
| `confidence` | 0.9500 | 0.00 | 984.60 | 0.964859 |
| `hybrid` | 0.9500 | 180.50 | 1165.10 | 0.815381 |
| `mv_6` | 0.9500 | 0.00 | 1975.60 | 0.480867 |
| `sc_6` | 0.9000 | 0.00 | 1973.70 | 0.455996 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.1000 | 0.9000 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.1000 | 0.9000 | 0.0000 | 0.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 5245.50 | 0.104852 |
| `always` | 0.6000 | 6049.45 | 11294.95 | 0.053121 |
| `disagreement` | 0.6000 | 1877.40 | 7122.90 | 0.084235 |
| `confidence` | 0.5500 | 577.65 | 5823.15 | 0.094451 |
| `hybrid` | 0.6000 | 1877.40 | 7122.90 | 0.084235 |
| `mv_6` | 0.5500 | 0.00 | 10485.75 | 0.052452 |
| `sc_6` | 0.5500 | 0.00 | 10481.50 | 0.052473 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.1667 | 1.0000 |
| `confidence` | 0.1000 | 0.9000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.1667 | 1.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 797.30 | 0.752540 |
| `always` | 0.5500 | 1261.10 | 2058.40 | 0.267198 |
| `disagreement` | 0.5500 | 60.00 | 857.30 | 0.641549 |
| `confidence` | 0.6000 | 137.85 | 935.15 | 0.641608 |
| `hybrid` | 0.5500 | 197.85 | 995.15 | 0.552681 |
| `mv_6` | 0.5500 | 0.00 | 1587.85 | 0.346380 |
| `sc_6` | 0.5500 | 0.00 | 1586.60 | 0.346653 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `confidence` | 0.1000 | 0.9000 | 0.0000 | 0.0000 |
| `hybrid` | 0.1500 | 0.8500 | 0.0000 | 0.0000 |

## 6. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00599`
- 问题预览：A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?
- 金标：`48`
- `mv_3`：`48` / score=`1.0`
- `always`：`48` / score=`1.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00672`
- 问题预览：Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...
- 金标：`2`
- `mv_3`：`10` / score=`0.0`
- `always`：`10` / score=`0.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a77897f55429949eeb29edc`
- 问题预览：Jason Regler, stated that he had the idea for the flashing wristbands during a song built around which instrument ?
- 金标：`an organ`
- `mv_3`：`fix you` / score=`0.0`
- `always`：`fix you` / score=`0.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5a7a567255429941d65f25bd`
- 问题预览：What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okham...
- 金标：`flotilla`
- `mv_3`：`attacked radar station in dwarka india` / score=`0.0`
- `always`：`attacked radar station in dwarka india` / score=`0.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5a80cf4c55429938b61421f6`
- 问题预览：What was the concept of the business Eric S .Pistorius worked for after being an attorney?
- 金标：`to ensure wide visibility and understanding of cases in a region`
- `mv_3`：`law firm` / score=`0.0`
- `always`：`circuit court` / score=`0.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

## 7. 下一轮默认 trigger 建议

- 推荐策略：`hybrid_trigger`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.422406`
- 规则是否通过：`True`

## 8. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

