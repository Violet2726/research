# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_core_only`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/trigger_voc_v2_core_only/smoke20/20260505T102409Z-trigger_voc_v2_core_only-smoke20-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`55323.00`；若按 4 套 trigger 独立重跑则为 `136417.00`；共享前缀节省比例=`0.5945`
- `hotpotqa`：共享实际 token=`135791.00`；若按 4 套 trigger 独立重跑则为 `369159.00`；共享前缀节省比例=`0.6322`
- `overall`：共享实际 token=`236877.00`；若按 4 套 trigger 独立重跑则为 `617088.00`；共享前缀节省比例=`0.6161`
- `strategyqa`：共享实际 token=`45763.00`；若按 4 套 trigger 独立重跑则为 `111512.00`；共享前缀节省比例=`0.5896`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6333 | 0.00 | 2488.22 | 0.254533 |
| `always` | 0.7167 | 1459.73 | 3947.95 | 0.181529 |
| `disagreement` | 0.7167 | 585.63 | 3073.85 | 0.233150 |
| `voc_v2` | 0.7167 | 774.78 | 3263.00 | 0.219634 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0833 | 1.0000 | 0.9167 | 0.0000 | 1459.73 | 3947.95 | 0.181529 |
| `disagreement` | 0.3833 | 0.6167 | 0.2174 | 1.0000 | 0.3000 | 0.0000 | 585.63 | 3073.85 | 0.233150 |
| `voc_v2` | 0.5167 | 0.4833 | 0.1613 | 1.0000 | 0.4333 | 0.0000 | 774.78 | 3263.00 | 0.219634 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.9167 | 1.0000 | 1459.73 | 3947.95 |
| `disagreement` | 1.0000 | 0.0000 | 0.7826 | 0.3833 | 585.63 | 3073.85 |
| `voc_v2` | 1.0000 | 0.0000 | 0.8387 | 0.5167 | 774.78 | 3263.00 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5000 | 0.00 | 1174.20 | 0.425822 |
| `always` | 0.7500 | 1591.95 | 2766.15 | 0.271135 |
| `disagreement` | 0.7500 | 853.15 | 2027.35 | 0.369941 |
| `voc_v2` | 0.7500 | 853.15 | 2027.35 | 0.369941 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.2500 | 1.0000 |
| `disagreement` | 0.5000 | 0.5000 | 0.5000 | 1.0000 |
| `voc_v2` | 0.5000 | 0.5000 | 0.5000 | 1.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 5317.30 | 0.131646 |
| `always` | 0.7000 | 1472.25 | 6789.55 | 0.103100 |
| `disagreement` | 0.7000 | 371.45 | 5688.75 | 0.123050 |
| `voc_v2` | 0.7000 | 662.35 | 5979.65 | 0.117064 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.2500 | 0.7500 | 0.0000 | 0.0000 |
| `voc_v2` | 0.4500 | 0.5500 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 973.15 | 0.719314 |
| `always` | 0.7000 | 1315.00 | 2288.15 | 0.305924 |
| `disagreement` | 0.7000 | 532.30 | 1505.45 | 0.464977 |
| `voc_v2` | 0.7000 | 808.85 | 1782.00 | 0.392817 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.4000 | 0.6000 | 0.0000 | 0.0000 |
| `voc_v2` | 0.6000 | 0.4000 | 0.0000 | 0.0000 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00057`
- 问题预览：A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 po...
- 金标：`83`
- `mv_3`：`83` / score=`1.0`
- `always`：`83` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00588`
- 问题预览：Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next t...
- 金标：`52`
- `mv_3`：`51` / score=`0.0`
- `always`：`51` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00906`
- 问题预览：Tim has a box with 7 blue shoe boxes and 9 red shoe boxes. If he uses 3 blue shoeboxes and 1/3 red of his shoeboxes t...
- 金标：`10`
- `mv_3`：`6` / score=`0.0`
- `always`：`12` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00945`
- 问题预览：James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...
- 金标：`17`
- `mv_3`：`19` / score=`0.0`
- `always`：`19` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-01132`
- 问题预览：Valerie earns $5000 per month, 1/2 of what her brother earns. If their mother earns twice their combined salary, what...
- 金标：`45000`
- `mv_3`：`35000` / score=`0.0`
- `always`：`35000` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.173495`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

