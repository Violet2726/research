# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_qwen_math_turbo_lite`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-math-turbo`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`local/runs/selective_comm/trigger_voc_v2_qwen_math_turbo_lite/smoke20/20260428T065704Z-trigger_voc_v2_qwen_math_turbo_lite-smoke20-dashscope-qwen-math-turbo`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`69012.00`；若按 4 套 trigger 独立重跑则为 `145720.00`；共享前缀节省比例=`0.5264`
- `hotpotqa`：共享实际 token=`146014.00`；若按 4 套 trigger 独立重跑则为 `377997.00`；共享前缀节省比例=`0.6137`
- `overall`：共享实际 token=`275553.00`；若按 4 套 trigger 独立重跑则为 `645774.00`；共享前缀节省比例=`0.5733`
- `strategyqa`：共享实际 token=`60527.00`；若按 4 套 trigger 独立重跑则为 `122057.00`；共享前缀节省比例=`0.5041`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.3500 | 0.00 | 2836.93 | 0.123373 |
| `always` | 0.2833 | 1755.62 | 4592.55 | 0.061694 |
| `disagreement` | 0.3500 | 216.67 | 3053.60 | 0.114619 |
| `voc_v2` | 0.3667 | 279.82 | 3116.75 | 0.117644 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0833 | 1.0000 | 0.9167 | 0.0000 | 1755.62 | 4592.55 | 0.061694 |
| `disagreement` | 0.1167 | 0.8833 | 0.1429 | 0.2000 | 0.1000 | 0.8000 | 216.67 | 3053.60 | 0.114619 |
| `voc_v2` | 0.1500 | 0.8500 | 0.2222 | 0.4000 | 0.1167 | 0.6000 | 279.82 | 3116.75 | 0.117644 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.7667 | 1.0000 | 1755.62 | 4592.55 |
| `disagreement` | 0.2000 | 0.1111 | 0.7143 | 0.1167 | 216.67 | 3053.60 |
| `voc_v2` | 0.4000 | 0.1111 | 0.6667 | 0.1500 | 279.82 | 3116.75 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4500 | 0.00 | 1643.70 | 0.273773 |
| `always` | 0.2500 | 1806.90 | 3450.60 | 0.072451 |
| `disagreement` | 0.4500 | 274.00 | 1917.70 | 0.234656 |
| `voc_v2` | 0.4500 | 274.00 | 1917.70 | 0.234656 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1000 | 1.0000 |
| `disagreement` | 0.1500 | 0.8500 | 0.3333 | 0.5000 |
| `voc_v2` | 0.1500 | 0.8500 | 0.3333 | 0.5000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 5455.50 | 0.000000 |
| `always` | 0.0000 | 1845.20 | 7300.70 | 0.000000 |
| `disagreement` | 0.0000 | 291.80 | 5747.30 | 0.000000 |
| `voc_v2` | 0.0000 | 396.35 | 5851.85 | 0.000000 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.1500 | 0.8500 | 0.0000 | 0.0000 |
| `voc_v2` | 0.2000 | 0.8000 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1411.60 | 0.425050 |
| `always` | 0.6000 | 1614.75 | 3026.35 | 0.198259 |
| `disagreement` | 0.6000 | 84.20 | 1495.80 | 0.401123 |
| `voc_v2` | 0.6500 | 169.10 | 1580.70 | 0.411210 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1500 | 1.0000 |
| `disagreement` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `voc_v2` | 0.1000 | 0.9000 | 0.5000 | 0.3333 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00057`
- 问题预览：A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 po...
- 金标：`83`
- `mv_3`：`3` / score=`0.0`
- `always`：`5` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00588`
- 问题预览：Jared is trying to increase his typing speed. He starts with 47 words per minute (WPM). After some lessons the next t...
- 金标：`52`
- `mv_3`：`3` / score=`0.0`
- `always`：`52` / score=`1.0`
- 说明：always_communicate 能纠错，但 voc_trigger_v2 在该题 early exit，漏掉了有益通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00599`
- 问题预览：A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?
- 金标：`48`
- `mv_3`：`48` / score=`1.0`
- `always`：`1` / score=`0.0`
- 说明：always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00618`
- 问题预览：The school auditorium has 4 rows of seats. There are 18 seats in each row. One-fourth of the seats were occupied by t...
- 金标：`36`
- `mv_3`：`36` / score=`1.0`
- `always`：`54` / score=`0.0`
- 说明：always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00730`
- 问题预览：Parker chews 4 pieces of gum a day. A pack of gum has 15 pieces of chewing gum per pack. How many packs of gum will h...
- 金标：`8`
- `mv_3`：`8` / score=`1.0`
- `always`：`:` / score=`0.0`
- 说明：always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`-0.083334`
- 相对 `always_communicate` 的总 token 下降比例：`0.321347`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

