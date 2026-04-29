# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_qwen_math_turbo_lite`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-math-turbo-latest`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`local/runs/selective_comm/trigger_voc_v2_qwen_math_turbo_lite/smoke20/20260428T064540Z-trigger_voc_v2_qwen_math_turbo_lite-smoke20-dashscope-qwen-math-turbo-latest`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`69338.00`；若按 4 套 trigger 独立重跑则为 `146078.00`；共享前缀节省比例=`0.5253`
- `hotpotqa`：共享实际 token=`146210.00`；若按 4 套 trigger 独立重跑则为 `378195.00`；共享前缀节省比例=`0.6134`
- `overall`：共享实际 token=`275817.00`；若按 4 套 trigger 独立重跑则为 `643840.00`；共享前缀节省比例=`0.5716`
- `strategyqa`：共享实际 token=`60269.00`；若按 4 套 trigger 独立重跑则为 `119567.00`；共享前缀节省比例=`0.4959`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.3500 | 0.00 | 2835.47 | 0.123436 |
| `always` | 0.3000 | 1761.48 | 4596.95 | 0.065261 |
| `disagreement` | 0.3833 | 213.97 | 3049.43 | 0.125706 |
| `voc_v2` | 0.3833 | 248.82 | 3084.28 | 0.124286 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1000 | 1.0000 | 0.9000 | 0.0000 | 1761.48 | 4596.95 | 0.065261 |
| `disagreement` | 0.1167 | 0.8833 | 0.2857 | 0.3333 | 0.0833 | 0.6667 | 213.97 | 3049.43 | 0.125706 |
| `voc_v2` | 0.1333 | 0.8667 | 0.2500 | 0.3333 | 0.1000 | 0.6667 | 248.82 | 3084.28 | 0.124286 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.7500 | 1.0000 | 1761.48 | 4596.95 |
| `disagreement` | 0.3333 | 0.0000 | 0.7143 | 0.1167 | 213.97 | 3049.43 |
| `voc_v2` | 0.3333 | 0.0000 | 0.7500 | 0.1333 | 248.82 | 3084.28 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4000 | 0.00 | 1644.55 | 0.243228 |
| `always` | 0.3000 | 1822.35 | 3466.90 | 0.086533 |
| `disagreement` | 0.5000 | 273.95 | 1918.50 | 0.260620 |
| `voc_v2` | 0.5000 | 273.95 | 1918.50 | 0.260620 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1500 | 1.0000 |
| `disagreement` | 0.1500 | 0.8500 | 0.6667 | 0.6667 |
| `voc_v2` | 0.1500 | 0.8500 | 0.6667 | 0.6667 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.0000 | 0.00 | 5455.55 | 0.000000 |
| `always` | 0.0000 | 1854.95 | 7310.50 | 0.000000 |
| `disagreement` | 0.0000 | 291.80 | 5747.35 | 0.000000 |
| `voc_v2` | 0.0000 | 396.35 | 5851.90 | 0.000000 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.1500 | 0.8500 | 0.0000 | 0.0000 |
| `voc_v2` | 0.2000 | 0.8000 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 1406.30 | 0.462206 |
| `always` | 0.6000 | 1607.15 | 3013.45 | 0.199107 |
| `disagreement` | 0.6500 | 76.15 | 1482.45 | 0.438463 |
| `voc_v2` | 0.6500 | 76.15 | 1482.45 | 0.438463 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1500 | 1.0000 |
| `disagreement` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `voc_v2` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |

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
- 相对 `always_communicate` 的准确率下降：`-0.083333`
- 相对 `always_communicate` 的总 token 下降比例：`0.329059`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

