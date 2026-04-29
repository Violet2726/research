# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`local/runs/selective_comm/trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy/pilot100/20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`290458.00`；若按 4 套 trigger 独立重跑则为 `728918.00`；共享前缀节省比例=`0.6015`
- `hotpotqa`：共享实际 token=`663028.00`；若按 4 套 trigger 独立重跑则为 `1781918.00`；共享前缀节省比例=`0.6279`
- `overall`：共享实际 token=`1178956.00`；若按 4 套 trigger 独立重跑则为 `3037967.00`；共享前缀节省比例=`0.6119`
- `strategyqa`：共享实际 token=`225470.00`；若按 4 套 trigger 独立重跑则为 `527131.00`；共享前缀节省比例=`0.5723`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6300 | 0.00 | 2468.65 | 0.255200 |
| `always` | 0.7500 | 1461.20 | 3929.85 | 0.190847 |
| `disagreement` | 0.7233 | 508.37 | 2977.02 | 0.242972 |
| `voc_v2` | 0.7333 | 751.03 | 3219.68 | 0.227766 |
| `mv_6` | 0.6400 | 0.00 | 4941.03 | 0.129528 |
| `sc_6` | 0.6467 | 0.00 | 4938.75 | 0.130937 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1267 | 1.0000 | 0.8733 | 0.0000 | 1461.20 | 3929.85 | 0.190847 |
| `disagreement` | 0.3233 | 0.6767 | 0.3093 | 0.7895 | 0.2233 | 0.2105 | 508.37 | 2977.02 | 0.242972 |
| `voc_v2` | 0.5000 | 0.5000 | 0.2200 | 0.8684 | 0.3900 | 0.1316 | 751.03 | 3219.68 | 0.227766 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.8667 | 1.0000 | 1461.20 | 3929.85 |
| `disagreement` | 0.7895 | 1.0000 | 0.6701 | 0.3233 | 508.37 | 2977.02 |
| `voc_v2` | 0.8684 | 1.0000 | 0.7667 | 0.5000 | 751.03 | 3219.68 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4800 | 0.00 | 1226.50 | 0.391358 |
| `always` | 0.7800 | 1678.08 | 2904.58 | 0.268541 |
| `disagreement` | 0.7300 | 930.17 | 2156.67 | 0.338485 |
| `voc_v2` | 0.7600 | 1001.43 | 2227.93 | 0.341124 |
| `mv_6` | 0.4900 | 0.00 | 2456.54 | 0.199468 |
| `sc_6` | 0.4700 | 0.00 | 2452.09 | 0.191673 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.3100 | 1.0000 |
| `disagreement` | 0.5300 | 0.4700 | 0.4906 | 0.8387 |
| `voc_v2` | 0.5700 | 0.4300 | 0.5088 | 0.9355 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 5219.47 | 0.134113 |
| `always` | 0.7000 | 1410.81 | 6630.28 | 0.105576 |
| `disagreement` | 0.6900 | 282.98 | 5502.45 | 0.125399 |
| `voc_v2` | 0.6900 | 466.98 | 5686.45 | 0.121341 |
| `mv_6` | 0.7000 | 0.00 | 10442.58 | 0.067033 |
| `sc_6` | 0.7100 | 0.00 | 10442.17 | 0.067994 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0100 | 1.0000 |
| `disagreement` | 0.2000 | 0.8000 | 0.0000 | 0.0000 |
| `voc_v2` | 0.3300 | 0.6700 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7100 | 0.00 | 959.99 | 0.739591 |
| `always` | 0.7700 | 1294.71 | 2254.70 | 0.341509 |
| `disagreement` | 0.7500 | 311.96 | 1271.95 | 0.589646 |
| `voc_v2` | 0.7500 | 784.67 | 1744.66 | 0.429883 |
| `mv_6` | 0.7300 | 0.00 | 1923.98 | 0.379422 |
| `sc_6` | 0.7600 | 0.00 | 1921.99 | 0.395423 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0600 | 1.0000 |
| `disagreement` | 0.2400 | 0.7600 | 0.1667 | 0.6667 |
| `voc_v2` | 0.6000 | 0.4000 | 0.0667 | 0.6667 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00002`
- 问题预览：Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased...
- 金标：`70000`
- `mv_3`：`90000` / score=`0.0`
- `always`：`90000` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00100`
- 问题预览：Jerome had 4 friends who came to visit him on a certain day. The first friend pressed on the doorbell 20 times before...
- 金标：`175`
- `mv_3`：`115` / score=`0.0`
- `always`：`115` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00143`
- 问题预览：John buys milk for 2 dollars, eggs for 3 dollars, light bulbs for 3 dollars, cups for 3 dollars, and roach traps for ...
- 金标：`16`
- `mv_3`：`16.5` / score=`0.0`
- `always`：`16.5` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00177`
- 问题预览：Zaid spends 1/4 of his salary on rent, 1/3 on car fuel and donates half of the remaining amount to his favorite chari...
- 金标：`350`
- `mv_3`：`1500` / score=`0.0`
- `always`：`1550` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00187`
- 问题预览：Mandy owes Benedict $100. They agreed to have monthly interest of 2%. If Mandy was able to pay it after 3 months, how...
- 金标：`106`
- `mv_3`：`106.12` / score=`0.0`
- `always`：`106.12` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`0.016667`
- 相对 `always_communicate` 的总 token 下降比例：`0.180712`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

