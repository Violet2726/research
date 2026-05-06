# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_early_exit_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_trigger_json`
- 运行目录：`runs/selective_comm/trigger_early_exit_v1/pilot100/20260506T140652Z-trigger_early_exit_v1-pilot100-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`294987.00`；若按 4 套 trigger 独立重跑则为 `641826.00`；共享前缀节省比例=`0.5404`
- `hotpotqa`：共享实际 token=`1105593.00`；若按 4 套 trigger 独立重跑则为 `2919774.00`；共享前缀节省比例=`0.6213`
- `overall`：共享实际 token=`1627300.00`；若按 4 套 trigger 独立重跑则为 `4126259.00`；共享前缀节省比例=`0.6056`
- `strategyqa`：共享实际 token=`226720.00`；若按 4 套 trigger 独立重跑则为 `564659.00`；共享前缀节省比例=`0.5985`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8433 | 0.00 | 2362.95 | 0.356898 |
| `always` | 0.8400 | 3061.38 | 5424.33 | 0.154858 |
| `disagreement` | 0.8400 | 574.60 | 2937.55 | 0.285952 |
| `confidence` | 0.8467 | 62.35 | 2425.31 | 0.349097 |
| `hybrid` | 0.8400 | 604.05 | 2967.00 | 0.283114 |
| `mv_6` | 0.8467 | 0.00 | 4731.19 | 0.178954 |
| `sc_6` | 0.8567 | 0.00 | 4734.91 | 0.180926 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0133 | 1.0000 | 0.9867 | 0.0000 | 3061.38 | 5424.33 | 0.154858 |
| `disagreement` | 0.1600 | 0.8400 | 0.0833 | 1.0000 | 0.1467 | 0.0000 | 574.60 | 2937.55 | 0.285952 |
| `confidence` | 0.0300 | 0.9700 | 0.1111 | 0.2500 | 0.0267 | 0.7500 | 62.35 | 2425.31 | 0.349097 |
| `hybrid` | 0.1800 | 0.8200 | 0.0741 | 1.0000 | 0.1667 | 0.0000 | 604.05 | 2967.00 | 0.283114 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.9700 | 1.0000 | 3061.38 | 5424.33 |
| `disagreement` | 1.0000 | 1.0000 | 0.8125 | 0.1600 | 574.60 | 2937.55 |
| `confidence` | 0.2500 | 0.0000 | 0.8889 | 0.0300 | 62.35 | 2425.31 |
| `hybrid` | 1.0000 | 1.0000 | 0.8333 | 0.1800 | 604.05 | 2967.00 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.9700 | 0.00 | 1070.57 | 0.906059 |
| `always` | 0.9600 | 1879.30 | 2949.87 | 0.325438 |
| `disagreement` | 0.9600 | 128.34 | 1198.91 | 0.800727 |
| `confidence` | 0.9700 | 0.00 | 1070.57 | 0.906059 |
| `hybrid` | 0.9600 | 128.34 | 1198.91 | 0.800727 |
| `mv_6` | 0.9600 | 0.00 | 2150.71 | 0.446364 |
| `sc_6` | 0.9600 | 0.00 | 2157.61 | 0.444937 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.0600 | 0.9400 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.0600 | 0.9400 | 0.0000 | 0.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5182.11 | 0.144729 |
| `always` | 0.7500 | 5873.82 | 11055.93 | 0.067837 |
| `disagreement` | 0.7500 | 1264.12 | 6446.23 | 0.116347 |
| `confidence` | 0.7500 | 67.24 | 5249.35 | 0.142875 |
| `hybrid` | 0.7500 | 1264.12 | 6446.23 | 0.116347 |
| `mv_6` | 0.7900 | 0.00 | 10372.73 | 0.076161 |
| `sc_6` | 0.7700 | 0.00 | 10371.70 | 0.074240 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0200 | 1.0000 |
| `disagreement` | 0.2100 | 0.7900 | 0.0952 | 1.0000 |
| `confidence` | 0.0100 | 0.9900 | 0.0000 | 0.0000 |
| `hybrid` | 0.2100 | 0.7900 | 0.0952 | 1.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8100 | 0.00 | 836.18 | 0.968691 |
| `always` | 0.8100 | 1431.02 | 2267.20 | 0.357269 |
| `disagreement` | 0.8100 | 331.34 | 1167.52 | 0.693778 |
| `confidence` | 0.8200 | 119.82 | 956.00 | 0.857741 |
| `hybrid` | 0.8100 | 419.69 | 1255.87 | 0.644971 |
| `mv_6` | 0.7900 | 0.00 | 1670.13 | 0.473017 |
| `sc_6` | 0.8400 | 0.00 | 1675.43 | 0.501364 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0200 | 1.0000 |
| `disagreement` | 0.2100 | 0.7900 | 0.0952 | 1.0000 |
| `confidence` | 0.0800 | 0.9200 | 0.1250 | 0.5000 |
| `hybrid` | 0.2700 | 0.7300 | 0.0741 | 1.0000 |

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
- 样本：`gsm8k-00187`
- 问题预览：Mandy owes Benedict $100. They agreed to have monthly interest of 2%. If Mandy was able to pay it after 3 months, how...
- 金标：`106`
- `mv_3`：`106.12` / score=`0.0`
- `always`：`106.12` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00371`
- 问题预览：A shoe store was having a weekend sale on a brand of popular tennis shoes. On Friday the store sold 14 pairs of tenni...
- 金标：`50`
- `mv_3`：`50` / score=`1.0`
- `always`：`100` / score=`0.0`
- 说明：always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00672`
- 问题预览：Christina records her mood every day on a calendar. Over the past thirty days of moods, she had twelve good days and ...
- 金标：`2`
- `mv_3`：`0` / score=`0.0`
- `always`：`0` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00811`
- 问题预览：Felix notices that kids in the neighborhood are always getting things stuck in trees. Since he is an expert tree clim...
- 金标：`60`
- `mv_3`：`60` / score=`1.0`
- `always`：`60` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`hybrid_trigger`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.45302`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

