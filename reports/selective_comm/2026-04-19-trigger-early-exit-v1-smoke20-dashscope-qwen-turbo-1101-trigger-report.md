# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger-early-exit-v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- Prompt Version：`v1-trigger-json-short-reasoning`
- 运行目录：`runs/selective_comm/trigger-early-exit-v1/smoke20/20260419T081522Z-trigger-early-exit-v1-smoke20-dashscope-qwen-turbo-1101`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容消融和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`52102.00`，若按 4 套 trigger 独立重跑则为 `131024.00`，共享前缀节省比例=`0.6023`
- `hotpotqa`：共享实际 token=`214558.00`，若按 4 套 trigger 独立重跑则为 `569517.00`，共享前缀节省比例=`0.6233`
- `overall`：共享实际 token=`306791.00`，若按 4 套 trigger 独立重跑则为 `800292.00`，共享前缀节省比例=`0.6167`
- `strategyqa`：共享实际 token=`40131.00`，若按 4 套 trigger 独立重跑则为 `99751.00`，共享前缀节省比例=`0.5977`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7167 | 0.00 | 2286.92 | 0.313377 |
| `always` | 0.7333 | 2826.27 | 5113.18 | 0.143420 |
| `disagreement` | 0.7333 | 649.35 | 2936.27 | 0.249750 |
| `confidence` | 0.7167 | 45.20 | 2332.12 | 0.307303 |
| `hybrid` | 0.7333 | 669.72 | 2956.63 | 0.248030 |
| `mv_6` | 0.7000 | 0.00 | 4577.90 | 0.152909 |
| `sc_6` | 0.6833 | 0.00 | 4572.98 | 0.149428 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0333 | 1.0000 | 0.9667 | 0.0000 | 2826.27 | 5113.18 | 0.143420 |
| `disagreement` | 0.2167 | 0.7833 | 0.1538 | 1.0000 | 0.1833 | 0.0000 | 649.35 | 2936.27 | 0.249750 |
| `confidence` | 0.0333 | 0.9667 | 0.0000 | 0.0000 | 0.0333 | 1.0000 | 45.20 | 2332.12 | 0.307303 |
| `hybrid` | 0.2333 | 0.7667 | 0.1429 | 1.0000 | 0.2000 | 0.0000 | 669.72 | 2956.63 | 0.248030 |

## 5. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8500 | 0.00 | 971.00 | 0.875386 |
| `always` | 0.9000 | 1634.10 | 2605.10 | 0.345476 |
| `disagreement` | 0.9000 | 516.55 | 1487.55 | 0.605022 |
| `confidence` | 0.8500 | 0.00 | 971.00 | 0.875386 |
| `hybrid` | 0.9000 | 516.55 | 1487.55 | 0.605022 |
| `mv_6` | 0.8500 | 0.00 | 1943.60 | 0.437333 |
| `sc_6` | 0.8000 | 0.00 | 1943.60 | 0.411607 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.1667 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.1667 | 1.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5104.65 | 0.146925 |
| `always` | 0.7500 | 5623.25 | 10727.90 | 0.069911 |
| `disagreement` | 0.7500 | 1217.00 | 6321.65 | 0.118640 |
| `confidence` | 0.7500 | 0.00 | 5104.65 | 0.146925 |
| `hybrid` | 0.7500 | 1217.00 | 6321.65 | 0.118640 |
| `mv_6` | 0.7000 | 0.00 | 10227.15 | 0.068445 |
| `sc_6` | 0.7000 | 0.00 | 10215.75 | 0.068522 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.2000 | 0.8000 | 0.2500 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.2000 | 0.8000 | 0.2500 | 1.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 785.10 | 0.700548 |
| `always` | 0.5500 | 1221.45 | 2006.55 | 0.274102 |
| `disagreement` | 0.5500 | 214.50 | 999.60 | 0.550220 |
| `confidence` | 0.5500 | 135.60 | 920.70 | 0.597372 |
| `hybrid` | 0.5500 | 275.60 | 1060.70 | 0.518526 |
| `mv_6` | 0.5500 | 0.00 | 1562.95 | 0.351899 |
| `sc_6` | 0.5500 | 0.00 | 1559.60 | 0.352655 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.1500 | 0.8500 | 0.0000 | 0.0000 |
| `confidence` | 0.1000 | 0.9000 | 0.0000 | 0.0000 |
| `hybrid` | 0.2000 | 0.8000 | 0.0000 | 0.0000 |

## 6. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00599`
- 问题预览：A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?
- 金标：`48`
- `mv_3`：`46` / score=`0.0`
- `always`：`46` / score=`0.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00616`
- 问题预览：Jane planted a beanstalk in her backyard. After the first week, it was 3 inches tall. It doubled in height the second...
- 金标：`10`
- `mv_3`：`10` / score=`1.0`
- `always`：`10` / score=`1.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00618`
- 问题预览：The school auditorium has 4 rows of seats. There are 18 seats in each row. One-fourth of the seats were occupied by t...
- 金标：`36`
- `mv_3`：`36` / score=`1.0`
- `always`：`36` / score=`1.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00906`
- 问题预览：Tim has a box with 7 blue shoe boxes and 9 red shoe boxes. If he uses 3 blue shoeboxes and 1/3 red of his shoeboxes t...
- 金标：`10`
- `mv_3`：`10` / score=`1.0`
- `always`：`10` / score=`1.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00945`
- 问题预览：James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...
- 金标：`17`
- `mv_3`：`17` / score=`1.0`
- `always`：`17` / score=`1.0`
- 说明：通信本身没有带来收益，但 disagreement_triggered 仍然进入了通信。

## 7. 下一轮默认 trigger 建议

- 推荐策略：`hybrid_trigger`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.421763`
- 规则是否通过：`True`

## 8. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

