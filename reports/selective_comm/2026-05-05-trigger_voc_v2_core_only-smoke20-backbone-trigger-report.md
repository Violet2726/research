# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_core_only`
- Phase：`smoke20`
- Backbone：`None`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/trigger_voc_v2_core_only/smoke20/20260505T053922Z-trigger_voc_v2_core_only-smoke20-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`56393.00`；若按 4 套 trigger 独立重跑则为 `145503.00`；共享前缀节省比例=`0.6124`
- `hotpotqa`：共享实际 token=`133890.00`；若按 4 套 trigger 独立重跑则为 `365094.00`；共享前缀节省比例=`0.6333`
- `overall`：共享实际 token=`235426.00`；若按 4 套 trigger 独立重跑则为 `624253.00`；共享前缀节省比例=`0.6229`
- `strategyqa`：共享实际 token=`45143.00`；若按 4 套 trigger 独立重跑则为 `113656.00`；共享前缀节省比例=`0.6028`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6333 | 0.00 | 2478.43 | 0.255538 |
| `always` | 0.8000 | 1445.33 | 3923.77 | 0.203886 |
| `disagreement` | 0.8000 | 685.20 | 3163.63 | 0.252874 |
| `voc_v2` | 0.8000 | 838.38 | 3316.82 | 0.241195 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1833 | 1.0000 | 0.8167 | 0.0000 | 1445.33 | 3923.77 | 0.203886 |
| `disagreement` | 0.4500 | 0.5500 | 0.4074 | 1.0000 | 0.2667 | 0.0000 | 685.20 | 3163.63 | 0.252874 |
| `voc_v2` | 0.5667 | 0.4333 | 0.3235 | 1.0000 | 0.3833 | 0.0000 | 838.38 | 3316.82 | 0.241195 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.8000 | 1.0000 | 1445.33 | 3923.77 |
| `disagreement` | 1.0000 | 1.0000 | 0.5556 | 0.4500 | 685.20 | 3163.63 |
| `voc_v2` | 1.0000 | 1.0000 | 0.6471 | 0.5667 | 838.38 | 3316.82 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4500 | 0.00 | 1202.45 | 0.374236 |
| `always` | 0.8500 | 1617.20 | 2819.65 | 0.301456 |
| `disagreement` | 0.8500 | 1025.30 | 2227.75 | 0.381551 |
| `voc_v2` | 0.8500 | 1025.30 | 2227.75 | 0.381551 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.4000 | 1.0000 |
| `disagreement` | 0.6000 | 0.4000 | 0.6667 | 1.0000 |
| `voc_v2` | 0.6000 | 0.4000 | 0.6667 | 1.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 5271.45 | 0.151761 |
| `always` | 0.7500 | 1423.05 | 6694.50 | 0.112032 |
| `disagreement` | 0.7500 | 438.20 | 5709.65 | 0.131357 |
| `voc_v2` | 0.7500 | 579.10 | 5850.55 | 0.128193 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| `voc_v2` | 0.4000 | 0.6000 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 961.40 | 0.676097 |
| `always` | 0.8000 | 1295.75 | 2257.15 | 0.354429 |
| `disagreement` | 0.8000 | 592.10 | 1553.50 | 0.514966 |
| `voc_v2` | 0.8000 | 910.75 | 1872.15 | 0.427316 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1500 | 1.0000 |
| `disagreement` | 0.4500 | 0.5500 | 0.3333 | 1.0000 |
| `voc_v2` | 0.7000 | 0.3000 | 0.2143 | 1.0000 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00222`
- 问题预览：Andy plants 90 geraniums and 40 fewer petunias that geraniums. How many flowers does he plant total?
- 金标：`140`
- `mv_3`：`130` / score=`0.0`
- `always`：`130` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00599`
- 问题预览：A pet store currently has 5 dogs, 2 cats, and 10 birds. How many legs in total do the pets in the store have?
- 金标：`48`
- `mv_3`：`44` / score=`0.0`
- `always`：`44` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00906`
- 问题预览：Tim has a box with 7 blue shoe boxes and 9 red shoe boxes. If he uses 3 blue shoeboxes and 1/3 red of his shoeboxes t...
- 金标：`10`
- `mv_3`：`10` / score=`1.0`
- `always`：`10` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00945`
- 问题预览：James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...
- 金标：`17`
- `mv_3`：`14` / score=`0.0`
- `always`：`14` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5a80cf4c55429938b61421f6`
- 问题预览：What was the concept of the business Eric S .Pistorius worked for after being an attorney?
- 金标：`to ensure wide visibility and understanding of cases in a region`
- `mv_3`：`law firm` / score=`0.0`
- `always`：`law firm` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.154686`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

