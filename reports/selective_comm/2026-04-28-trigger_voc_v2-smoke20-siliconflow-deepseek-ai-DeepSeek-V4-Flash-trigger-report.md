# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2`
- Phase：`smoke20`
- Backbone：`siliconflow/deepseek-ai/DeepSeek-V4-Flash`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`local/runs/selective_comm/trigger_voc_v2/smoke20/20260428T161127Z-trigger_voc_v2-smoke20-siliconflow-deepseek-ai-DeepSeek-V4-Flash`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`53016.00`；若按 4 套 trigger 独立重跑则为 `274011.00`；共享前缀节省比例=`0.8065`
- `hotpotqa`：共享实际 token=`130487.00`；若按 4 套 trigger 独立重跑则为 `912545.00`；共享前缀节省比例=`0.8570`
- `overall`：共享实际 token=`229226.00`；若按 4 套 trigger 独立重跑则为 `1449621.00`；共享前缀节省比例=`0.8419`
- `strategyqa`：共享实际 token=`45723.00`；若按 4 套 trigger 独立重跑则为 `263065.00`；共享前缀节省比例=`0.8262`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7333 | 0.00 | 2434.57 | 0.301217 |
| `always` | 0.8167 | 1385.87 | 3820.43 | 0.213763 |
| `disagreement` | 0.8167 | 471.30 | 2905.87 | 0.281041 |
| `confidence` | 0.7333 | 20.78 | 2455.35 | 0.298667 |
| `hybrid` | 0.8167 | 515.05 | 2949.62 | 0.276872 |
| `confidence_095` | 0.7667 | 293.50 | 2728.07 | 0.281029 |
| `hybrid_relaxed` | 0.8167 | 604.88 | 3039.45 | 0.268689 |
| `claim_divergence` | 0.8167 | 674.00 | 3108.57 | 0.262715 |
| `voc_v2` | 0.8167 | 718.43 | 3153.00 | 0.259013 |
| `mv_6` | 0.7167 | 0.00 | 4851.58 | 0.147718 |
| `sc_6` | 0.7000 | 0.00 | 4854.83 | 0.144186 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0833 | 1.0000 | 0.9167 | 0.0000 | 1385.87 | 3820.43 | 0.213763 |
| `disagreement` | 0.3333 | 0.6667 | 0.2500 | 1.0000 | 0.2500 | 0.0000 | 471.30 | 2905.87 | 0.281041 |
| `confidence` | 0.0167 | 0.9833 | 0.0000 | 0.0000 | 0.0167 | 1.0000 | 20.78 | 2455.35 | 0.298667 |
| `hybrid` | 0.3667 | 0.6333 | 0.2273 | 1.0000 | 0.2833 | 0.0000 | 515.05 | 2949.62 | 0.276872 |
| `confidence_095` | 0.2167 | 0.7833 | 0.1538 | 0.4000 | 0.1833 | 0.6000 | 293.50 | 2728.07 | 0.281029 |
| `hybrid_relaxed` | 0.4333 | 0.5667 | 0.1923 | 1.0000 | 0.3500 | 0.0000 | 604.88 | 3039.45 | 0.268689 |
| `claim_divergence` | 0.4833 | 0.5167 | 0.1724 | 1.0000 | 0.4000 | 0.0000 | 674.00 | 3108.57 | 0.262715 |
| `voc_v2` | 0.5167 | 0.4833 | 0.1613 | 1.0000 | 0.4333 | 0.0000 | 718.43 | 3153.00 | 0.259013 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.9167 | 1.0000 | 1385.87 | 3820.43 |
| `disagreement` | 1.0000 | 0.0000 | 0.7500 | 0.3333 | 471.30 | 2905.87 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0167 | 20.78 | 2455.35 |
| `hybrid` | 1.0000 | 0.0000 | 0.7727 | 0.3667 | 515.05 | 2949.62 |
| `confidence_095` | 0.4000 | 0.0000 | 0.8462 | 0.2167 | 293.50 | 2728.07 |
| `hybrid_relaxed` | 1.0000 | 0.0000 | 0.8077 | 0.4333 | 604.88 | 3039.45 |
| `claim_divergence` | 1.0000 | 0.0000 | 0.8276 | 0.4833 | 674.00 | 3108.57 |
| `voc_v2` | 1.0000 | 0.0000 | 0.8387 | 0.5167 | 718.43 | 3153.00 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 1171.65 | 0.597448 |
| `always` | 0.9000 | 1479.15 | 2650.80 | 0.339520 |
| `disagreement` | 0.9000 | 481.85 | 1653.50 | 0.544300 |
| `confidence` | 0.7000 | 0.00 | 1171.65 | 0.597448 |
| `hybrid` | 0.9000 | 481.85 | 1653.50 | 0.544300 |
| `confidence_095` | 0.8000 | 153.35 | 1325.00 | 0.603774 |
| `hybrid_relaxed` | 0.9000 | 481.85 | 1653.50 | 0.544300 |
| `claim_divergence` | 0.9000 | 624.65 | 1796.30 | 0.501030 |
| `voc_v2` | 0.9000 | 624.65 | 1796.30 | 0.501030 |
| `mv_6` | 0.7000 | 0.00 | 2280.50 | 0.306950 |
| `sc_6` | 0.7000 | 0.00 | 2298.25 | 0.304580 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.2000 | 1.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.6667 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.6667 | 1.0000 |
| `confidence_095` | 0.1000 | 0.9000 | 1.0000 | 0.5000 |
| `hybrid_relaxed` | 0.3000 | 0.7000 | 0.6667 | 1.0000 |
| `claim_divergence` | 0.4000 | 0.6000 | 0.5000 | 1.0000 |
| `voc_v2` | 0.4000 | 0.6000 | 0.5000 | 1.0000 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 5128.45 | 0.136493 |
| `always` | 0.7500 | 1395.90 | 6524.35 | 0.114954 |
| `disagreement` | 0.7500 | 487.75 | 5616.20 | 0.133542 |
| `confidence` | 0.7000 | 0.00 | 5128.45 | 0.136493 |
| `hybrid` | 0.7500 | 556.65 | 5685.10 | 0.131924 |
| `confidence_095` | 0.7000 | 209.60 | 5338.05 | 0.131134 |
| `hybrid_relaxed` | 0.7500 | 627.85 | 5756.30 | 0.130292 |
| `claim_divergence` | 0.7500 | 626.50 | 5754.95 | 0.130323 |
| `voc_v2` | 0.7500 | 695.40 | 5823.85 | 0.128781 |
| `mv_6` | 0.7000 | 0.00 | 10248.55 | 0.068302 |
| `sc_6` | 0.7000 | 0.00 | 10247.20 | 0.068311 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.3500 | 0.6500 | 0.1429 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.4000 | 0.6000 | 0.1250 | 1.0000 |
| `confidence_095` | 0.1500 | 0.8500 | 0.0000 | 0.0000 |
| `hybrid_relaxed` | 0.4500 | 0.5500 | 0.1111 | 1.0000 |
| `claim_divergence` | 0.4500 | 0.5500 | 0.1111 | 1.0000 |
| `voc_v2` | 0.5000 | 0.5000 | 0.1000 | 1.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 1003.60 | 0.797130 |
| `always` | 0.8000 | 1282.55 | 2286.15 | 0.349933 |
| `disagreement` | 0.8000 | 444.30 | 1447.90 | 0.552524 |
| `confidence` | 0.8000 | 62.35 | 1065.95 | 0.750504 |
| `hybrid` | 0.8000 | 506.65 | 1510.25 | 0.529714 |
| `confidence_095` | 0.8000 | 517.55 | 1521.15 | 0.525918 |
| `hybrid_relaxed` | 0.8000 | 704.95 | 1708.55 | 0.468233 |
| `claim_divergence` | 0.8000 | 770.85 | 1774.45 | 0.450844 |
| `voc_v2` | 0.8000 | 835.25 | 1838.85 | 0.435055 |
| `mv_6` | 0.7500 | 0.00 | 2025.70 | 0.370242 |
| `sc_6` | 0.7000 | 0.00 | 2019.05 | 0.346698 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.3500 | 0.6500 | 0.0000 | 0.0000 |
| `confidence` | 0.0500 | 0.9500 | 0.0000 | 0.0000 |
| `hybrid` | 0.4000 | 0.6000 | 0.0000 | 0.0000 |
| `confidence_095` | 0.4000 | 0.6000 | 0.0000 | 0.0000 |
| `hybrid_relaxed` | 0.5500 | 0.4500 | 0.0000 | 0.0000 |
| `claim_divergence` | 0.6000 | 0.4000 | 0.0000 | 0.0000 |
| `voc_v2` | 0.6500 | 0.3500 | 0.0000 | 0.0000 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00906`
- 问题预览：Tim has a box with 7 blue shoe boxes and 9 red shoe boxes. If he uses 3 blue shoeboxes and 1/3 red of his shoeboxes t...
- 金标：`10`
- `mv_3`：`10` / score=`1.0`
- `always`：`10` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00945`
- 问题预览：James loves to go swimming and has to swim across a 20-mile lake. He can swim at a pace of 2 miles per hour. He swims...
- 金标：`17`
- `mv_3`：`17` / score=`1.0`
- `always`：`17` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`hotpotqa`
- 样本：`5a77897f55429949eeb29edc`
- 问题预览：Jason Regler, stated that he had the idea for the flashing wristbands during a song built around which instrument ?
- 金标：`an organ`
- `mv_3`：`organ` / score=`1.0`
- `always`：`organ` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`hotpotqa`
- 样本：`5a7a567255429941d65f25bd`
- 问题预览：What was Iqbal F. Qadir on when he participated in an attack on a radar station located on western shore of the Okham...
- 金标：`flotilla`
- `mv_3`：`part of flotilla that attacked radar station in dwarka india` / score=`0.0`
- `always`：`part of flotilla` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`hotpotqa`
- 样本：`5a80cf4c55429938b61421f6`
- 问题预览：What was the concept of the business Eric S .Pistorius worked for after being an attorney?
- 金标：`to ensure wide visibility and understanding of cases in a region`
- `mv_3`：`circuit court judge of seventh circuit of illinois` / score=`0.0`
- `always`：`circuit court judge of seventh circuit of illinois` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.174701`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

