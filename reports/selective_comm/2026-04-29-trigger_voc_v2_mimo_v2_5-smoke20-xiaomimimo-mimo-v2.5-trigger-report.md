# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2_mimo_v2_5`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`local/runs/selective_comm/trigger_voc_v2_mimo_v2_5/smoke20/20260429T020901Z-trigger_voc_v2_mimo_v2_5-smoke20-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`55839.00`；若按 4 套 trigger 独立重跑则为 `371093.00`；共享前缀节省比例=`0.8495`
- `hotpotqa`：共享实际 token=`133784.00`；若按 4 套 trigger 独立重跑则为 `941835.00`；共享前缀节省比例=`0.8580`
- `overall`：共享实际 token=`235532.00`；若按 4 套 trigger 独立重跑则为 `1579626.00`；共享前缀节省比例=`0.8509`
- `strategyqa`：共享实际 token=`45909.00`；若按 4 套 trigger 独立重跑则为 `266698.00`；共享前缀节省比例=`0.8279`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 2492.60 | 0.280831 |
| `always` | 0.7833 | 1432.93 | 3925.53 | 0.199548 |
| `disagreement` | 0.7500 | 755.90 | 3248.50 | 0.230876 |
| `confidence` | 0.7000 | 0.00 | 2492.60 | 0.280831 |
| `hybrid` | 0.7500 | 755.90 | 3248.50 | 0.230876 |
| `confidence_095` | 0.7167 | 484.45 | 2977.05 | 0.240731 |
| `hybrid_relaxed` | 0.7500 | 1053.15 | 3545.75 | 0.211521 |
| `claim_divergence` | 0.7667 | 928.50 | 3421.10 | 0.224100 |
| `voc_v2` | 0.7667 | 975.47 | 3468.07 | 0.221065 |
| `mv_6` | 0.6333 | 0.00 | 4989.62 | 0.126930 |
| `sc_6` | 0.6667 | 0.00 | 4989.65 | 0.133610 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1333 | 1.0000 | 0.8667 | 0.0000 | 1432.93 | 3925.53 | 0.199548 |
| `disagreement` | 0.5000 | 0.5000 | 0.2000 | 0.7500 | 0.4000 | 0.2500 | 755.90 | 3248.50 | 0.230876 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.00 | 2492.60 | 0.280831 |
| `hybrid` | 0.5000 | 0.5000 | 0.2000 | 0.7500 | 0.4000 | 0.2500 | 755.90 | 3248.50 | 0.230876 |
| `confidence_095` | 0.3500 | 0.6500 | 0.0952 | 0.2500 | 0.3167 | 0.7500 | 484.45 | 2977.05 | 0.240731 |
| `hybrid_relaxed` | 0.7167 | 0.2833 | 0.1395 | 0.7500 | 0.6167 | 0.2500 | 1053.15 | 3545.75 | 0.211521 |
| `claim_divergence` | 0.6333 | 0.3667 | 0.1842 | 0.8750 | 0.5167 | 0.1250 | 928.50 | 3421.10 | 0.224100 |
| `voc_v2` | 0.6667 | 0.3333 | 0.1750 | 0.8750 | 0.5500 | 0.1250 | 975.47 | 3468.07 | 0.221065 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 1.0000 | 0.8167 | 1.0000 | 1432.93 | 3925.53 |
| `disagreement` | 0.7500 | 1.0000 | 0.7000 | 0.5000 | 755.90 | 3248.50 |
| `confidence` | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.00 | 2492.60 |
| `hybrid` | 0.7500 | 1.0000 | 0.7000 | 0.5000 | 755.90 | 3248.50 |
| `confidence_095` | 0.2500 | 0.3333 | 0.8571 | 0.3500 | 484.45 | 2977.05 |
| `hybrid_relaxed` | 0.7500 | 1.0000 | 0.7907 | 0.7167 | 1053.15 | 3545.75 |
| `claim_divergence` | 0.8750 | 1.0000 | 0.7368 | 0.6333 | 928.50 | 3421.10 |
| `voc_v2` | 0.8750 | 1.0000 | 0.7500 | 0.6667 | 975.47 | 3468.07 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 1213.05 | 0.453403 |
| `always` | 0.7500 | 1578.90 | 2791.95 | 0.268629 |
| `disagreement` | 0.7000 | 1438.05 | 2651.10 | 0.264041 |
| `confidence` | 0.5500 | 0.00 | 1213.05 | 0.453403 |
| `hybrid` | 0.7000 | 1438.05 | 2651.10 | 0.264041 |
| `confidence_095` | 0.6000 | 81.10 | 1294.15 | 0.463625 |
| `hybrid_relaxed` | 0.7000 | 1438.05 | 2651.10 | 0.264041 |
| `claim_divergence` | 0.7000 | 1438.05 | 2651.10 | 0.264041 |
| `voc_v2` | 0.7000 | 1438.05 | 2651.10 | 0.264041 |
| `mv_6` | 0.4000 | 0.00 | 2424.00 | 0.165017 |
| `sc_6` | 0.5500 | 0.00 | 2400.40 | 0.229128 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.3000 | 1.0000 |
| `disagreement` | 0.9000 | 0.1000 | 0.2778 | 0.8333 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.9000 | 0.1000 | 0.2778 | 0.8333 |
| `confidence_095` | 0.0500 | 0.9500 | 1.0000 | 0.1667 |
| `hybrid_relaxed` | 0.9000 | 0.1000 | 0.2778 | 0.8333 |
| `claim_divergence` | 0.9000 | 0.1000 | 0.2778 | 0.8333 |
| `voc_v2` | 0.9000 | 0.1000 | 0.2778 | 0.8333 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5263.45 | 0.142492 |
| `always` | 0.7000 | 1425.75 | 6689.20 | 0.104646 |
| `disagreement` | 0.7000 | 438.65 | 5702.10 | 0.122762 |
| `confidence` | 0.7500 | 0.00 | 5263.45 | 0.142492 |
| `hybrid` | 0.7000 | 438.65 | 5702.10 | 0.122762 |
| `confidence_095` | 0.7000 | 724.00 | 5987.45 | 0.116911 |
| `hybrid_relaxed` | 0.7000 | 938.90 | 6202.35 | 0.112860 |
| `claim_divergence` | 0.7000 | 438.65 | 5702.10 | 0.122762 |
| `voc_v2` | 0.7000 | 579.55 | 5843.00 | 0.119801 |
| `mv_6` | 0.7500 | 0.00 | 10541.15 | 0.071150 |
| `sc_6` | 0.7500 | 0.00 | 10562.10 | 0.071009 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0000 | 0.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| `confidence_095` | 0.5000 | 0.5000 | 0.0000 | 0.0000 |
| `hybrid_relaxed` | 0.6500 | 0.3500 | 0.0000 | 0.0000 |
| `claim_divergence` | 0.3000 | 0.7000 | 0.0000 | 0.0000 |
| `voc_v2` | 0.4000 | 0.6000 | 0.0000 | 0.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 1001.30 | 0.798961 |
| `always` | 0.9000 | 1294.15 | 2295.45 | 0.392080 |
| `disagreement` | 0.8500 | 391.00 | 1392.30 | 0.610501 |
| `confidence` | 0.8000 | 0.00 | 1001.30 | 0.798961 |
| `hybrid` | 0.8500 | 391.00 | 1392.30 | 0.610501 |
| `confidence_095` | 0.8500 | 648.25 | 1649.55 | 0.515292 |
| `hybrid_relaxed` | 0.8500 | 782.50 | 1783.80 | 0.476511 |
| `claim_divergence` | 0.9000 | 908.80 | 1910.10 | 0.471180 |
| `voc_v2` | 0.9000 | 908.80 | 1910.10 | 0.471180 |
| `mv_6` | 0.7500 | 0.00 | 2003.70 | 0.374308 |
| `sc_6` | 0.7000 | 0.00 | 2006.45 | 0.348875 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1000 | 1.0000 |
| `disagreement` | 0.3000 | 0.7000 | 0.1667 | 0.5000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.3000 | 0.7000 | 0.1667 | 0.5000 |
| `confidence_095` | 0.5000 | 0.5000 | 0.1000 | 0.5000 |
| `hybrid_relaxed` | 0.6000 | 0.4000 | 0.0833 | 0.5000 |
| `claim_divergence` | 0.7000 | 0.3000 | 0.1429 | 1.0000 |
| `voc_v2` | 0.7000 | 0.3000 | 0.1429 | 1.0000 |

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
- 样本：`gsm8k-00222`
- 问题预览：Andy plants 90 geraniums and 40 fewer petunias that geraniums. How many flowers does he plant total?
- 金标：`140`
- `mv_3`：`140` / score=`1.0`
- `always`：`140` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00237`
- 问题预览：Johnny took his allowance of $20 and added an extra $10 to it. He then invested this sum of money, which tripled in a...
- 金标：`90`
- `mv_3`：`90` / score=`1.0`
- `always`：`90` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00238`
- 问题预览：Mary is two years younger than Joan, who is five years older than Jessa. If Jessa is 20 years old, what is the sum of...
- 金标：`68`
- `mv_3`：`59` / score=`0.0`
- `always`：`59` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00289`
- 问题预览：A car in the fast lane is traveling at 60 miles/hour. A car in the slow lane is traveling at half that speed. If the ...
- 金标：`16`
- `mv_3`：`16` / score=`1.0`
- `always`：`16` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`disagreement_triggered`
- 相对 `always_communicate` 的准确率下降：`0.016666`
- 相对 `always_communicate` 的总 token 下降比例：`0.116536`
- 规则是否通过：`False`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

