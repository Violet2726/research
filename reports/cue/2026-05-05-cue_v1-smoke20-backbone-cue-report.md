# CUE 实验报告

## 1. 实验概览

- 实验名：`cue_v1`
- Phase：`smoke20`
- Backbone：`None`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_v1/smoke20/20260505T053917Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法主线：独立求解 -> utility 估计 -> 定向通信 -> 可选局部审计。

## 2. Overall 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6200 | 0.00 | 2114.78 | 3.00 | 0.293175 |
| `always` | 0.6700 | 3216.87 | 5331.65 | 6.30 | 0.125665 |
| `disagreement` | 0.6700 | 1278.91 | 3393.69 | 4.48 | 0.197425 |
| `consensus_freeze` | 0.6700 | 3117.84 | 5232.62 | 6.24 | 0.128043 |
| `cue_v1` | 0.6700 | 522.90 | 2637.68 | 3.66 | 0.254011 |

## 3. Trigger 诊断

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0900 | 1.0000 | 0.9100 | 0.0000 |
| `disagreement` | 0.4000 | 0.6000 | 0.2250 | 1.0000 | 0.3100 | 0.0000 |
| `consensus_freeze` | 0.9800 | 0.0200 | 0.0918 | 1.0000 | 0.8900 | 0.0000 |
| `cue_v1` | 0.1800 | 0.8200 | 0.3333 | 0.6667 | 0.1200 | 0.3333 |

## 4. 分数据集结果

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1262.50 | 3.00 | 0.475248 |
| `always` | 0.8000 | 2331.05 | 3593.55 | 6.30 | 0.222621 |
| `disagreement` | 0.8000 | 1407.35 | 2669.85 | 4.95 | 0.299642 |
| `consensus_freeze` | 0.8000 | 2331.05 | 3593.55 | 6.30 | 0.222621 |
| `cue_v1` | 0.7500 | 541.55 | 1804.05 | 3.75 | 0.415731 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 5342.50 | 3.00 | 0.149743 |
| `always` | 0.7500 | 6634.10 | 11976.60 | 6.20 | 0.062622 |
| `disagreement` | 0.7500 | 1598.05 | 6940.55 | 3.75 | 0.108061 |
| `consensus_freeze` | 0.7500 | 6138.95 | 11481.45 | 5.90 | 0.065323 |
| `cue_v1` | 0.8000 | 307.50 | 5650.00 | 3.15 | 0.141593 |

### math500

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4000 | 0.00 | 1305.00 | 3.00 | 0.306513 |
| `always` | 0.4000 | 2603.95 | 3908.95 | 6.60 | 0.102329 |
| `disagreement` | 0.4000 | 1874.25 | 3179.25 | 5.55 | 0.125816 |
| `consensus_freeze` | 0.4000 | 2603.95 | 3908.95 | 6.60 | 0.102329 |
| `cue_v1` | 0.4000 | 1142.60 | 2447.60 | 4.55 | 0.163425 |

### mmlu_pro

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1635.40 | 3.00 | 0.366883 |
| `always` | 0.6500 | 2534.25 | 4169.65 | 6.15 | 0.155888 |
| `disagreement` | 0.6500 | 1034.35 | 2669.75 | 4.35 | 0.243468 |
| `consensus_freeze` | 0.6500 | 2534.25 | 4169.65 | 6.15 | 0.155888 |
| `cue_v1` | 0.6500 | 484.15 | 2119.55 | 3.65 | 0.306669 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 1028.50 | 3.00 | 0.680603 |
| `always` | 0.7500 | 1981.00 | 3009.50 | 6.25 | 0.249211 |
| `disagreement` | 0.7500 | 480.55 | 1509.05 | 3.80 | 0.497001 |
| `consensus_freeze` | 0.7500 | 1981.00 | 3009.50 | 6.25 | 0.249211 |
| `cue_v1` | 0.7500 | 138.70 | 1167.20 | 3.20 | 0.642563 |

## 5. 默认策略建议

- 推荐策略：`cue_v1`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.505279`

## 6. 局限

- 当前报告主要用于首轮机制验证。
- helpful / harmful 通信仍以 `always_communicate` 相对 `mv_3` 的变化作为 oracle 近似。
- 更大规模结论仍需更长周期实验进一步确认。

