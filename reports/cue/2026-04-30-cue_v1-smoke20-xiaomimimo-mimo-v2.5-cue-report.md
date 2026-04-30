# CUE 实验报告

## 1. 实验概览

- 实验名：`cue_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_v1/smoke20/20260430T052958Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法主线：独立求解 -> utility 估计 -> 定向通信 -> 可选局部审计。

## 2. Overall 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6200 | 0.00 | 2114.78 | 3.00 | 0.293175 |
| `always` | 0.6700 | 3234.91 | 5349.69 | 6.29 | 0.125241 |
| `disagreement` | 0.6700 | 1287.08 | 3401.86 | 4.47 | 0.196951 |
| `consensus_freeze` | 0.6700 | 3135.88 | 5250.66 | 6.23 | 0.127603 |
| `cue_v1` | 0.6600 | 494.36 | 2609.14 | 3.62 | 0.252957 |

## 3. Trigger 诊断

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0900 | 1.0000 | 0.9100 | 0.0000 |
| `disagreement` | 0.4000 | 0.6000 | 0.2250 | 1.0000 | 0.3100 | 0.0000 |
| `consensus_freeze` | 0.9800 | 0.0200 | 0.0918 | 1.0000 | 0.8900 | 0.0000 |
| `cue_v1` | 0.1700 | 0.8300 | 0.2941 | 0.5556 | 0.1200 | 0.4444 |

## 4. 分数据集结果

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1262.50 | 3.00 | 0.475248 |
| `always` | 0.8000 | 2282.55 | 3545.05 | 6.25 | 0.225667 |
| `disagreement` | 0.8000 | 1358.85 | 2621.35 | 4.90 | 0.305186 |
| `consensus_freeze` | 0.8000 | 2282.55 | 3545.05 | 6.25 | 0.225667 |
| `cue_v1` | 0.7000 | 398.85 | 1661.35 | 3.55 | 0.421344 |

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
| `always` | 0.4000 | 2621.70 | 3926.70 | 6.60 | 0.101867 |
| `disagreement` | 0.4000 | 1892.00 | 3197.00 | 5.55 | 0.125117 |
| `consensus_freeze` | 0.4000 | 2621.70 | 3926.70 | 6.60 | 0.101867 |
| `cue_v1` | 0.4000 | 1142.60 | 2447.60 | 4.55 | 0.163425 |

### mmlu_pro

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1635.40 | 3.00 | 0.366883 |
| `always` | 0.6500 | 2656.60 | 4292.00 | 6.15 | 0.151445 |
| `disagreement` | 0.6500 | 1105.95 | 2741.35 | 4.35 | 0.237109 |
| `consensus_freeze` | 0.6500 | 2656.60 | 4292.00 | 6.15 | 0.151445 |
| `cue_v1` | 0.6500 | 484.15 | 2119.55 | 3.65 | 0.306669 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 1028.50 | 3.00 | 0.680603 |
| `always` | 0.7500 | 1979.60 | 3008.10 | 6.25 | 0.249327 |
| `disagreement` | 0.7500 | 480.55 | 1509.05 | 3.80 | 0.497001 |
| `consensus_freeze` | 0.7500 | 1979.60 | 3008.10 | 6.25 | 0.249327 |
| `cue_v1` | 0.7500 | 138.70 | 1167.20 | 3.20 | 0.642563 |

## 5. 默认策略建议

- 推荐策略：`cue_v1`
- 相对 `always_communicate` 的准确率下降：`0.01`
- 相对 `always_communicate` 的总 token 下降比例：`0.512282`

## 6. 局限

- 当前报告主要用于首轮机制验证。
- helpful / harmful 通信仍以 `always_communicate` 相对 `mv_3` 的变化作为 oracle 近似。
- 更大规模结论仍需更长周期实验进一步确认。

