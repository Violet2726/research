# CUE 实验报告

## 1. 实验概览

- 实验名：`cue_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_v1/pilot100/20260507T121733Z-cue_v1-pilot100-xiaomimimo-mimo-v2.5`
- 方法主线：独立求解 -> utility 估计 -> 定向通信 -> 可选局部审计。

## 2. Overall 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5540 | 0.00 | 2134.73 | 3.00 | 0.259517 |
| `always` | 0.6340 | 3197.68 | 5332.41 | 6.25 | 0.118896 |
| `disagreement` | 0.6120 | 1209.69 | 3344.42 | 4.40 | 0.182991 |
| `consensus_freeze` | 0.6340 | 3072.73 | 5207.46 | 6.18 | 0.121748 |
| `cue_v1` | 0.5840 | 646.17 | 2780.90 | 3.78 | 0.210004 |

## 3. Trigger 诊断

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1060 | 1.0000 | 0.8940 | 0.0000 |
| `disagreement` | 0.3900 | 0.6100 | 0.2154 | 0.7925 | 0.3060 | 0.2075 |
| `consensus_freeze` | 0.9760 | 0.0240 | 0.1086 | 1.0000 | 0.8700 | 0.0000 |
| `cue_v1` | 0.2140 | 0.7860 | 0.2243 | 0.4528 | 0.1660 | 0.5472 |

## 4. 分数据集结果

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4700 | 0.00 | 1335.26 | 3.00 | 0.351991 |
| `always` | 0.7200 | 2414.43 | 3749.69 | 6.35 | 0.192016 |
| `disagreement` | 0.6700 | 1444.40 | 2779.66 | 5.02 | 0.241037 |
| `consensus_freeze` | 0.7200 | 2414.43 | 3749.69 | 6.35 | 0.192016 |
| `cue_v1` | 0.5800 | 759.08 | 2094.34 | 4.06 | 0.276937 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6600 | 0.00 | 5298.60 | 3.00 | 0.124561 |
| `always` | 0.6900 | 6462.93 | 11761.53 | 6.12 | 0.058666 |
| `disagreement` | 0.6900 | 1457.34 | 6755.94 | 3.69 | 0.102132 |
| `consensus_freeze` | 0.6900 | 5875.81 | 11174.41 | 5.82 | 0.061748 |
| `cue_v1` | 0.6700 | 571.03 | 5869.63 | 3.27 | 0.114147 |

### math500

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.3600 | 0.00 | 1368.39 | 3.00 | 0.263083 |
| `always` | 0.4600 | 2562.66 | 3931.05 | 6.43 | 0.117017 |
| `disagreement` | 0.4200 | 1655.08 | 3023.47 | 5.23 | 0.138913 |
| `consensus_freeze` | 0.4600 | 2543.92 | 3912.31 | 6.40 | 0.117578 |
| `cue_v1` | 0.3900 | 943.87 | 2312.26 | 4.24 | 0.168666 |

### mmlu_pro

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6100 | 0.00 | 1612.44 | 3.00 | 0.378309 |
| `always` | 0.6200 | 2612.29 | 4224.73 | 6.21 | 0.146755 |
| `disagreement` | 0.6000 | 1093.77 | 2706.21 | 4.35 | 0.221712 |
| `consensus_freeze` | 0.6200 | 2612.29 | 4224.73 | 6.21 | 0.146755 |
| `cue_v1` | 0.6000 | 737.65 | 2350.09 | 3.93 | 0.255309 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6700 | 0.00 | 1058.97 | 3.00 | 0.632690 |
| `always` | 0.6800 | 1936.09 | 2995.06 | 6.13 | 0.227041 |
| `disagreement` | 0.6800 | 397.86 | 1456.83 | 3.69 | 0.466767 |
| `consensus_freeze` | 0.6800 | 1917.21 | 2976.18 | 6.10 | 0.228481 |
| `cue_v1` | 0.6800 | 219.21 | 1278.18 | 3.38 | 0.532006 |

## 5. 默认策略建议

- 推荐策略：`disagreement_triggered`
- 相对 `always_communicate` 的准确率下降：`0.05`
- 相对 `always_communicate` 的总 token 下降比例：`0.478491`

## 6. 局限

- 当前报告主要用于首轮机制验证。
- helpful / harmful 通信仍以 `always_communicate` 相对 `mv_3` 的变化作为 oracle 近似。
- 更大规模结论仍需更长周期实验进一步确认。

