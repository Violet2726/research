# CUE 实验报告

## 1. 实验概览

- 实验名：`cue_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`cue_v1_json`
- 运行目录：`runs/cue/cue_v1/smoke20/20260505T100651Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法主线：独立求解 -> utility 估计 -> 定向通信 -> 可选局部审计。

## 2. Overall 主结果

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6200 | 0.00 | 2124.89 | 3.00 | 0.291780 |
| `always` | 0.6600 | 3208.33 | 5333.22 | 6.25 | 0.123753 |
| `disagreement` | 0.6500 | 1241.44 | 3366.33 | 4.30 | 0.193089 |
| `consensus_freeze` | 0.6600 | 3138.79 | 5263.68 | 6.22 | 0.125388 |
| `cue_v1` | 0.6600 | 617.17 | 2742.06 | 3.63 | 0.240695 |

## 3. Trigger 诊断

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0700 | 1.0000 | 0.9300 | 0.0000 |
| `disagreement` | 0.3600 | 0.6400 | 0.1667 | 0.8571 | 0.3000 | 0.1429 |
| `consensus_freeze` | 0.9900 | 0.0100 | 0.0707 | 1.0000 | 0.9200 | 0.0000 |
| `cue_v1` | 0.1800 | 0.8200 | 0.2778 | 0.7143 | 0.1300 | 0.2857 |

## 4. 分数据集结果

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 1278.10 | 3.00 | 0.430326 |
| `always` | 0.6500 | 2357.90 | 3636.00 | 6.35 | 0.178768 |
| `disagreement` | 0.6500 | 1389.55 | 2667.65 | 4.95 | 0.243660 |
| `consensus_freeze` | 0.6500 | 2357.90 | 3636.00 | 6.35 | 0.178768 |
| `cue_v1` | 0.6000 | 313.20 | 1591.30 | 3.45 | 0.377050 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.8000 | 0.00 | 5347.60 | 3.00 | 0.149600 |
| `always` | 0.8000 | 6763.95 | 12111.55 | 6.25 | 0.066053 |
| `disagreement` | 0.8000 | 2371.70 | 7719.30 | 4.10 | 0.103636 |
| `consensus_freeze` | 0.8000 | 6416.25 | 11763.85 | 6.10 | 0.068005 |
| `cue_v1` | 0.8500 | 1187.00 | 6534.60 | 3.55 | 0.130077 |

### math500

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4500 | 0.00 | 1307.90 | 3.00 | 0.344063 |
| `always` | 0.5500 | 2461.90 | 3769.80 | 6.40 | 0.145896 |
| `disagreement` | 0.5000 | 1339.90 | 2647.80 | 4.85 | 0.188836 |
| `consensus_freeze` | 0.5500 | 2461.90 | 3769.80 | 6.40 | 0.145896 |
| `cue_v1` | 0.5000 | 957.50 | 2265.40 | 4.30 | 0.220712 |

### mmlu_pro

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 1633.05 | 3.00 | 0.367411 |
| `always` | 0.6500 | 2529.55 | 4162.60 | 6.10 | 0.156152 |
| `disagreement` | 0.6500 | 760.75 | 2393.80 | 4.00 | 0.271535 |
| `consensus_freeze` | 0.6500 | 2529.55 | 4162.60 | 6.10 | 0.156152 |
| `cue_v1` | 0.6500 | 513.65 | 2146.70 | 3.65 | 0.302790 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 1057.80 | 3.00 | 0.661751 |
| `always` | 0.6500 | 1928.35 | 2986.15 | 6.15 | 0.217672 |
| `disagreement` | 0.6500 | 345.30 | 1403.10 | 3.60 | 0.463260 |
| `consensus_freeze` | 0.6500 | 1928.35 | 2986.15 | 6.15 | 0.217672 |
| `cue_v1` | 0.7000 | 114.50 | 1172.30 | 3.20 | 0.597117 |

## 5. 默认策略建议

- 推荐策略：`cue_v1`
- 相对 `always_communicate` 的准确率下降：`0.0`
- 相对 `always_communicate` 的总 token 下降比例：`0.485853`

## 6. 局限

- 当前报告主要用于首轮机制验证。
- helpful / harmful 通信仍以 `always_communicate` 相对 `mv_3` 的变化作为 oracle 近似。
- 更大规模结论仍需更长周期实验进一步确认。

