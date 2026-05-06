# Trigger / Early-exit 实验报告

## 1. 实验范围与公平性说明

- 实验名：`trigger_voc_v2`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`selective_comm_voc_json_v2`
- 运行目录：`runs/selective_comm/trigger_voc_v2/pilot100/20260506T140735Z-trigger_voc_v2-pilot100-xiaomimimo-mimo-v2.5`
- 数据集：`GSM8K + StrategyQA + HotpotQA`，当前轮次只解释 `smoke20`。
- 共享前缀设计：4 个 trigger 策略共享同一份 Stage A 与同一份 Stage B，不重复发 4 套网络请求。
- 方法边界：本轮只回答“何时通信 / 何时 early exit”，不混入消息内容压缩和局部审计。

## 2. 共享前缀设计与预算节省说明

- `gsm8k`：共享实际 token=`290598.00`；若按 4 套 trigger 独立重跑则为 `1638471.00`；共享前缀节省比例=`0.8226`
- `hotpotqa`：共享实际 token=`672678.00`；若按 4 套 trigger 独立重跑则为 `4662219.00`；共享前缀节省比例=`0.8557`
- `overall`：共享实际 token=`1189543.00`；若按 4 套 trigger 独立重跑则为 `7611536.00`；共享前缀节省比例=`0.8437`
- `strategyqa`：共享实际 token=`226267.00`；若按 4 套 trigger 独立重跑则为 `1310846.00`；共享前缀节省比例=`0.8274`

## 3. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6033 | 0.00 | 2486.05 | 0.242688 |
| `always` | 0.7400 | 1479.10 | 3965.14 | 0.186626 |
| `disagreement` | 0.7233 | 516.60 | 3002.65 | 0.240898 |
| `confidence` | 0.6033 | 16.44 | 2502.49 | 0.241093 |
| `hybrid` | 0.7233 | 546.29 | 3032.33 | 0.238540 |
| `confidence_095` | 0.6367 | 574.22 | 3060.27 | 0.208043 |
| `hybrid_relaxed` | 0.7300 | 924.11 | 3410.16 | 0.214066 |
| `claim_divergence` | 0.7300 | 652.07 | 3138.11 | 0.232624 |
| `voc_v2` | 0.7300 | 774.59 | 3260.64 | 0.223883 |
| `mv_6` | 0.6500 | 0.00 | 4967.32 | 0.130855 |
| `sc_6` | 0.6433 | 0.00 | 4971.37 | 0.129408 |

## 4. Trigger 诊断表

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall | False Trigger Rate | Missed Beneficial Comm Rate | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.1367 | 1.0000 | 0.8633 | 0.0000 | 1479.10 | 3965.14 | 0.186626 |
| `disagreement` | 0.3233 | 0.6767 | 0.3711 | 0.8780 | 0.2033 | 0.1220 | 516.60 | 3002.65 | 0.240898 |
| `confidence` | 0.0133 | 0.9867 | 0.0000 | 0.0000 | 0.0133 | 1.0000 | 16.44 | 2502.49 | 0.241093 |
| `hybrid` | 0.3467 | 0.6533 | 0.3462 | 0.8780 | 0.2267 | 0.1220 | 546.29 | 3032.33 | 0.238540 |
| `confidence_095` | 0.4100 | 0.5900 | 0.0813 | 0.2439 | 0.3767 | 0.7561 | 574.22 | 3060.27 | 0.208043 |
| `hybrid_relaxed` | 0.6200 | 0.3800 | 0.2043 | 0.9268 | 0.4933 | 0.0732 | 924.11 | 3410.16 | 0.214066 |
| `claim_divergence` | 0.4200 | 0.5800 | 0.3016 | 0.9268 | 0.2933 | 0.0732 | 652.07 | 3138.11 | 0.232624 |
| `voc_v2` | 0.5100 | 0.4900 | 0.2484 | 0.9268 | 0.3833 | 0.0732 | 774.59 | 3260.64 | 0.223883 |

## 5. VoC 诊断表

| Policy | Helpful Recall | Harmful Trigger Rate | Neutral Waste Rate | Trigger Rate | Avg Comm Tokens | Avg Total Tokens |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.8633 | 1.0000 | 1479.10 | 3965.14 |
| `disagreement` | 0.8780 | 0.0000 | 0.6289 | 0.3233 | 516.60 | 3002.65 |
| `confidence` | 0.0000 | 0.0000 | 1.0000 | 0.0133 | 16.44 | 2502.49 |
| `hybrid` | 0.8780 | 0.0000 | 0.6538 | 0.3467 | 546.29 | 3032.33 |
| `confidence_095` | 0.2439 | 0.0000 | 0.9187 | 0.4100 | 574.22 | 3060.27 |
| `hybrid_relaxed` | 0.9268 | 0.0000 | 0.7957 | 0.6200 | 924.11 | 3410.16 |
| `claim_divergence` | 0.9268 | 0.0000 | 0.6984 | 0.4200 | 652.07 | 3138.11 |
| `voc_v2` | 0.9268 | 0.0000 | 0.7516 | 0.5100 | 774.59 | 3260.64 |

## 6. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4800 | 0.00 | 1221.41 | 0.392988 |
| `always` | 0.8100 | 1684.57 | 2905.98 | 0.278736 |
| `disagreement` | 0.7600 | 908.51 | 2129.92 | 0.356821 |
| `confidence` | 0.4800 | 0.00 | 1221.41 | 0.392988 |
| `hybrid` | 0.7600 | 908.51 | 2129.92 | 0.356821 |
| `confidence_095` | 0.5400 | 166.48 | 1387.89 | 0.389080 |
| `hybrid_relaxed` | 0.7800 | 957.82 | 2179.23 | 0.357925 |
| `claim_divergence` | 0.7800 | 993.77 | 2215.18 | 0.352116 |
| `voc_v2` | 0.7800 | 993.77 | 2215.18 | 0.352116 |
| `mv_6` | 0.5100 | 0.00 | 2432.02 | 0.209702 |
| `sc_6` | 0.5200 | 0.00 | 2437.77 | 0.213310 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.3300 | 1.0000 |
| `disagreement` | 0.5100 | 0.4900 | 0.5490 | 0.8485 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.5100 | 0.4900 | 0.5490 | 0.8485 |
| `confidence_095` | 0.0900 | 0.9100 | 0.6667 | 0.1818 |
| `hybrid_relaxed` | 0.5400 | 0.4600 | 0.5556 | 0.9091 |
| `claim_divergence` | 0.5600 | 0.4400 | 0.5357 | 0.9091 |
| `voc_v2` | 0.5600 | 0.4400 | 0.5357 | 0.9091 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 5271.18 | 0.123312 |
| `always` | 0.6800 | 1455.60 | 6726.78 | 0.101088 |
| `disagreement` | 0.6800 | 309.09 | 5580.27 | 0.121858 |
| `confidence` | 0.6500 | 0.00 | 5271.18 | 0.123312 |
| `hybrid` | 0.6800 | 309.09 | 5580.27 | 0.121858 |
| `confidence_095` | 0.6700 | 630.63 | 5901.81 | 0.113524 |
| `hybrid_relaxed` | 0.6800 | 821.95 | 6093.13 | 0.111601 |
| `claim_divergence` | 0.6800 | 367.79 | 5638.97 | 0.120589 |
| `voc_v2` | 0.6800 | 558.60 | 5829.78 | 0.116642 |
| `mv_6` | 0.7200 | 0.00 | 10538.25 | 0.068323 |
| `sc_6` | 0.6900 | 0.00 | 10543.33 | 0.065444 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0300 | 1.0000 |
| `disagreement` | 0.2100 | 0.7900 | 0.1429 | 1.0000 |
| `confidence` | 0.0000 | 1.0000 | 0.0000 | 0.0000 |
| `hybrid` | 0.2100 | 0.7900 | 0.1429 | 1.0000 |
| `confidence_095` | 0.4300 | 0.5700 | 0.0465 | 0.6667 |
| `hybrid_relaxed` | 0.5600 | 0.4400 | 0.0536 | 1.0000 |
| `claim_divergence` | 0.2500 | 0.7500 | 0.1200 | 1.0000 |
| `voc_v2` | 0.3800 | 0.6200 | 0.0789 | 1.0000 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6800 | 0.00 | 965.55 | 0.704262 |
| `always` | 0.7300 | 1297.12 | 2262.67 | 0.322628 |
| `disagreement` | 0.7300 | 332.20 | 1297.75 | 0.562512 |
| `confidence` | 0.6800 | 49.32 | 1014.87 | 0.670037 |
| `hybrid` | 0.7300 | 421.26 | 1386.81 | 0.526388 |
| `confidence_095` | 0.7000 | 925.56 | 1891.11 | 0.370153 |
| `hybrid_relaxed` | 0.7300 | 992.56 | 1958.11 | 0.372808 |
| `claim_divergence` | 0.7300 | 594.64 | 1560.19 | 0.467892 |
| `voc_v2` | 0.7300 | 771.40 | 1736.95 | 0.420277 |
| `mv_6` | 0.7200 | 0.00 | 1931.69 | 0.372731 |
| `sc_6` | 0.7200 | 0.00 | 1933.00 | 0.372478 |

| Policy | Trigger Rate | Early Exit Rate | Oracle Precision | Oracle Recall |
| --- | ---: | ---: | ---: | ---: |
| `always` | 1.0000 | 0.0000 | 0.0500 | 1.0000 |
| `disagreement` | 0.2500 | 0.7500 | 0.2000 | 1.0000 |
| `confidence` | 0.0400 | 0.9600 | 0.0000 | 0.0000 |
| `hybrid` | 0.3200 | 0.6800 | 0.1562 | 1.0000 |
| `confidence_095` | 0.7100 | 0.2900 | 0.0282 | 0.4000 |
| `hybrid_relaxed` | 0.7600 | 0.2400 | 0.0658 | 1.0000 |
| `claim_divergence` | 0.4500 | 0.5500 | 0.1111 | 1.0000 |
| `voc_v2` | 0.5900 | 0.4100 | 0.0847 | 1.0000 |

## 7. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00021`
- 问题预览：Raymond and Samantha are cousins. Raymond was born 6 years before Samantha. Raymond had a son at the age of 23. If Sa...
- 金标：`14`
- `mv_3`：`8` / score=`0.0`
- `always`：`14` / score=`1.0`
- 说明：always_communicate 能纠错，但 voc_trigger_v2 在该题 early exit，漏掉了有益通信。

### Case 2

- 数据集：`gsm8k`
- 样本：`gsm8k-00057`
- 问题预览：A wooden bridge can carry no more than 5000 pounds. A delivery truck filled with identical boxes, each weighing 15 po...
- 金标：`83`
- `mv_3`：`83` / score=`1.0`
- `always`：`83` / score=`1.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-00132`
- 问题预览：Pam and Fred went to a carnival. Pam rode the roller coaster 2 times while Fred rode it 4 times. After that, each of ...
- 金标：`60`
- `mv_3`：`48` / score=`0.0`
- `always`：`60` / score=`1.0`
- 说明：always_communicate 能纠错，但 voc_trigger_v2 在该题 early exit，漏掉了有益通信。

### Case 4

- 数据集：`gsm8k`
- 样本：`gsm8k-00143`
- 问题预览：John buys milk for 2 dollars, eggs for 3 dollars, light bulbs for 3 dollars, cups for 3 dollars, and roach traps for ...
- 金标：`16`
- `mv_3`：`18.4` / score=`0.0`
- `always`：`18.4` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

### Case 5

- 数据集：`gsm8k`
- 样本：`gsm8k-00177`
- 问题预览：Zaid spends 1/4 of his salary on rent, 1/3 on car fuel and donates half of the remaining amount to his favorite chari...
- 金标：`350`
- `mv_3`：`1800` / score=`0.0`
- `always`：`600` / score=`0.0`
- 说明：通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。

## 8. 下一轮默认 trigger 建议

- 推荐策略：`voc_trigger_v2`
- 相对 `always_communicate` 的准确率下降：`0.01`
- 相对 `always_communicate` 的总 token 下降比例：`0.177675`
- 规则是否通过：`True`

## 9. 局限

- 本轮只覆盖 `smoke20`，因此只报告描述性结果，不做统计显著性结论。
- 当前只比较 trigger / early-exit，不比较消息内容压缩和局部审计。
- `mv_3` 直接复用共享 Stage A，因此它是本实验内部的无通信基线，不代表独立运行时的物理网络成本。

