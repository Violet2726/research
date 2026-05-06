# Free-MAD-lite Smoke20 报告

## 1. 实验概览

- 实验名：`free_mad_lite_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/free_mad_lite/free_mad_lite_v1/pilot100/20260506T140642Z-free_mad_lite_v1-pilot100-xiaomimimo-mimo-v2.5`
- 方法：`mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote`、`free_mad_lite_llm_trajectory`。
- 说明：本实验只验证 single-round anti-conformity 与 LLM trajectory judge，不复现完整 Free-MAD score model 或攻击鲁棒性实验。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Judge Fallback | Changed Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6267 | 0.00 | 2130.69 | 3.00 | 0.294114 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7533 | 357.21 | 4929.66 | 6.00 | 0.152817 | 0.0000 | 0.0800 |
| `anti_conformity_final_vote` | 0.7067 | 357.21 | 5080.65 | 6.00 | 0.139090 | 0.0000 | 0.1244 |
| `free_mad_lite_llm_trajectory` | 0.7867 | 357.21 | 6347.07 | 7.00 | 0.123942 | 0.0033 | 0.1244 |

## 3. 机制诊断

- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的 overall accuracy delta 95% bootstrap CI：[0.003333, 0.063333]（smoke20 小样本，仅作方向性参考）。
- Judge fallback rate：`0.003333`
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.5300 | 0.00 | 941.30 | 0.563051 |
| `vanilla_mad_r1_final_vote` | 0.8200 | 308.36 | 2644.32 | 0.310099 |
| `anti_conformity_final_vote` | 0.7800 | 308.36 | 2807.89 | 0.277789 |
| `free_mad_lite_llm_trajectory` | 0.9100 | 308.36 | 3739.77 | 0.243330 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6400 | 0.00 | 4838.01 | 0.132286 |
| `vanilla_mad_r1_final_vote` | 0.6900 | 424.40 | 10350.73 | 0.066662 |
| `anti_conformity_final_vote` | 0.6600 | 424.40 | 10480.57 | 0.062974 |
| `free_mad_lite_llm_trajectory` | 0.7100 | 424.40 | 12620.81 | 0.056256 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.7100 | 0.00 | 612.76 | 1.158692 |
| `vanilla_mad_r1_final_vote` | 0.7500 | 338.88 | 1793.92 | 0.418079 |
| `anti_conformity_final_vote` | 0.6800 | 338.88 | 1953.48 | 0.348097 |
| `free_mad_lite_llm_trajectory` | 0.7400 | 338.88 | 2680.64 | 0.276053 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- Free-MAD-lite 使用 LLM judge 近似轨迹裁决，不包含论文完整 score-based decision 训练或攻击场景。
