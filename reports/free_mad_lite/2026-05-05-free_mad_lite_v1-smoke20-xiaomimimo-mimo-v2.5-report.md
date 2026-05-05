# Free-MAD-lite Smoke20 报告

## 1. 实验概览

- 实验名：`free_mad_lite_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/free_mad_lite/free_mad_lite_v1/smoke20/20260505T050228Z-free_mad_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法：`mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote`、`free_mad_lite_llm_trajectory`。
- 说明：本实验只验证 single-round anti-conformity 与 LLM trajectory judge，不复现完整 Free-MAD score model 或攻击鲁棒性实验。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Judge Fallback | Changed Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6333 | 0.00 | 2132.80 | 3.00 | 0.296949 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7833 | 361.10 | 4938.27 | 6.00 | 0.158625 | 0.0000 | 0.0722 |
| `anti_conformity_final_vote` | 0.7167 | 361.10 | 5088.82 | 6.00 | 0.140832 | 0.0000 | 0.1333 |
| `free_mad_lite_llm_trajectory` | 0.7667 | 361.10 | 6346.37 | 7.00 | 0.120804 | 0.0167 | 0.1333 |

## 3. 机制诊断

- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的 overall accuracy delta 95% bootstrap CI：[-0.100000, 0.066667]（smoke20 小样本，仅作方向性参考）。
- Judge fallback rate：`0.016667`
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.5000 | 0.00 | 894.05 | 0.559253 |
| `vanilla_mad_r1_final_vote` | 0.9000 | 346.50 | 2591.45 | 0.347296 |
| `anti_conformity_final_vote` | 0.9500 | 346.50 | 2758.55 | 0.344384 |
| `free_mad_lite_llm_trajectory` | 0.9000 | 346.50 | 3671.85 | 0.245108 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.7000 | 0.00 | 4898.65 | 0.142897 |
| `vanilla_mad_r1_final_vote` | 0.6500 | 420.20 | 10463.50 | 0.062121 |
| `anti_conformity_final_vote` | 0.6500 | 420.20 | 10578.55 | 0.061445 |
| `free_mad_lite_llm_trajectory` | 0.7000 | 420.20 | 12732.95 | 0.054975 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.7000 | 0.00 | 605.70 | 1.155688 |
| `vanilla_mad_r1_final_vote` | 0.8000 | 316.60 | 1759.85 | 0.454584 |
| `anti_conformity_final_vote` | 0.5500 | 316.60 | 1929.35 | 0.285070 |
| `free_mad_lite_llm_trajectory` | 0.7000 | 316.60 | 2634.30 | 0.265725 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- Free-MAD-lite 使用 LLM judge 近似轨迹裁决，不包含论文完整 score-based decision 训练或攻击场景。
