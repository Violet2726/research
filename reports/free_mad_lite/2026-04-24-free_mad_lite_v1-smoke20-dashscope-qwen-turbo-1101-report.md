# Free-MAD-lite Smoke20 报告

## 1. 实验概览

- 实验名：`free_mad_lite_v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/free_mad_lite/free_mad_lite_v1/smoke20/20260424T063104Z-free_mad_lite_v1-smoke20-dashscope-qwen-turbo-1101`
- 方法：`mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote`、`free_mad_lite_llm_trajectory`。
- 说明：本实验只验证 single-round anti-conformity 与 LLM trajectory judge，不复现完整 Free-MAD score model 或攻击鲁棒性实验。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Judge Fallback | Changed Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6167 | 0.00 | 2034.03 | 3.00 | 0.303174 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7667 | 224.40 | 4537.10 | 6.00 | 0.168977 | 0.0000 | 0.0500 |
| `anti_conformity_final_vote` | 0.6833 | 224.40 | 4627.93 | 6.00 | 0.147654 | 0.0000 | 0.0167 |
| `free_mad_lite_llm_trajectory` | 0.7500 | 224.40 | 5625.85 | 7.00 | 0.133313 | 0.0000 | 0.0167 |

## 3. 机制诊断

- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的 overall accuracy delta 95% bootstrap CI：[-0.083333, 0.033333]（smoke20 小样本，仅作方向性参考）。
- Judge fallback rate：`0.0`
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.5500 | 0.00 | 766.20 | 0.717828 |
| `vanilla_mad_r1_final_vote` | 1.0000 | 242.90 | 2173.35 | 0.460119 |
| `anti_conformity_final_vote` | 0.8500 | 242.90 | 2263.95 | 0.375450 |
| `free_mad_lite_llm_trajectory` | 1.0000 | 242.90 | 2942.35 | 0.339864 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6500 | 0.00 | 4788.85 | 0.135732 |
| `vanilla_mad_r1_final_vote` | 0.6500 | 232.60 | 10000.60 | 0.064996 |
| `anti_conformity_final_vote` | 0.6500 | 232.60 | 10080.75 | 0.064479 |
| `free_mad_lite_llm_trajectory` | 0.6500 | 232.60 | 11960.85 | 0.054344 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6500 | 0.00 | 547.05 | 1.188191 |
| `vanilla_mad_r1_final_vote` | 0.6500 | 197.70 | 1437.35 | 0.452221 |
| `anti_conformity_final_vote` | 0.5500 | 197.70 | 1539.10 | 0.357352 |
| `free_mad_lite_llm_trajectory` | 0.6000 | 197.70 | 1974.35 | 0.303897 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- Free-MAD-lite 使用 LLM judge 近似轨迹裁决，不包含论文完整 score-based decision 训练或攻击场景。
