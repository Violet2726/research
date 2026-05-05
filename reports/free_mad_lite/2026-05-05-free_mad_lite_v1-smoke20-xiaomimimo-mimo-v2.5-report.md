# Free-MAD-lite Smoke20 报告

## 1. 实验概览

- 实验名：`free_mad_lite_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/free_mad_lite/free_mad_lite_v1/smoke20/20260505T084629Z-free_mad_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法：`mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote`、`free_mad_lite_llm_trajectory`。
- 说明：本实验只验证 single-round anti-conformity 与 LLM trajectory judge，不复现完整 Free-MAD score model 或攻击鲁棒性实验。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Judge Fallback | Changed Answer |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.7167 | 0.00 | 2121.58 | 3.00 | 0.337798 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7833 | 337.90 | 4881.50 | 6.00 | 0.160470 | 0.0000 | 0.0722 |
| `anti_conformity_final_vote` | 0.7000 | 337.90 | 5036.27 | 6.00 | 0.138992 | 0.0000 | 0.1556 |
| `free_mad_lite_llm_trajectory` | 0.7833 | 337.90 | 6289.42 | 7.00 | 0.124548 | 0.0167 | 0.1556 |

## 3. 机制诊断

- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的 overall accuracy delta 95% bootstrap CI：[-0.083333, 0.083333]（smoke20 小样本，仅作方向性参考）。
- Judge fallback rate：`0.016667`
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.6500 | 0.00 | 884.85 | 0.734588 |
| `vanilla_mad_r1_final_vote` | 0.8500 | 289.60 | 2474.90 | 0.343448 |
| `anti_conformity_final_vote` | 0.9500 | 289.60 | 2629.30 | 0.361313 |
| `free_mad_lite_llm_trajectory` | 1.0000 | 289.60 | 3523.20 | 0.283833 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.7000 | 0.00 | 4867.80 | 0.143802 |
| `vanilla_mad_r1_final_vote` | 0.7000 | 400.00 | 10384.60 | 0.067408 |
| `anti_conformity_final_vote` | 0.6500 | 400.00 | 10515.30 | 0.061815 |
| `free_mad_lite_llm_trajectory` | 0.6500 | 400.00 | 12652.80 | 0.051372 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3_initial` | 0.8000 | 0.00 | 612.10 | 1.306976 |
| `vanilla_mad_r1_final_vote` | 0.8000 | 324.10 | 1785.00 | 0.448179 |
| `anti_conformity_final_vote` | 0.5000 | 324.10 | 1964.20 | 0.254557 |
| `free_mad_lite_llm_trajectory` | 0.7000 | 324.10 | 2692.25 | 0.260006 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- Free-MAD-lite 使用 LLM judge 近似轨迹裁决，不包含论文完整 score-based decision 训练或攻击场景。
