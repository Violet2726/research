# SID-lite Smoke20 报告

## 1. 实验概览

- 实验名：`sid_lite_v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/sid_lite/sid_lite_v1/smoke20/20260424T061208Z-sid_lite_v1-smoke20-dashscope-qwen-turbo-1101`
- 方法：`mv_3`、`always_full`、`compression_only`、`sid_lite`。
- 说明：本实验是黑盒 SID-lite 近似，DashScope Chat API 不暴露 logits/attention，因此用自报置信度和结构化语义字段近似 self signals。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6167 | 0.00 | 2271.30 | 3.00 | 0.271504 | 1.0000 | 0.5061 |
| `always_full` | 0.6000 | 445.40 | 5219.10 | 6.00 | 0.114962 | 0.0000 | 0.5061 |
| `compression_only` | 0.6500 | 220.03 | 4856.90 | 6.00 | 0.133830 | 0.0000 | 0.5061 |
| `sid_lite` | 0.6500 | 76.77 | 3077.87 | 4.05 | 0.211185 | 0.6500 | 0.5061 |

## 3. 机制诊断

- `sid_lite` 相对 `always_full` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.116667]（smoke20 小样本，仅作方向性参考）。
- SID 早退率：`0.65`
- 非法 confidence fail-open 数：`1`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 958.70 | 0.625848 |
| `always_full` | 0.6000 | 415.60 | 2617.90 | 0.229191 |
| `compression_only` | 0.7000 | 214.60 | 2234.80 | 0.313227 |
| `sid_lite` | 0.7000 | 117.80 | 1690.25 | 0.414140 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 5091.50 | 0.127664 |
| `always_full` | 0.6000 | 556.10 | 10960.75 | 0.054741 |
| `compression_only` | 0.6500 | 242.30 | 10466.50 | 0.062103 |
| `sid_lite` | 0.6500 | 59.10 | 6493.30 | 0.100103 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 763.70 | 0.785649 |
| `always_full` | 0.6000 | 364.50 | 2078.65 | 0.288649 |
| `compression_only` | 0.6000 | 203.20 | 1869.40 | 0.320959 |
| `sid_lite` | 0.6000 | 53.40 | 1050.05 | 0.571401 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- SID-lite 不读取真实 token logits 或 attention maps，因此不是 full SID reproduction。
