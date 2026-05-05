# SID-lite Smoke20 报告

## 1. 实验概览

- 实验名：`sid_lite_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sid_lite/sid_lite_v1/smoke20/20260505T050232Z-sid_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法：`mv_3`、`always_full`、`compression_only`、`sid_lite`。
- 说明：本实验是黑盒 SID-lite 近似，DashScope Chat API 不暴露 logits/attention，因此用自报置信度和结构化语义字段近似 self signals。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6667 | 0.00 | 2479.93 | 3.00 | 0.268824 | 1.0000 | 0.4372 |
| `always_full` | 0.7667 | 764.53 | 6098.92 | 6.00 | 0.125705 | 0.0000 | 0.4372 |
| `compression_only` | 0.7000 | 323.50 | 5339.60 | 6.00 | 0.131096 | 0.0000 | 0.4372 |
| `sid_lite` | 0.7000 | 122.03 | 3575.55 | 4.20 | 0.195774 | 0.6000 | 0.4372 |

## 3. 机制诊断

- `sid_lite` 相对 `always_full` 的 overall accuracy delta 95% bootstrap CI：[-0.150000, 0.033333]（smoke20 小样本，仅作方向性参考）。
- SID 早退率：`0.6`
- 非法 confidence fail-open 数：`3`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5500 | 0.00 | 1182.40 | 0.465156 |
| `always_full` | 0.8000 | 610.80 | 3501.15 | 0.228496 |
| `compression_only` | 0.6000 | 279.10 | 2741.20 | 0.218882 |
| `sid_lite` | 0.6000 | 147.90 | 2102.80 | 0.285334 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5306.70 | 0.141331 |
| `always_full` | 0.7500 | 937.90 | 11841.95 | 0.063334 |
| `compression_only` | 0.8000 | 365.30 | 10970.45 | 0.072923 |
| `sid_lite` | 0.8000 | 117.30 | 7245.55 | 0.110413 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7000 | 0.00 | 950.70 | 0.736300 |
| `always_full` | 0.7500 | 744.90 | 2953.65 | 0.253923 |
| `compression_only` | 0.7000 | 326.10 | 2307.15 | 0.303405 |
| `sid_lite` | 0.7000 | 100.90 | 1378.30 | 0.507872 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- SID-lite 不读取真实 token logits 或 attention maps，因此不是 full SID reproduction。
