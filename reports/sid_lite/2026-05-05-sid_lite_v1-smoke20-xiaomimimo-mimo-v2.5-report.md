# SID-lite Smoke20 报告

## 1. 实验概览

- 实验名：`sid_lite_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sid_lite/sid_lite_v1/smoke20/20260505T090800Z-sid_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- 方法：`mv_3`、`always_full`、`compression_only`、`sid_lite`。
- 说明：本实验是黑盒 SID-lite 近似，DashScope Chat API 不暴露 logits/attention，因此用自报置信度和结构化语义字段近似 self signals。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 2501.33 | 3.00 | 0.239872 | 1.0000 | 0.4266 |
| `always_full` | 0.7833 | 788.13 | 6166.45 | 6.00 | 0.127031 | 0.0000 | 0.4266 |
| `compression_only` | 0.6833 | 323.57 | 5403.25 | 6.00 | 0.126467 | 0.0000 | 0.4266 |
| `sid_lite` | 0.6833 | 136.37 | 3873.88 | 4.35 | 0.176395 | 0.5500 | 0.4266 |

## 3. 机制诊断

- `sid_lite` 相对 `always_full` 的 overall accuracy delta 95% bootstrap CI：[-0.216667, 0.000000]（smoke20 小样本，仅作方向性参考）。
- SID 早退率：`0.55`
- 非法 confidence fail-open 数：`2`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4500 | 0.00 | 1208.50 | 0.372362 |
| `always_full` | 0.9000 | 615.70 | 3538.25 | 0.254363 |
| `compression_only` | 0.5000 | 270.70 | 2830.30 | 0.176660 |
| `sid_lite` | 0.5000 | 132.50 | 2171.65 | 0.230240 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.7500 | 0.00 | 5331.85 | 0.140664 |
| `always_full` | 0.8500 | 985.80 | 11928.45 | 0.071258 |
| `compression_only` | 0.8000 | 371.80 | 11027.65 | 0.072545 |
| `sid_lite` | 0.8000 | 147.30 | 7903.65 | 0.101219 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 963.65 | 0.622633 |
| `always_full` | 0.6000 | 762.90 | 3032.65 | 0.197847 |
| `compression_only` | 0.7500 | 328.20 | 2351.80 | 0.318905 |
| `sid_lite` | 0.7500 | 129.30 | 1546.35 | 0.485013 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- SID-lite 不读取真实 token logits 或 attention maps，因此不是 full SID reproduction。
