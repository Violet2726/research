# SID-lite Smoke20 报告

## 1. 实验概览

- 实验名：`sid_lite_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sid_lite/sid_lite_v1/pilot100/20260507T121711Z-sid_lite_v1-pilot100-xiaomimimo-mimo-v2.5`
- 方法：`mv_3`、`always_full`、`compression_only`、`sid_lite`。
- 说明：本实验是黑盒 SID-lite 近似，DashScope Chat API 不暴露 logits/attention，因此用自报置信度和结构化语义字段近似 self signals。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit | Compression |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5933 | 0.00 | 2491.86 | 3.00 | 0.238109 | 1.0000 | 0.4263 |
| `always_full` | 0.7133 | 778.81 | 6119.19 | 6.00 | 0.116573 | 0.0000 | 0.4263 |
| `compression_only` | 0.6200 | 325.26 | 5375.17 | 6.00 | 0.115345 | 0.0000 | 0.4263 |
| `sid_lite` | 0.6200 | 106.87 | 3549.78 | 4.11 | 0.174659 | 0.6300 | 0.4263 |

## 3. 机制诊断

- `sid_lite` 相对 `always_full` 的 overall accuracy delta 95% bootstrap CI：[-0.136667, -0.053333]（smoke20 小样本，仅作方向性参考）。
- SID 早退率：`0.63`
- 非法 confidence fail-open 数：`12`

## 4. 数据集分表

### gsm8k

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4600 | 0.00 | 1256.83 | 0.366000 |
| `always_full` | 0.7800 | 612.72 | 3618.90 | 0.215535 |
| `compression_only` | 0.5100 | 264.68 | 2891.10 | 0.176403 |
| `sid_lite` | 0.5100 | 123.46 | 2219.26 | 0.229806 |

### hotpotqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6700 | 0.00 | 5271.08 | 0.127109 |
| `always_full` | 0.6800 | 965.22 | 11785.66 | 0.057697 |
| `compression_only` | 0.6800 | 378.78 | 10915.32 | 0.062298 |
| `sid_lite` | 0.6800 | 116.72 | 7135.45 | 0.095299 |

### strategyqa

| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 947.67 | 0.685893 |
| `always_full` | 0.6800 | 758.50 | 2953.00 | 0.230274 |
| `compression_only` | 0.6700 | 332.32 | 2319.10 | 0.288905 |
| `sid_lite` | 0.6700 | 80.42 | 1294.62 | 0.517526 |

## 5. 局限

- 当前只运行 smoke20，不能作为最终显著性结论。
- SID-lite 不读取真实 token logits 或 attention maps，因此不是 full SID reproduction。
