# SID-lite 科研报告

## 摘要

- 总体准确率最高的方法是 `always_full`，准确率为 0.7033。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.246397。
- `sid_lite` 相对 `always_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.113333, -0.050000]`。

## 实验概览

- 实验名：`sid_lite_mechanism_validation`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- SID-lite 关注在黑盒条件下，能否用自报置信度和结构化语义字段近似 self-signals，从而在不读 logits 的情况下实现选择性通信。
- 主指标为准确率；成本指标采用平均通信 token / 题与平均总 token / 题；机制指标重点是早退率、压缩比和 fail-open 计数。
- 本实验固定比较 `mv_3`、`always_full`、`compression_only` 与 `sid_lite`，因此可以直接分离“只压缩”和“有门控”的差异。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 | 早退率 | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.6133 | 0.00 | 2489.21 | 3.00 | 0.246397 | 1.0000 | 0.4331 |
| `always_full` | 0.7033 | 782.13 | 6121.90 | 6.00 | 0.114888 | 0.0000 | 0.4331 |
| `compression_only` | 0.6233 | 327.93 | 5378.86 | 6.00 | 0.115886 | 0.0000 | 0.4331 |
| `sid_lite` | 0.6233 | 118.66 | 3603.72 | 4.19 | 0.172970 | 0.6033 | 0.4331 |

## 机制诊断

- SID 早退率：`0.603333`。
- 非法 confidence fail-open 计数：`8`。
- 如果 `sid_lite` 的早退率很高，但准确率与 `always_full` 接近，说明门控近似已经具备明显工程价值。
- 如果 fail-open 计数偏高，则应优先改进置信度提取和结构化恢复逻辑，而不是继续压缩通信预算。

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.5200 | 0.00 | 1242.07 | 0.418656 |
| `always_full` | 0.7500 | 613.22 | 3610.27 | 0.207741 |
| `compression_only` | 0.5400 | 270.42 | 2889.99 | 0.186852 |
| `sid_lite` | 0.5400 | 143.68 | 2283.66 | 0.236463 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6600 | 0.00 | 5282.01 | 0.124952 |
| `always_full` | 0.6900 | 978.66 | 11823.50 | 0.058358 |
| `compression_only` | 0.6700 | 384.24 | 10938.90 | 0.061249 |
| `sid_lite` | 0.6700 | 132.12 | 7230.32 | 0.092665 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.6600 | 0.00 | 943.55 | 0.699486 |
| `always_full` | 0.6700 | 754.52 | 2931.94 | 0.228518 |
| `compression_only` | 0.6600 | 329.14 | 2307.70 | 0.285999 |
| `sid_lite` | 0.6600 | 80.18 | 1297.17 | 0.508800 |

## 结论与建议

- 如果 `sid_lite` 能在明显降低通信成本时维持与 `always_full` 接近的准确率，则后续优先值得扩大样本确认。
- 若 `compression_only` 已经贡献了大部分成本收益，而 `sid_lite` 仅带来有限额外收益，应更谨慎评估门控复杂度是否值得。
- 正式进入更大规模 phase 前，应联合考察前沿图、早退率图和 fail-open 计数，避免只凭总体准确率判断。

## 局限性

- SID-lite 不读取真实 token logits 或 attention，因此并不是对完整 SID 的严格复现，而是黑盒近似版本。
- 当前报告只反映当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。

## 复现与产物说明

- 运行目录：`runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。
- 本地报告与发布报告共享同一套 run 内图资产，便于后续复核和引用。

## 图表资产

### SID-lite 成本-性能前沿

![SID-lite 成本-性能前沿](../../runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各 SID-lite 变体的准确率相对于平均总 token 的位置关系。*

### SID-lite 效率排序

![SID-lite 效率排序](../../runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### SID-lite 跨数据集表现

![SID-lite 跨数据集表现](../../runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各 SID-lite 变体在不同数据集上的准确率分布。*

### SID 门控权衡

![SID 门控权衡](../../runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5/figures/sid_gate_tradeoff.svg)

*总体早退率相对于准确率的变化。*

### 置信度失效 fail-open 计数

![置信度失效 fail-open 计数](../../runs/sid_lite/sid_lite_mechanism_validation/pilot100/20260509T125911Z-sid_lite_mechanism_validation-pilot100-xiaomimimo-mimo-v2.5/figures/invalid_confidence_fail_open.svg)

*因置信度信号无效而触发 fail-open 的样本计数。*
