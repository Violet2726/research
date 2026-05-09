# SID-lite 科研报告

## 摘要

- 总体准确率最高的方法是 `always_full`，准确率为 0.7833。
- 总体效率最高的方法是 `mv_3`，每千 token 准确率为 0.281679。
- `sid_lite` 相对 `always_full` 的总体准确率差异 bootstrap 95% CI 为 `[-0.166667, 0.000000]`。

## 实验概览

- 实验名：`sid_lite_mechanism_validation`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- SID-lite 关注在黑盒条件下，能否用自报置信度和结构化语义字段近似 self-signals，从而在不读 logits 的情况下实现选择性通信。
- 主指标为准确率；成本指标采用平均通信 token / 题与平均总 token / 题；机制指标重点是早退率、压缩比和 fail-open 计数。
- 本实验固定比较 `mv_3`、`always_full`、`compression_only` 与 `sid_lite`，因此可以直接分离“只压缩”和“有门控”的差异。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 | 早退率 | 压缩比 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 2485.10 | 3.00 | 0.281679 | 1.0000 | 0.4369 |
| `always_full` | 0.7833 | 773.47 | 6109.07 | 6.00 | 0.128225 | 0.0000 | 0.4369 |
| `compression_only` | 0.7000 | 327.13 | 5372.57 | 6.00 | 0.130292 | 0.0000 | 0.4369 |
| `sid_lite` | 0.7000 | 132.73 | 3834.32 | 4.25 | 0.182562 | 0.5833 | 0.4369 |

## 机制诊断

- SID 早退率：`0.583333`。
- 非法 confidence fail-open 计数：`2`。
- 如果 `sid_lite` 的早退率很高，但准确率与 `always_full` 接近，说明门控近似已经具备明显工程价值。
- 如果 fail-open 计数偏高，则应优先改进置信度提取和结构化恢复逻辑，而不是继续压缩通信预算。

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 1177.25 | 0.594606 |
| `always_full` | 0.9000 | 597.00 | 3441.25 | 0.261533 |
| `compression_only` | 0.7000 | 280.10 | 2803.45 | 0.249692 |
| `sid_lite` | 0.7000 | 144.60 | 2147.95 | 0.325892 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 5337.95 | 0.131136 |
| `always_full` | 0.7500 | 993.80 | 11969.50 | 0.062659 |
| `compression_only` | 0.7000 | 377.90 | 11015.00 | 0.063550 |
| `sid_lite` | 0.7000 | 174.10 | 8057.90 | 0.086871 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3` | 0.7000 | 0.00 | 940.10 | 0.744602 |
| `always_full` | 0.7000 | 729.60 | 2916.45 | 0.240018 |
| `compression_only` | 0.7000 | 323.40 | 2299.25 | 0.304447 |
| `sid_lite` | 0.7000 | 79.50 | 1297.10 | 0.539665 |

## 结论与建议

- 如果 `sid_lite` 能在明显降低通信成本时维持与 `always_full` 接近的准确率，则后续优先值得扩大样本确认。
- 若 `compression_only` 已经贡献了大部分成本收益，而 `sid_lite` 仅带来有限额外收益，应更谨慎评估门控复杂度是否值得。
- 正式进入更大规模 phase 前，应联合考察前沿图、早退率图和 fail-open 计数，避免只凭总体准确率判断。

## 局限性

- SID-lite 不读取真实 token logits 或 attention，因此并不是对完整 SID 的严格复现，而是黑盒近似版本。
- 当前报告只反映当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。

## 复现与产物说明

- 运行目录：`runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。
- 本地报告与发布报告共享同一套 run 内图资产，便于后续复核和引用。

## 图表资产

### SID-lite 成本-性能前沿

![SID-lite 成本-性能前沿](../../runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各 SID-lite 变体的准确率相对于平均总 token 的位置关系。*

### SID-lite 效率排序

![SID-lite 效率排序](../../runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### SID-lite 跨数据集表现

![SID-lite 跨数据集表现](../../runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各 SID-lite 变体在不同数据集上的准确率分布。*

### SID 门控权衡

![SID 门控权衡](../../runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/sid_gate_tradeoff.svg)

*总体早退率相对于准确率的变化。*

### 置信度失效 fail-open 计数

![置信度失效 fail-open 计数](../../runs/sid_lite/sid_lite_mechanism_validation/smoke20/20260509T125841Z-sid_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/invalid_confidence_fail_open.svg)

*因置信度信号无效而触发 fail-open 的样本计数。*
