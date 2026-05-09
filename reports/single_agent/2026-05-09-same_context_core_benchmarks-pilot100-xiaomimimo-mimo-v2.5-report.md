# 单智能体科研报告

## 实验概览

- 实验名：`same_context_core_benchmarks`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt Version：`single_agent_reasoning_json_v1`
- 运行目录：`runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5`

## 研究问题与判读口径

- 本实验把单智能体方法作为无通信比较锚点，核心问题是不同调用预算是否带来稳定的性能收益。
- 主指标为准确率；成本指标采用平均总 token / 题；效率指标采用每千 token 准确率。
- 对于自洽类方法，还额外关注预算扩张是否带来边际收益递减，以及 rerun 波动是否足够可控。

## 总体结果

表 1 汇总总体层面的准确率、成本和效率，便于直接选取强基线与等预算对照。

| 方法 | 准确率 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 | 准确率标准差 |
| --- | --- | --- | --- | --- | --- |
|  |  |  |  |  |  |

## 预算与稳定性分析

- 若某个自洽方法的准确率提升不再明显，而 token 成本持续上升，应优先把它作为等预算对照而非默认主方法。

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `cot_1` | 0.9600 | 258.17 | 1.00 | 3.718480 |
| `sc_5` | 0.9600 | 1288.42 | 5.00 | 0.745099 |

### hotpotqa

| 方法 | 准确率 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `sc_5` | 0.7300 | 7959.48 | 5.00 | 0.091715 |
| `cot_1` | 0.7100 | 1592.21 | 1.00 | 0.445921 |

### strategyqa

| 方法 | 准确率 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `cot_1` | 0.8300 | 177.63 | 1.00 | 4.672634 |
| `sc_5` | 0.7300 | 887.57 | 5.00 | 0.822470 |

## 结论与建议

- 如果目标是构造强无通信基线，应同时记录“总体准确率最佳”和“单位成本效率最佳”两条结论，避免只看单一排序。
- 若后续通信方法需要做等预算对照，应优先选取 token 规模最接近的 `cot_1` 或 `sc_k` 配置，而不是仅按准确率最高的方法对比。
- 正式进入更大规模 phase 前，建议优先观察自洽类方法的收益是否已经饱和，以控制整体请求成本。

## 局限性

- 本报告只反映当前 phase 的统计汇总，不直接等同于更大样本下的最终结论。
- 单智能体报告不包含通信行为，因此无法解释多智能体收益来源，只能作为下游比较锚点。

## 复现与产物说明

- 运行目录：`runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`run_summary.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_tables.md`。
- 图表与表格共享同一份 run 内数据源，便于后续复核和重渲染。

## 图表资产

### 单智能体成本-性能前沿

![单智能体成本-性能前沿](../../runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，准确率相对于平均总 token 的位置关系。*

### 单智能体效率排序

![单智能体效率排序](../../runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 单智能体跨数据集表现

![单智能体跨数据集表现](../../runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各方法在不同数据集上的准确率分布。*

### 自洽采样规模效应

![自洽采样规模效应](../../runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5/figures/self_consistency_scaling.svg)

*自洽方法在调用预算增加时的准确率变化。*

### 方法预算构成

![方法预算构成](../../runs/single_agent/same_context_core_benchmarks/pilot100/20260509T125858Z-same_context_core_benchmarks-pilot100-xiaomimimo-mimo-v2.5/figures/method_budget_profile.svg)

*总体结果上，各方法的 prompt 和 completion token 构成。*
