# 多智能体 Debate vs Vote 科研报告

## 摘要

- 配对增益最大的分组是 `gsm8k` / `mad_3a_r1`，debate 相对 vote 的准确率差值为 +0.3500。
- 该实验严格共享同一批初始候选答案，因此所有差异都可解释为 debate 环节本身带来的净收益或净伤害。

## 实验概览

- 实验名：`same_context_controlled_debate`
- Phase：`smoke20`
- Prompt Version：`multi_agent_controlled_json`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- 本实验比较同一批初始候选答案上的两种聚合方式：直接投票的 `initial vote`，以及单轮 debate 后再投票的 `debate vote`。
- 主指标为配对准确率差值；机制指标包括 corrected / harmed、consensus 变化、flip rate，以及 debate 增量 token / 时延。
- 因为是严格配对设计，所以准确率差值、McNemar 检验和 bootstrap 区间都可以直接解释 debate 的净贡献。

## 分组结果总表

| 数据集 | 方法 | 题量 | initial vote 准确率 | debate vote 准确率 | 准确率差值 | 纠正题数 | 伤害题数 | 每千 debate token 增益 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `gsm8k` | `mad_3a_r1` | 20 | 0.6000 | 0.9500 | +0.3500 | 7 | 0 | +0.208607 |
| `hotpotqa` | `mad_3a_r1` | 20 | 0.7500 | 0.8000 | +0.0500 | 1 | 0 | +0.008931 |
| `strategyqa` | `mad_3a_r1` | 20 | 0.7000 | 0.7000 | +0.0000 | 0 | 0 | +0.000000 |

## 配对统计与机制诊断

### gsm8k / mad_3a_r1

| 指标 | 数值 |
| --- | --- |
| 初始一致率 | 0.4500 |
| debate 后一致率 | 0.8000 |
| 翻票率 | 0.3500 |
| 增量 token / 题 | 1677.80 |
| 增量时延 / 题 (ms) | 6548.27 |
| McNemar exact p | None |
| Bootstrap 95% CI | statistics are only reported for pilot100 |

### hotpotqa / mad_3a_r1

| 指标 | 数值 |
| --- | --- |
| 初始一致率 | 0.6000 |
| debate 后一致率 | 0.8500 |
| 翻票率 | 0.0500 |
| 增量 token / 题 | 5598.25 |
| 增量时延 / 题 (ms) | 9687.09 |
| McNemar exact p | None |
| Bootstrap 95% CI | statistics are only reported for pilot100 |

### strategyqa / mad_3a_r1

| 指标 | 数值 |
| --- | --- |
| 初始一致率 | 0.8000 |
| debate 后一致率 | 0.9500 |
| 翻票率 | 0.0000 |
| 增量 token / 题 | 1176.10 |
| 增量时延 / 题 (ms) | 5168.81 |
| McNemar exact p | None |
| Bootstrap 95% CI | statistics are only reported for pilot100 |

## 结论与建议

- 如果某个分组的 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。
- 如果 corrected 明显多于 harmed，且 bootstrap 区间稳定高于 0，则说明 debate 在该分组具备真实增益。
- 进入更大样本 phase 前，应优先挑选配对增益稳定为正的设置继续扩展，而不是默认所有 debate 都值得保留。

## 局限性

- `smoke20` 更适合联调与方向观察，不作为显著性结论来源；正式结论仍应依赖 `pilot100` 及以上 phase。
- 若某个数据集存在明显噪声或较强题型异质性，应单独解释其 debate 收益，而不是简单并入总体叙述。

## 复现与产物说明

- 运行目录：`runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`paired_debate_vs_vote.json`、`metrics.json`、`report.md`、`figure_manifest.json`、`figures/`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核与引用。

## 图表资产

### 多智能体成本-性能前沿

![多智能体成本-性能前沿](../../runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，多智能体方法及其对照的准确率相对于平均总 token 的位置关系。*

### 多智能体效率排序

![多智能体效率排序](../../runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### 多智能体跨数据集表现

![多智能体跨数据集表现](../../runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各多智能体方法及其对照在不同数据集上的准确率分布。*

### Debate 增益分解

![Debate 增益分解](../../runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5/figures/debate_delta_breakdown.svg)

*按分组展示纠正率、伤害率和净准确率差值。*

### 配对效应置信区间

![配对效应置信区间](../../runs/multi_agent/same_context_controlled_debate/smoke20/20260509T125836Z-same_context_controlled_debate-smoke20-xiaomimimo-mimo-v2.5/figures/paired_effect_ci.svg)

*debate 相对 initial vote 的配对 bootstrap 置信区间。*
