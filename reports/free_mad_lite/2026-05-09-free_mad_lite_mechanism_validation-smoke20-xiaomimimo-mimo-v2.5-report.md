# Free-MAD-lite 科研报告

## 摘要

- 总体准确率最高的方法是 `free_mad_lite_llm_trajectory`，准确率为 0.8167。
- 总体效率最高的方法是 `mv_3_initial`，每千 token 准确率为 0.313657。
- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的总体准确率差异 bootstrap 95% CI 为 `[-0.050000, 0.083333]`。

## 实验概览

- 实验名：`free_mad_lite_mechanism_validation`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- Free-MAD-lite 关注单轮 anti-conformity 与 LLM trajectory judge 是否足以带来稳定收益，而不复现完整 score-model 训练流程。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；机制指标重点是 changed answer、corrected、harmed 与 judge fallback rate。
- 本实验固定比较 `mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote` 和 `free_mad_lite_llm_trajectory`，因此可以隔离轨迹裁决环节的贡献。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 | Judge fallback rate | Changed answer rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.6667 | 0.00 | 2125.47 | 3.00 | 0.313657 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.8000 | 358.47 | 4937.13 | 6.00 | 0.162037 | 0.0000 | 0.1000 |
| `anti_conformity_final_vote` | 0.7500 | 358.47 | 5081.45 | 6.00 | 0.147596 | 0.0000 | 0.1222 |
| `free_mad_lite_llm_trajectory` | 0.8167 | 358.47 | 6369.98 | 7.00 | 0.128205 | 0.0167 | 0.1222 |

## 机制诊断

- Judge fallback rate：`0.016667`；Judge fallback count：`1`。
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`。
- 如果 changed answer rate 很高，但 corrected rate 没有同步提高，说明轨迹裁决更像是在频繁改写答案，而不是真正识别正确轨迹。

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.6000 | 0.00 | 860.55 | 0.697229 |
| `vanilla_mad_r1_final_vote` | 0.8500 | 288.20 | 2493.65 | 0.340866 |
| `anti_conformity_final_vote` | 0.9000 | 288.20 | 2637.00 | 0.341297 |
| `free_mad_lite_llm_trajectory` | 0.9500 | 288.20 | 3576.45 | 0.265627 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.7000 | 0.00 | 4893.85 | 0.143037 |
| `vanilla_mad_r1_final_vote` | 0.7500 | 432.10 | 10482.50 | 0.071548 |
| `anti_conformity_final_vote` | 0.7000 | 432.10 | 10615.50 | 0.065941 |
| `free_mad_lite_llm_trajectory` | 0.7000 | 432.10 | 12794.50 | 0.054711 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.7000 | 0.00 | 622.00 | 1.125402 |
| `vanilla_mad_r1_final_vote` | 0.8000 | 355.10 | 1835.25 | 0.435908 |
| `anti_conformity_final_vote` | 0.6500 | 355.10 | 1991.85 | 0.326330 |
| `free_mad_lite_llm_trajectory` | 0.8000 | 355.10 | 2739.00 | 0.292077 |

## 结论与建议

- 若 `free_mad_lite_llm_trajectory` 在总体准确率和每千 token 准确率上都优于 `vanilla_mad_r1_final_vote`，说明轨迹裁决在当前设置下具有独立价值。
- 若 judge fallback rate 偏高，应优先增强 judge 的稳定性，再考虑扩大 anti-conformity 的使用范围。
- 进入更大样本 phase 前，建议同时核对轨迹裁决效果图和 fallback 图，避免只看总体准确率结论。

## 局限性

- 当前实现只验证单轮 anti-conformity 与 LLM trajectory judge，不包含论文中的完整 score-based 决策训练流程。
- 本报告反映的是当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。

## 复现与产物说明

- 运行目录：`runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`trajectory_scores.jsonl`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。

## 图表资产

### Free-MAD-lite 成本-性能前沿

![Free-MAD-lite 成本-性能前沿](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各 Free-MAD-lite 变体的准确率相对于平均总 token 的位置关系。*

### Free-MAD-lite 效率排序

![Free-MAD-lite 效率排序](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### Free-MAD-lite 跨数据集表现

![Free-MAD-lite 跨数据集表现](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各 Free-MAD-lite 变体在不同数据集上的准确率分布。*

### 轨迹裁决效果

![轨迹裁决效果](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/trajectory_score_panel.svg)

*总体层面 changed / corrected / harmed 三类比例对比。*

### Judge fallback 概览

![Judge fallback 概览](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/20260509T125837Z-free_mad_lite_mechanism_validation-smoke20-xiaomimimo-mimo-v2.5/figures/judge_fallback_summary.svg)

*总体层面 judge fallback rate 对比。*
