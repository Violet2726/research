# Free-MAD-lite 科研报告

## 摘要

- 总体准确率最高的方法是 `free_mad_lite_llm_trajectory`，准确率为 0.7720。
- 总体效率最高的方法是 `mv_3_initial`，每千 token 准确率为 0.273716。
- `free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的总体准确率差异 bootstrap 95% CI 为 `[-0.010856, 0.025332]`。

## 实验概览

- 实验名：`free_mad_lite_mechanism_validation`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5`

## 研究问题与实验设计

- Free-MAD-lite 关注单轮 anti-conformity 与 LLM trajectory judge 是否足以带来稳定收益，而不复现完整 score-model 训练流程。
- 主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；机制指标重点是 changed answer、corrected、harmed 与 judge fallback rate。
- 本实验固定比较 `mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote` 和 `free_mad_lite_llm_trajectory`，因此可以隔离轨迹裁决环节的贡献。

## 总体结果

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 | Judge fallback rate | Changed answer rate |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.6273 | 0.00 | 2291.66 | 3.00 | 0.273716 | 0.0000 | 0.0000 |
| `vanilla_mad_r1_final_vote` | 0.7648 | 365.74 | 5271.72 | 6.00 | 0.145072 | 0.0000 | 0.0937 |
| `anti_conformity_final_vote` | 0.7129 | 365.74 | 5419.82 | 6.00 | 0.131537 | 0.0000 | 0.1263 |
| `free_mad_lite_llm_trajectory` | 0.7720 | 365.74 | 6745.59 | 7.00 | 0.114447 | 0.0145 | 0.1263 |

## 机制诊断

- Judge fallback rate：`0.014475`；Judge fallback count：`12`。
- Anti-conformity prompt hash：`9f6b2d76ed11e9753c3655a24292e91bb59e053e53c83bd1aa6eff8ac7fdcc38`。
- 如果 changed answer rate 很高，但 corrected rate 没有同步提高，说明轨迹裁决更像是在频繁改写答案，而不是真正识别正确轨迹。

## 分数据集表现

### gsm8k

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.5167 | 0.00 | 959.69 | 0.538370 |
| `vanilla_mad_r1_final_vote` | 0.8467 | 313.63 | 2689.38 | 0.314819 |
| `anti_conformity_final_vote` | 0.8167 | 313.63 | 2845.78 | 0.286975 |
| `free_mad_lite_llm_trajectory` | 0.8900 | 313.63 | 3780.49 | 0.235419 |

### hotpotqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.6600 | 0.00 | 4906.37 | 0.134519 |
| `vanilla_mad_r1_final_vote` | 0.6733 | 437.23 | 10514.95 | 0.064036 |
| `anti_conformity_final_vote` | 0.6533 | 437.23 | 10641.79 | 0.061393 |
| `free_mad_lite_llm_trajectory` | 0.6633 | 437.23 | 12813.29 | 0.051769 |

### strategyqa

| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每千 token 准确率 |
| --- | --- | --- | --- | --- |
| `mv_3_initial` | 0.7293 | 0.00 | 611.21 | 1.193147 |
| `vanilla_mad_r1_final_vote` | 0.7773 | 340.36 | 1785.83 | 0.435257 |
| `anti_conformity_final_vote` | 0.6550 | 340.36 | 1950.93 | 0.335748 |
| `free_mad_lite_llm_trajectory` | 0.7598 | 340.36 | 2681.03 | 0.283408 |

## 结论与建议

- 若 `free_mad_lite_llm_trajectory` 在总体准确率和每千 token 准确率上都优于 `vanilla_mad_r1_final_vote`，说明轨迹裁决在当前设置下具有独立价值。
- 若 judge fallback rate 偏高，应优先增强 judge 的稳定性，再考虑扩大 anti-conformity 的使用范围。
- 进入更大样本 phase 前，建议同时核对轨迹裁决效果图和 fallback 图，避免只看总体准确率结论。

## 局限性

- 当前实现只验证单轮 anti-conformity 与 LLM trajectory judge，不包含论文中的完整 score-based 决策训练流程。
- 本报告反映的是当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。

## 复现与产物说明

- 运行目录：`runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5`
- 关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`trajectory_scores.jsonl`。
- 本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。

## 图表资产

### Free-MAD-lite 成本-性能前沿

![Free-MAD-lite 成本-性能前沿](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*总体结果上，各 Free-MAD-lite 变体的准确率相对于平均总 token 的位置关系。*

### Free-MAD-lite 效率排序

![Free-MAD-lite 效率排序](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*基于每千 token 准确率的总体效率排序。*

### Free-MAD-lite 跨数据集表现

![Free-MAD-lite 跨数据集表现](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*各 Free-MAD-lite 变体在不同数据集上的准确率分布。*

### 轨迹裁决效果

![轨迹裁决效果](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5/figures/trajectory_score_panel.svg)

*总体层面 changed / corrected / harmed 三类比例对比。*

### Judge fallback 概览

![Judge fallback 概览](../../runs/free_mad_lite/free_mad_lite_mechanism_validation/confirmatory300/20260509T125952Z-free_mad_lite_mechanism_validation-confirmatory300-xiaomimimo-mimo-v2.5/figures/judge_fallback_summary.svg)

*总体层面 judge fallback rate 对比。*
