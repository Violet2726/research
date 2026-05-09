# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`end_to_end_main`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6667 | 0.00 | 0.00 | 2453.77 | 3.00 | 0.271691 | 0.0000 | 1.0000 |
| `always_communicate` | 0.7333 | 2759.32 | 326.83 | 5539.92 | 6.32 | 0.132373 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.7000 | 1060.77 | 0.00 | 3514.53 | 4.25 | 0.199173 | 0.4167 | 0.5833 |
| `final_round_vote_baseline` | 0.7000 | 2759.32 | 0.00 | 5213.08 | 6.00 | 0.134278 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.7167 | 1060.77 | 297.98 | 3812.52 | 4.55 | 0.187977 | 0.4167 | 0.5833 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.083333]（探索性）

## 5. 失败案例

### Case 1

- 数据集：`strategyqa`
- 样本：`62e8d2b87e80f78b152c`
- 问题预览：Does an individual oceanographer study many sciences?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

## 5. 下一轮建议

- 当前最佳 overall 方法：`always_communicate`
- trigger 选择记录：drop_questions=`None`，threshold=`None`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Audit gain versus cost

![Audit gain versus cost](../../runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5/figures/audit_gain_vs_cost.svg)

*Overall audit-token cost versus accuracy across auditing and end-to-end variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/end_to_end_main/smoke20/20260509T125844Z-end_to_end_main-smoke20-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
