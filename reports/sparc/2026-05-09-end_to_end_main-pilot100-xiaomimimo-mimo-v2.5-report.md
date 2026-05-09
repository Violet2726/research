# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`end_to_end_main`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6400 | 0.00 | 0.00 | 2445.76 | 3.00 | 0.261677 | 0.0000 | 1.0000 |
| `always_communicate` | 0.6800 | 2750.57 | 242.71 | 5439.05 | 6.27 | 0.125022 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.6633 | 976.66 | 0.00 | 3422.43 | 4.24 | 0.193819 | 0.4133 | 0.5867 |
| `final_round_vote_baseline` | 0.6633 | 2750.57 | 0.00 | 5196.34 | 6.00 | 0.127654 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.6767 | 976.66 | 236.94 | 3659.37 | 4.50 | 0.184914 | 0.4133 | 0.5867 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[-0.010000, 0.036667]（探索性）

## 5. 失败案例

### Case 1

- 数据集：`gsm8k`
- 样本：`gsm8k-00002`
- 问题预览：Josh decides to try flipping a house. He buys a house for $80,000 and then puts in $50,000 in repairs. This increased...
- 金标：`70000`
- 主方法：`120000` / score=`0.0`
- 参考方法：`70000` / score=`1.0`
- 说明：three_way::disagreement_step_only

### Case 2

- 数据集：`strategyqa`
- 样本：`1932e05f10680ece229f`
- 问题预览：Would the top of Mount Fuji stick out of the Sea of Japan?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 3

- 数据集：`strategyqa`
- 样本：`2800bf5c809ed1224b42`
- 问题预览：Would a kindergarten teacher make a lesson of the New Testament?
- 金标：`no`
- 主方法：`yes` / score=`0.0`
- 参考方法：`no` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 4

- 数据集：`strategyqa`
- 样本：`5b45ccb915731a9e5f05`
- 问题预览：Would it be hard to get toilet paper if there were no loggers?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 5

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

![SPARC frontier](../../runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Audit gain versus cost

![Audit gain versus cost](../../runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5/figures/audit_gain_vs_cost.svg)

*Overall audit-token cost versus accuracy across auditing and end-to-end variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/end_to_end_main/pilot100/20260509T125917Z-end_to_end_main-pilot100-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
