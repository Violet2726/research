# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`end_to_end_main`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6128 | 0.00 | 0.00 | 2602.72 | 3.00 | 0.235440 | 0.0000 | 1.0000 |
| `always_communicate` | 0.6695 | 2904.14 | 243.62 | 5750.49 | 6.27 | 0.116422 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.6321 | 886.75 | 0.00 | 3489.47 | 4.12 | 0.181141 | 0.3739 | 0.6261 |
| `final_round_vote_baseline` | 0.6345 | 2904.14 | 0.00 | 5506.87 | 6.00 | 0.115220 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.6647 | 886.75 | 239.20 | 3728.67 | 4.39 | 0.178255 | 0.3739 | 0.6261 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[0.014475, 0.045868]（探索性）

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

- 数据集：`gsm8k`
- 样本：`gsm8k-00581`
- 问题预览：Max plans to watch two movies this weekend. The first movie is 1 hour and 30 minutes long while the second movie is 2...
- 金标：`215`
- 主方法：`155` / score=`0.0`
- 参考方法：`215` / score=`1.0`
- 说明：early_exit_by_hybrid_rule

### Case 3

- 数据集：`gsm8k`
- 样本：`gsm8k-01135`
- 问题预览：3 trees each had 7 blue birds in them. 2 different trees each had 4 blue birds. 1 final tree had 3 blue birds. How ma...
- 金标：`32`
- 主方法：`35` / score=`0.0`
- 参考方法：`32` / score=`1.0`
- 说明：three_way::disagreement_step_only

### Case 4

- 数据集：`strategyqa`
- 样本：`12b0bb830de101c0f118`
- 问题预览：Was Bruce Lee absent from the 1964 University of Washington graduation ceremony?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 5

- 数据集：`strategyqa`
- 样本：`1932e05f10680ece229f`
- 问题预览：Would the top of Mount Fuji stick out of the Sea of Japan?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

## 5. 下一轮建议

- 当前最佳 overall 方法：`always_communicate`
- trigger 选择记录：drop_questions=`None`，threshold=`None`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Audit gain versus cost

![Audit gain versus cost](../../runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/audit_gain_vs_cost.svg)

*Overall audit-token cost versus accuracy across auditing and end-to-end variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/end_to_end_main/confirmatory300/20260509T130027Z-end_to_end_main-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
