# SPARC 审计消融报告

## 1. 实验范围与公平性说明

- 实验名：`local_auditing_ablation`
- Phase：`confirmatory300`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5`
- 本轮固定消息内容，只比较最终聚合与局部审计方式。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority_vote` | 0.6128 | 0.00 | 0.00 | 2602.72 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `weighted_vote_fallback` | 0.6357 | 2904.14 | 0.00 | 5506.87 | 0.0000 | 0.0000 | 0.0000 | 28 |
| `single_judge` | 0.7189 | 2904.14 | 1065.70 | 6572.56 | 1.0000 | 0.0000 | 0.0084 | 77 |
| `final_round_vote` | 0.6345 | 2904.14 | 0.00 | 5506.87 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `local_auditing` | 0.6695 | 2904.14 | 243.62 | 5750.49 | 0.2449 | 0.0241 | 0.0097 | 36 |

## 3. 探索性区间

- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：[0.019300, 0.051870]（探索性）

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
- 样本：`gsm8k-01135`
- 问题预览：3 trees each had 7 blue birds in them. 2 different trees each had 4 blue birds. 1 final tree had 3 blue birds. How ma...
- 金标：`32`
- 主方法：`35` / score=`0.0`
- 参考方法：`32` / score=`1.0`
- 说明：three_way::disagreement_step_only

### Case 3

- 数据集：`strategyqa`
- 样本：`12b0bb830de101c0f118`
- 问题预览：Was Bruce Lee absent from the 1964 University of Washington graduation ceremony?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 4

- 数据集：`strategyqa`
- 样本：`1932e05f10680ece229f`
- 问题预览：Would the top of Mount Fuji stick out of the Sea of Japan?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

### Case 5

- 数据集：`strategyqa`
- 样本：`2800bf5c809ed1224b42`
- 问题预览：Would a kindergarten teacher make a lesson of the New Testament?
- 金标：`no`
- 主方法：`yes` / score=`0.0`
- 参考方法：`no` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

## 5. 下一轮建议

- 推荐默认聚合方式：`single_judge`
- 推荐依据：overall accuracy=`0.7189`，total tokens=`6572.56`。

## 图表资产

### SPARC frontier

![SPARC frontier](../../runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/frontier_overall.svg)

*Overall accuracy versus average total tokens across SPARC variants and controls.*

### SPARC efficiency ranking

![SPARC efficiency ranking](../../runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/efficiency_rank_overall.svg)

*Overall efficiency ranking measured by accuracy per 1K tokens.*

### SPARC score by dataset

![SPARC score by dataset](../../runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/score_by_dataset.svg)

*Per-dataset accuracy map across SPARC variants and controls.*

### Audit gain versus cost

![Audit gain versus cost](../../runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/audit_gain_vs_cost.svg)

*Overall audit-token cost versus accuracy across auditing and end-to-end variants.*

### Trigger selection profile

![Trigger selection profile](../../runs/sparc/local_auditing_ablation/confirmatory300/20260509T130054Z-local_auditing_ablation-confirmatory300-xiaomimimo-mimo-v2.5/figures/trigger_selection_profile.svg)

*Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.*
