# SPARC 审计消融报告

## 1. 实验范围与公平性说明

- 实验名：`auditing_ablation_v1`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/auditing_ablation_v1/pilot100/20260507T121729Z-auditing_ablation_v1-pilot100-xiaomimimo-mimo-v2.5`
- 本轮固定消息内容，只比较最终聚合与局部审计方式。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority_vote` | 0.6267 | 0.00 | 0.00 | 2443.48 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `weighted_vote_fallback` | 0.6300 | 2754.70 | 0.00 | 5198.18 | 0.0000 | 0.0000 | 0.0000 | 7 |
| `single_judge` | 0.7033 | 2754.70 | 1009.73 | 6207.91 | 1.0000 | 0.0000 | 0.0100 | 26 |
| `final_round_vote` | 0.6267 | 2754.70 | 0.00 | 5198.18 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `local_auditing` | 0.6667 | 2754.70 | 235.60 | 5433.79 | 0.2267 | 0.0333 | 0.0067 | 13 |

## 3. 探索性区间

- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：[0.013333, 0.066667]（探索性）

## 5. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5ab29caa554299545a2cf9d3`
- 问题预览：Which gaming console was both Yakuza Kiwami and Yakuza 0 released on?
- 金标：`PlayStation 4`
- 主方法：`playstation 3 and playstation 4` / score=`0.0`
- 参考方法：`playstation 4` / score=`1.0`
- 说明：three_way::critical_evidence_only

### Case 2

- 数据集：`strategyqa`
- 样本：`1932e05f10680ece229f`
- 问题预览：Would the top of Mount Fuji stick out of the Sea of Japan?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

## 5. 下一轮建议

- 推荐默认聚合方式：`single_judge`
- 推荐依据：overall accuracy=`0.7033`，total tokens=`6207.91`。

