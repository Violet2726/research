# SPARC 审计消融报告

## 1. 实验范围与公平性说明

- 实验名：`auditing_ablation_v1`
- Phase：`smoke20`
- Backbone：`None`
- 运行目录：`runs/sparc/auditing_ablation_v1/smoke20/20260505T053826Z-auditing_ablation_v1-smoke20-xiaomimimo-mimo-v2.5`
- 本轮固定消息内容，只比较最终聚合与局部审计方式。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority_vote` | 0.6000 | 0.00 | 0.00 | 2429.68 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `weighted_vote_fallback` | 0.6333 | 2751.80 | 0.00 | 5181.48 | 0.0000 | 0.0000 | 0.0000 | 2 |
| `single_judge` | 0.8167 | 2751.80 | 1006.93 | 6188.42 | 1.0000 | 0.0000 | 0.0000 | 11 |
| `final_round_vote` | 0.6500 | 2751.80 | 0.00 | 5181.48 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `local_auditing` | 0.6833 | 2751.80 | 325.42 | 5506.90 | 0.2833 | 0.0667 | 0.0167 | 4 |

## 3. 探索性区间

- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.100000]（探索性）

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

- 推荐默认聚合方式：`single_judge`
- 推荐依据：overall accuracy=`0.8167`，total tokens=`6188.42`。

