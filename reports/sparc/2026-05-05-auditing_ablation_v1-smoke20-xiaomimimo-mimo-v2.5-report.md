# SPARC 审计消融报告

## 1. 实验范围与公平性说明

- 实验名：`auditing_ablation_v1`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/auditing_ablation_v1/smoke20/20260505T100552Z-auditing_ablation_v1-smoke20-xiaomimimo-mimo-v2.5`
- 本轮固定消息内容，只比较最终聚合与局部审计方式。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority_vote` | 0.6500 | 0.00 | 0.00 | 2432.85 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `weighted_vote_fallback` | 0.6833 | 2748.40 | 0.00 | 5181.25 | 0.0000 | 0.0000 | 0.0000 | 2 |
| `single_judge` | 0.8333 | 2748.40 | 1004.78 | 6186.03 | 1.0000 | 0.0000 | 0.0000 | 9 |
| `final_round_vote` | 0.6833 | 2748.40 | 0.00 | 5181.25 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `local_auditing` | 0.7167 | 2748.40 | 260.50 | 5441.75 | 0.2667 | 0.0333 | 0.0000 | 2 |

## 3. 探索性区间

- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：[0.000000, 0.083333]（探索性）

## 5. 失败案例

- 当前 smoke20 下没有稳定失败案例。

## 5. 下一轮建议

- 推荐默认聚合方式：`single_judge`
- 推荐依据：overall accuracy=`0.8333`，total tokens=`6186.03`。

