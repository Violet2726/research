# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`sparc_v1_smoke`
- Phase：`smoke20`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/sparc_v1_smoke/smoke20/20260505T092749Z-sparc_v1_smoke-smoke20-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6500 | 0.00 | 0.00 | 2432.85 | 3.00 | 0.267176 | 0.0000 | 1.0000 |
| `always_communicate` | 0.7167 | 2748.40 | 260.50 | 5441.75 | 6.30 | 0.131698 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.6667 | 822.52 | 0.00 | 3255.37 | 4.10 | 0.204790 | 0.3667 | 0.6333 |
| `final_round_vote_baseline` | 0.6833 | 2748.40 | 0.00 | 5181.25 | 6.00 | 0.131886 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.7000 | 822.52 | 231.35 | 3486.72 | 4.38 | 0.200762 | 0.3667 | 0.6333 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.066667]（探索性）

## 5. 失败案例

### Case 1

- 数据集：`hotpotqa`
- 样本：`5a89fea655429970aeb701eb`
- 问题预览：In which film did Emilio Estevez star in in the same year as Nightmares
- 金标：`The Outsiders`
- 主方法：`nightmares` / score=`0.0`
- 参考方法：`outsiders` / score=`1.0`
- 说明：early_exit_by_hybrid_rule

## 5. 下一轮建议

- 当前最佳 overall 方法：`always_communicate`
- trigger 选择记录：drop_questions=`None`，threshold=`None`。

