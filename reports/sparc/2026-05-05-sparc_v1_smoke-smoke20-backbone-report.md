# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`sparc_v1_smoke`
- Phase：`smoke20`
- Backbone：`None`
- 运行目录：`runs/sparc/sparc_v1_smoke/smoke20/20260505T050235Z-sparc_v1_smoke-smoke20-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6000 | 0.00 | 0.00 | 2429.68 | 3.00 | 0.246946 | 0.0000 | 1.0000 |
| `always_communicate` | 0.6833 | 2751.80 | 325.42 | 5506.90 | 6.35 | 0.124087 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.6500 | 1248.60 | 0.00 | 3678.28 | 4.45 | 0.176713 | 0.4833 | 0.5167 |
| `final_round_vote_baseline` | 0.6500 | 2751.80 | 0.00 | 5181.48 | 6.00 | 0.125447 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.6833 | 1248.60 | 325.42 | 4003.70 | 4.80 | 0.170675 | 0.4833 | 0.5167 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.100000]（探索性）

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

- 当前最佳 overall 方法：`sparc_v1`
- trigger 选择记录：drop_questions=`None`，threshold=`None`。

