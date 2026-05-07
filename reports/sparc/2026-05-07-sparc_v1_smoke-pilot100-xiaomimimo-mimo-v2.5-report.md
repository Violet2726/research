# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`sparc_v1_smoke`
- Phase：`pilot100`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/sparc/sparc_v1_smoke/pilot100/20260507T121717Z-sparc_v1_smoke-pilot100-xiaomimimo-mimo-v2.5`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`reference_missing`
- 未找到 trigger 参考运行，使用默认策略。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.6267 | 0.00 | 0.00 | 2443.48 | 3.00 | 0.256465 | 0.0000 | 1.0000 |
| `always_communicate` | 0.6667 | 2754.70 | 235.60 | 5433.79 | 6.26 | 0.122689 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.6267 | 891.98 | 0.00 | 3335.46 | 4.11 | 0.187880 | 0.3700 | 0.6300 |
| `final_round_vote_baseline` | 0.6267 | 2754.70 | 0.00 | 5198.18 | 6.00 | 0.120555 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.6667 | 891.98 | 235.60 | 3571.07 | 4.37 | 0.186686 | 0.3700 | 0.6300 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[0.013333, 0.066667]（探索性）

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

- 当前最佳 overall 方法：`sparc_v1`
- trigger 选择记录：drop_questions=`None`，threshold=`None`。

