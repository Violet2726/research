# SPARC v1 Smoke 报告

## 1. 实验范围与公平性说明

- 实验名：`sparc_v1_smoke`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/sparc/sparc_v1/20260421T083209Z-sparc_v1_smoke-smoke20-dashscope-qwen-turbo-1101`
- 选中的 trigger 策略：`hybrid_trigger`
- trigger 选择原因：`default_policy_kept`
- trigger 参考运行：`local\runs\selective_comm\trigger-early-exit-v1\smoke20\20260419T081522Z-trigger-early-exit-v1-smoke20-dashscope-qwen-turbo-1101`

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5333 | 0.00 | 0.00 | 2262.87 | 3.00 | 0.235689 | 0.0000 | 1.0000 |
| `always_communicate` | 0.6000 | 2537.25 | 166.00 | 4966.12 | 6.22 | 0.120819 | 1.0000 | 0.0000 |
| `hybrid_trigger_baseline` | 0.5833 | 720.77 | 0.00 | 2983.63 | 3.95 | 0.195511 | 0.3167 | 0.6833 |
| `final_round_vote_baseline` | 0.5833 | 2537.25 | 0.00 | 4800.12 | 6.00 | 0.121525 | 1.0000 | 0.0000 |
| `sparc_v1` | 0.6000 | 720.77 | 158.77 | 3142.40 | 4.15 | 0.190937 | 0.3167 | 0.6833 |

## 3. 探索性区间

- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.083333]（探索性）

## 5. 失败案例

### Case 1

- 数据集：`strategyqa`
- 样本：`b848db708048f54dfb6c`
- 问题预览：Is Lines on the Antiquity of Microbes briefer than any haiku?
- 金标：`yes`
- 主方法：`no` / score=`0.0`
- 参考方法：`yes` / score=`1.0`
- 说明：two_way_majority::disagreement_step_only

## 5. 下一轮建议

- 当前最佳 overall 方法：`sparc_v1`
- trigger 选择记录：drop_questions=`0.0`，threshold=`1.0`。

