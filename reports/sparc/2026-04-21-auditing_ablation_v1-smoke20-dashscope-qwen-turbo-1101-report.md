# SPARC 审计消融报告

## 1. 实验范围与公平性说明

- 实验名：`auditing_ablation_v1`
- Phase：`smoke20`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`local/runs/sparc/auditing_ablation/20260421T085539Z-auditing_ablation_v1-smoke20-dashscope-qwen-turbo-1101`
- 本轮固定消息内容，只比较最终聚合与局部审计方式。

## 2. 主结果表

| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `majority_vote` | 0.5333 | 0.00 | 0.00 | 2262.87 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `single_judge` | 0.6000 | 2537.25 | 822.30 | 5622.42 | 1.0000 | 0.0000 | 0.0167 | 2 |
| `final_round_vote` | 0.5833 | 2537.25 | 0.00 | 4800.12 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `local_auditing` | 0.6000 | 2537.25 | 166.00 | 4966.12 | 0.2167 | 0.0000 | 0.0167 | 2 |

## 3. 探索性区间

- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：[-0.033333, 0.083333]（探索性）

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

- 推荐默认聚合方式：`local_auditing`
- 推荐依据：overall accuracy=`0.6000`，total tokens=`4966.12`。

