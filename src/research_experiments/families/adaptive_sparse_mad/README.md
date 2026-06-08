# adaptive_sparse_mad

`adaptive_sparse_mad` 用于验证 A-SMAD：异质求解 + 稀疏通信 + 轨迹裁决的 same-context 新框架。

## 入口

- CLI：`research_cli experiment --family adaptive_sparse_mad`
- 配置：`configs/families/adaptive_sparse_mad/`
- 默认运行目录：`local/runs/adaptive_sparse_mad/<experiment>/<phase>/<run_id>/`

## 当前口径

- 主实验线：`same_context_main`
- hard-transfer 主实验线：`same_context_hard_transfer_stage_a_v2`
- 当前快速主方法：`hetero_vote_3`
- 当前强基线：`cot_1 / mv_3 / sc_5`
- 当前判断：
  先继续强化 `Stage A` 求解能力，优先解决 `all_three_wrong`，再处理 `clean_pseudo_majority`

## 常用命令

```powershell
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad render-report --run-dir local/runs/adaptive_sparse_mad/same_context_main/count20/<run_id>

# Hard-transfer: mmlu_pro / gpqa_diamond / math500
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_hard_transfer_stage_a_v2.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_hard_transfer_stage_a_v2.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_hard_transfer_stage_a_v2.toml --phase count100 --model xiaomimimo/mimo-v2.5
```

## Hard-transfer 状态

- `same_context_main` 保持冻结，作为 A0 参考。
- `same_context_hard_transfer_stage_a_v2/count20/20260606T132806Z-xiaomimimo-mimo-v2.5`
  `hetero_vote_3 = 0.7833`
  `gpqa_diamond = 0.70`
  `math500 = 0.90`
  `mmlu_pro = 0.75`
  当前 `count20` 仍然守住主线 gate。
- `same_context_hard_transfer_stage_a_v2/count100/20260606T132807Z-xiaomimimo-mimo-v2.5`
  `hetero_vote_3 = 0.7300`
  `gpqa_diamond = 0.62`
  `math500 = 0.77`
  `mmlu_pro = 0.80`
  当前最可信的 `count100` hard-transfer 主结果，稳定高于 `cot_1 / mv_3 / sc_5`。

## 误差分桶与 solver 贡献

- 当前 `count100` 主线分桶：
  `all_three_wrong = 58`
  `clean_pseudo_majority = 21`
  `confidence_miscalibration = 0`
  `constraint_mismatch = 2`
- 当前 `count100` solver contribution：
  `solver_cot any_correct = 211, solo_correct = 13`
  `solver_l2m any_correct = 202, solo_correct = 13`
  `solver_skeptic any_correct = 198, solo_correct = 9`
- 当前每个 run 都会写出：
  `diagnostics/stage_a_error_buckets.json`
  `diagnostics/stage_a_solver_contributions.json`
  供后续做 solver / aggregator 迭代决策。

## 当前决策

- `count100` 仍是主验证口径，`count300` 不进入默认流程。
- `Stage A v2` 继续作为 hard-transfer 主线。
- 当前主线已经纳入一个非常保守的通用聚合修正：
  在 `2:1` 分裂里，若少数派是干净的 `solver_cot`，则允许尊重该少数派；
  它在 `count20` 不退，在 `count100` 把总体从 `0.7267` 推到 `0.7300`。
- 已验证失败的 prompt-only、slot-extractor、intersection 和 evidence-SC 迭代分支已清理，当前只保留 `v2 / v4 / v5` 与主线对比配置。
- 下一步优先方向：
  不再继续微调第三个 solver，
  也不急着扩通信或 judge，
  而是重做更真正异质的 `Stage A` solver 组合，再基于分桶结果推进通用的 `constraint-aware aggregator`。
