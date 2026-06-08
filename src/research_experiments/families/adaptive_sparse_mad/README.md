# adaptive_sparse_mad

`adaptive_sparse_mad` 用于验证 A-SMAD：异质 Stage A 求解、稀疏触发验证、证据感知聚合的 same-context 框架。

## 当前主线

- Stage A 异质 solver：`solver_cot / solver_l2m / solver_skeptic`
- Stage A 基线聚合：`hetero_vote_3`
- 机制主线：
  - `ega_only_v4`：只验证 EGA 聚合本身是否有效
  - `adaptive_gate_v4`：通用稀疏验证主线
  - `adaptive_dual_open_v5`：开放问答上的双 verifier 主线

## 当前保留配置

- `same_context_main.toml`
  - `gsm8k / strategyqa / hotpotqa` 的 v2 基线包
- `same_context_main_v4.toml`
  - `gsm8k / strategyqa / hotpotqa` 的通用 v4 主线包
- `same_context_hotpot_stage_a_v2.toml`
  - HotpotQA 的 Stage A 基线
- `same_context_hotpot_stage_a_v4.toml`
  - HotpotQA 的 `adaptive_gate_v4`
- `same_context_hotpot_stage_a_v4_ablate.toml`
  - HotpotQA 的主线对比包：`hetero_vote_3 / ega_only_v4 / adaptive_gate_v4 / adaptive_dual_open_v5`
- `same_context_hotpot_stage_a_v5.toml`
  - HotpotQA 的当前最强开放问答主线
- `same_context_competition_math_stage_a_v2.toml`
  - competition_math 的 Stage A 基线
- `same_context_competition_math_stage_a_v4.toml`
  - competition_math 的 `adaptive_gate_v4`
- `same_context_hard_transfer_stage_a_v2.toml`
  - `mmlu_pro / gpqa_diamond / math500` 的 Stage A 基线
- `same_context_hard_transfer_stage_a_v4.toml`
  - `mmlu_pro / gpqa_diamond / math500` 的 `adaptive_gate_v4`

## 已清理内容

- 已删除无价值的 prompt-only、slot-extractor、intersection、evidence-SC 试验分支
- 已删除重复的 generalization 配置；后续统一用 `same_context_main(_v4)` 和分数据集配置推进
- 校验逻辑只保留当前主线策略，不再为历史废弃策略保留常驻兼容

## 运行与验证

```powershell
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v4.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v4.toml --phase count100 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad validate-run --run-dir local/runs/adaptive_sparse_mad/<experiment>/count100/<run_id>
uv run research_cli experiment --family adaptive_sparse_mad render-report --run-dir local/runs/adaptive_sparse_mad/<experiment>/count100/<run_id>
```

## 研究原则

- 主要在 `count100` 上做 paired significance 验证
- 不针对单一数据集做提示词特调
- 优先推进机制创新，而不是继续堆 verifier 数量或细碎 prompt 微调
- 若某条机制在跨数据集上不稳定，应及时收缩或删除
