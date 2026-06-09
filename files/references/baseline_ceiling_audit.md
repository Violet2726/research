# Baseline Ceiling Audit Notes

本页固定记录 `xiaomimimo/mimo-v2.5` 固定预算 baseline ceiling 审计的论文映射、口径和执行边界。

## 原始论文映射

| 方法 | 主引用 | 本地文件 |
| --- | --- | --- |
| CoT | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | `NeurIPS-2022-chain-of-thought-prompting-elicits-reasoning-in-large-language-models-Paper-Conference.pdf` |
| zero-shot CoT | [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | 当前仅保留在线主引用 |
| Self-Consistency / majority-vote sampling | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | `Self-Consistency Improves Chain of Thought Reasoning in Language Models.pdf` |

## 边界参考

下列工作只作为“simple baseline 仍可能继续提升”的边界参考，不纳入本轮正式 ceiling 候选：

- [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)
- [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/abs/2302.12246)
- [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)
- [Optimal Self-Consistency for Efficient Reasoning with Large Language Models](https://arxiv.org/abs/2511.12309)

## 审计口径

- 目标数据集固定为 `competition_math / gpqa_diamond / gsm8k / hotpotqa / math500 / mmlu_pro`
- 固定预算固定为 `cot_1 / mv_3 / sc_5`
- 允许搜索：
  - prompt family
  - 全局 temperature
  - 仍沿用仓库现有答案归一化与评测器
- 不允许：
  - 按数据集单独调 prompt
  - 扩大调用预算
  - 修改判分规则以制造收益

## split 口径

- `count20` 只作 screen
- `count100` 才作 authoritative ceiling
- `competition_math` 主口径固定为 `count100_total_seed0`
- 当前仓库的 `count20` 是 `count100` 的真子集，不能当作独立开发集解释

## 当前实现入口

- canonical simple baseline:
  - `configs/families/single_agent/experiments/canonical_simple_baselines.toml`
  - `configs/families/single_agent/methods/common.toml`
- screening configs:
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml`
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_unified_control.toml`
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_zero_shot_cot.toml`
- screening method catalog:
  - `configs/families/single_agent/methods/baseline_ceiling_candidates.toml`
- audit tool:
  - `uv run python -m research_experiments.families.single_agent.ceiling_audit ...`

## 当前冻结结论

- `xiaomimimo/mimo-v2.5` 的标准 simple baseline 固化为 `cot_1@temp=0.7 / mv_3@temp=0.7 / sc_5@temp=0.7`。
- `cot_1` 的旧 official `temp=0.0` 不能继续作为主结论对照；`count100` 上 `temp=0.7` 的 mean accuracy 从 `0.6900` 提升到 `0.7000`。
- `mv_3` 和 `sc_5` 的 `temp=0.7` 已基本是当前固定预算下的较优选择；继续盲目扩温度搜索的优先级低于重审旧 baseline 结论。
- `single_agent_reasoning_json_v1` 与 `unified_control_v1_port` 在当前 no-comm 路径下消息等价；canonical 入口保留 `single_agent_reasoning_json_v1`。
- `zero_shot_cot_v1` 未进入 canonical baseline，后续只作为边界候选记录，不作为正式 baseline。

## 复核入口

用 canonical baseline 重审同上下文/full-context 主方法结论；完整产物见 `files/references/canonical_baseline_recheck.md`。

最小示例命令：

```powershell
uv run python -m research_experiments.families.single_agent.ceiling_audit rebaseline-conclusions --canonical-summary-json local/reports/single_agent/baseline_ceiling_summary/ceiling_summary.json --run-dir local/runs/adaptive_sparse_mad/same_context_full_counterfactual_v1/count100/20260608T065426Z-xiaomimimo-mimo-v2.5 --run-dir local/runs/adaptive_sparse_mad/same_context_hotpot_stage_a_v5/count100/20260608T031630Z-xiaomimimo-mimo-v2.5 --output-dir files/references
```

注意：split-context 结论不能直接用 full-context canonical simple baseline 裁定，应继续使用 split no-comm baseline 单独复核。
