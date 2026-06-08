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

- screening configs:
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml`
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_unified_control.toml`
  - `configs/families/single_agent/experiments/baseline_ceiling_v1_zero_shot_cot.toml`
- screening method catalog:
  - `configs/families/single_agent/methods/baseline_ceiling_candidates.toml`
- audit tool:
  - `uv run python -m research_experiments.families.single_agent.ceiling_audit ...`
