# Baseline Ceiling Audit Notes

本页记录 `xiaomimimo/mimo-v2.5` 固定预算 simple baseline ceiling 审计的边界、论文映射和当前主线结论。旧的 prompt/temperature screening 流程已经完成其审计职责，不再作为正式可运行入口维护。

## 原始论文映射

| 方法 | 主引用 | 本地文件 |
| --- | --- | --- |
| CoT | [Chain-of-Thought Prompting Elicits Reasoning in Large Language Models](https://arxiv.org/abs/2201.11903) | `NeurIPS-2022-chain-of-thought-prompting-elicits-reasoning-in-large-language-models-Paper-Conference.pdf` |
| zero-shot CoT | [Large Language Models are Zero-Shot Reasoners](https://arxiv.org/abs/2205.11916) | 当前仅保留在线主引用 |
| Self-Consistency / majority-vote sampling | [Self-Consistency Improves Chain of Thought Reasoning in Language Models](https://arxiv.org/abs/2203.11171) | `Self-Consistency Improves Chain of Thought Reasoning in Language Models.pdf` |

## 边界参考

下列工作只作为“simple baseline 仍可能继续提升”的边界参考，不纳入当前 canonical simple baseline：

- [Automatic Chain of Thought Prompting in Large Language Models](https://arxiv.org/abs/2210.03493)
- [Active Prompting with Chain-of-Thought for Large Language Models](https://arxiv.org/abs/2302.12246)
- [Universal Self-Consistency for Large Language Model Generation](https://arxiv.org/abs/2311.17311)
- [Optimal Self-Consistency for Efficient Reasoning with Large Language Models](https://arxiv.org/abs/2511.12309)

## 当前主线

- 正式入口：`configs/families/single_agent/experiments/canonical_simple_baselines.toml`
- 方法目录：`configs/families/single_agent/methods/common.toml`
- 覆盖数据集：`competition_math / gpqa_diamond / gsm8k / hotpotqa / math500 / mmlu_pro`
- 固定预算：`cot_1 / mv_3 / sc_5` 对应 `1 / 3 / 5` calls
- 解码设置：三种方法统一 `temperature=0.7 / top_p=1.0 / max_output_tokens=256`
- 主 prompt：`single_agent_reasoning_json_v1`
- authoritative 口径：`count100`，3 reruns
- `competition_math` 主口径：`count100_seed42`

## 已退休的旧逻辑

- `baseline_ceiling_v1_current_prompt.toml`
- `baseline_ceiling_v1_unified_control.toml`
- `baseline_ceiling_v1_zero_shot_cot.toml`
- `baseline_ceiling_candidates.toml`
- `ceiling_audit select-screening`
- `ceiling_audit summarize-ceiling`
- `ceiling_audit reference-audit`
- `unified_control_v1_port` 与 `zero_shot_cot_v1` 作为 single_agent 可运行 prompt 版本

这些旧入口只属于本轮审计的探索过程。它们的结论已经沉淀为：`cot_1@temp=0.0` 不能继续作为强基线主对照；`cot_1/mv_3/sc_5` 在该基座模型上的正式 simple baseline 应统一切换到 `temp=0.7`。

## 当前冻结结论

- `xiaomimimo/mimo-v2.5` 的 canonical simple baseline 固定为 `cot_1@temp=0.7 / mv_3@temp=0.7 / sc_5@temp=0.7`。
- `count20` 是 `count100` 的真子集，只能用于 sanity check，不再包装为独立测试集或正式筛选依据。
- 新主方法若只超过旧 `cot_1@temp=0.0`，不能宣称超过 strong simple baseline。
- 新主方法结论应优先使用 `canonical_baseline_recheck` 产物，对齐同模型、同 split、同固定预算的 canonical simple baseline。

## 复核入口

最小示例：

```powershell
uv run python -m research_experiments.families.single_agent.ceiling_audit rebaseline-conclusions --canonical-summary-json local/reports/single_agent/baseline_ceiling_summary/ceiling_summary.json --run-dir local/runs/adaptive_sparse_mad/same_context_full_counterfactual_v1/count100/20260608T065426Z-xiaomimimo-mimo-v2.5 --output-dir files/references
```

注意：split-context 结论不能直接用 full-context canonical simple baseline 裁定，应继续使用 split no-comm baseline 单独复核。
