# single_agent

`single_agent` 只承载单智能体、无通信的 simple baseline 运行逻辑。当前正式主线已经收敛到 `xiaomimimo/mimo-v2.5` 的 canonical simple baseline；旧的 prompt/temperature screening 配置不再作为可运行入口维护。

## 入口

- CLI：`research_cli experiment --family single_agent`
- 配置：`configs/families/single_agent/`
- 默认运行目录：`local/runs/single_agent/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/single_agent/`

## Canonical Simple Baseline

正式入口固定为：

```powershell
uv run research_cli experiment --family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/canonical_simple_baselines.toml
uv run research_cli experiment --family single_agent run --experiment configs/families/single_agent/experiments/canonical_simple_baselines.toml --phase count100 --model xiaomimimo/mimo-v2.5
```

主线约束：

- 数据集固定为 `competition_math / gpqa_diamond / gsm8k / hotpotqa / math500 / mmlu_pro`
- 方法固定为 `cot_1 / mv_3 / sc_5`
- 全局解码固定为 `temperature=0.7 / top_p=1.0 / max_output_tokens=256`
- 预算严格固定为 `1 / 3 / 5` calls
- `competition_math` 的 count100 split 固定为 `count100_total_seed0`
- `count100` 使用 3 reruns，是当前 authoritative 口径
- `count20` 只用于轻量 sanity check，不再作为正式筛选链路

## 复核工具

`ceiling_audit.py` 现在只保留 canonical rebaseline 复核入口，用于判断主方法结论是否仍然超过 strong simple baseline：

```powershell
uv run python -m research_experiments.families.single_agent.ceiling_audit rebaseline-conclusions --canonical-summary-json local/reports/single_agent/baseline_ceiling_summary/ceiling_summary.json --run-dir local/runs/<family>/<experiment>/count100/<run_id> --output-dir files/references
```

旧的 `baseline_ceiling_v1_*` screening 配置、`baseline_ceiling_candidates.toml`、`select-screening`、`summarize-ceiling` 不再是主线的一部分。历史结论保留在 `files/references/`，新的主结论应优先对齐 `canonical_simple_baselines.toml`。
