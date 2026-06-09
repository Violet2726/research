# single_agent

`single_agent` 用于运行单智能体基线实验，当前覆盖 `cot`、`mv_*` 与 `sc_*` 这类无通信 baseline。

## 入口

- CLI：`research_cli experiment --family single_agent`
- 配置：`configs/families/single_agent/`
- 默认运行目录：`local/runs/single_agent/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/single_agent/`

## 常用命令

```powershell
uv run research_cli experiment --family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/same_context_core_benchmarks.toml
uv run research_cli experiment --family single_agent run --experiment configs/families/single_agent/experiments/same_context_core_benchmarks.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family single_agent validate-run --run-dir local/runs/single_agent/same_context_core_benchmarks/count20/<run_id>
uv run research_cli experiment --family single_agent render-report --run-dir local/runs/single_agent/same_context_core_benchmarks/count20/<run_id>
```

## Canonical Simple Baseline

`xiaomimimo/mimo-v2.5` 的正式 simple baseline 口径已经固定为：

- `cot_1@temp=0.7`
- `mv_3@temp=0.7`
- `sc_5@temp=0.7`

统一入口为：

```powershell
uv run research_cli experiment --family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/canonical_simple_baselines.toml
uv run research_cli experiment --family single_agent run --experiment configs/families/single_agent/experiments/canonical_simple_baselines.toml --phase count100 --model xiaomimimo/mimo-v2.5
```

该入口覆盖 `competition_math / gpqa_diamond / gsm8k / hotpotqa / math500 / mmlu_pro`，其中 `competition_math` 的 count100 split 固定为 `count100_total_seed0`。`count100` 使用 3 reruns；后续主结论应优先对齐这组 canonical baseline，而不是旧的 `cot_1@temp=0.0`。

## Baseline Ceiling Audit

```powershell
uv run research_cli experiment --family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml
uv run research_cli experiment --family single_agent run --experiment configs/families/single_agent/experiments/baseline_ceiling_v1_unified_control.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run python -m research_experiments.families.single_agent.ceiling_audit select-screening --run-dir local/runs/single_agent/baseline_ceiling_v1_current_prompt/count20/<run_id> --run-dir local/runs/single_agent/baseline_ceiling_v1_unified_control/count20/<run_id> --run-dir local/runs/single_agent/baseline_ceiling_v1_zero_shot_cot/count20/<run_id> --output-dir local/reports/single_agent/baseline_ceiling_selection
```
