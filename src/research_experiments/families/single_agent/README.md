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

```powershell
uv run research_cli experiment --family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/baseline_ceiling_v1_current_prompt.toml
uv run research_cli experiment --family single_agent run --experiment configs/families/single_agent/experiments/baseline_ceiling_v1_unified_control.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run python -m research_experiments.families.single_agent.ceiling_audit select-screening --run-dir local/runs/single_agent/baseline_ceiling_v1_current_prompt/count20/<run_id> --run-dir local/runs/single_agent/baseline_ceiling_v1_unified_control/count20/<run_id> --run-dir local/runs/single_agent/baseline_ceiling_v1_zero_shot_cot/count20/<run_id> --output-dir local/reports/single_agent/baseline_ceiling_selection
```
