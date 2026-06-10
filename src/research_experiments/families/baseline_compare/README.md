# baseline_compare

- 默认报告目录：`local/reports/baseline_compare/`
- 固定六方法基准包：`cot_1 / sc_3 / sc_5 / mad_3a_r1 / mad_3a_r2 / mad_5a_r1`
- 主用途：为后续创新方法提供稳定、可复用的 same-context 基准对比结果

```powershell
uv run research_cli experiment --family baseline_compare inspect-experiment --experiment configs/families/baseline_compare/experiments/core_six_method_baseline.toml
uv run research_cli experiment --family baseline_compare run --experiment configs/families/baseline_compare/experiments/core_six_method_baseline.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family baseline_compare render-report --run-dir local/runs/baseline_compare/core_six_method_baseline/count20/<run_id>
```

