# RCTA-MAD

`risk_controlled_trace_mad` is the active MAD innovation family. BRD-MAD and SGSA-MAD are frozen historical mechanisms.

RCTA uses five SC-aligned trajectories, one trace synthesizer on answer disagreement, optional safe executable certificates, and one frozen dataset/model-independent replacement-risk router. It never uses dataset, task, model, question embedding, gold labels, or model-reported confidence as router features.

Phases use only canonical names: `count20_seed42`, `count300_seed42`, and `full_seed42`. All viewed `count300_seed42` IDs are development data; full-run primary statistics exclude those IDs without inventing another split name.

```powershell
uv run research_cli experiment --family risk_controlled_trace_mad inspect-experiment --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml --phase count20_seed42 --model dashscope/qwen-flash
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml --phase count20_seed42 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml --phase count300_seed42 --model dashscope/qwen-flash
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad.toml --phase count300_seed42 --model xiaomimimo/mimo-v2.5
uv run python -m research_experiments.families.risk_controlled_trace_mad.fit_router --run-dir <qwen-count300> --run-dir <mimo-count300> --output configs/families/risk_controlled_trace_mad/router/rcta_v1.json
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad_full.toml --phase full_seed42 --model dashscope/qwen-flash
uv run research_cli experiment --family risk_controlled_trace_mad run --experiment configs/families/risk_controlled_trace_mad/experiments/rcta_mad_full.toml --phase full_seed42 --model xiaomimimo/mimo-v2.5
uv run python -m research_experiments.families.risk_controlled_trace_mad.analyze_cross_backbone --run-dir <qwen-full> --run-dir <mimo-full> --output local/reports/risk_controlled_trace_mad/rcta_cross_backbone_full_seed42.json
uv run research_cli experiment --family risk_controlled_trace_mad validate-run --run-dir local/runs/risk_controlled_trace_mad/rcta_mad/count300_seed42/<run-id>
uv run research_cli experiment --family risk_controlled_trace_mad render-report --run-dir local/runs/risk_controlled_trace_mad/rcta_mad/count300_seed42/<run-id>
```

Do not run `full_seed42` unless the generated router artifact has `development_gate_passed=true` and its hash validates.
