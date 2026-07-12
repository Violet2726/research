# A-SMAD (historical family)

`adaptive_sparse_mad` is a frozen historical mechanism family. It remains in the repository so that earlier experiments and artifacts can be reproduced, but it is no longer an active innovation line. Do not add V10/V11-style variants here; new MAD mechanism work belongs in the unified `risk_controlled_trace_mad` versioned experiment.

## Preserved entry points

- `same_context_main_v5.toml`: original same-context configuration.
- `same_context_main_v6.toml`: later sparse rescue/probe variants.
- `same_context_full_counterfactual_v1_screen.toml`: historical multi-dataset screen.
- `same_context_full_counterfactual_v1.toml`: historical multi-dataset evaluation.

## Historical interpretation

The archived A-SMAD runs did not establish a validated advantage over the matched self-consistency controls. In particular, later versions must not be cited as evidence of a causal debate gain when validation failed or when no answer was actually corrected. Preserve the implementation and reports for auditability; evaluate all new claims on the BRD-MAD line with pre-registered candidate-oracle and safety diagnostics.

## Reproduction only

```powershell
uv run research_cli experiment --family adaptive_sparse_mad inspect-experiment --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml
uv run research_cli experiment --family adaptive_sparse_mad run --experiment configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml --phase count100 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family adaptive_sparse_mad validate-run --run-dir local/runs/adaptive_sparse_mad/<experiment>/<phase>/<run_id>
uv run research_cli experiment --family adaptive_sparse_mad render-report --run-dir local/runs/adaptive_sparse_mad/<experiment>/<phase>/<run_id>
```

All source, configuration, and report files are UTF-8. The repository encoding test rejects common mojibake signatures in maintained documentation.
