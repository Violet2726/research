# cred_mad

`cred_mad` implements CRED-MAD: Contractual Refutation Evidence Debate.

The framework separates three effects that are usually entangled in MAD:

- initial ensemble gain from five independent contract-bearing agents;
- router gain from skipping low-value debate;
- debate gain from targeted refutation, defense, and survival-score aggregation.

Run:

```powershell
uv run research_cli experiment --family cred_mad inspect-experiment --experiment configs/families/cred_mad/experiments/cred_mad_main.toml
uv run research_cli experiment --family cred_mad run --experiment configs/families/cred_mad/experiments/cred_mad_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_mad validate-run --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
uv run research_cli experiment --family cred_mad render-report --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
```
