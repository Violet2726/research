# cred_mad

`cred_mad` implements CRED-MAD: Contractual Refutation Evidence Debate.

The framework separates three effects that are usually entangled in MAD:

- initial ensemble gain from five independent contract-bearing agents;
- router gain from skipping low-value debate;
- debate gain from targeted refutation, defense, and survival-score aggregation.

Current CRED runs use `json_object_answer_v3`: one JSON object containing compact reasoning,
answer, confidence, key evidence, and risk fields. This protocol uses provider JSON-object
response formatting when the selected model supports it.
Router risk is driven only by the structured `risk_level` enum (`none`, `low`, `medium`, `high`);
`risk_summary` is explanatory text and is not parsed as a trigger signal.

Prompt version `cred_mad_json_object_lock_v4` makes every Stage A agent solve with the same
strong single-agent workflow first, then applies the role as an audit lens. The main debate method
is lock-only survival aggregation (`cred_refute_queue_v1_lock`); the older unlocked refutation
branch is removed from the maintained CRED mainline because it introduced harm in count100 results.

Run:

```powershell
uv run research_cli experiment --family cred_mad inspect-experiment --experiment configs/families/cred_mad/experiments/cred_mad_main.toml
uv run research_cli experiment --family cred_mad run --experiment configs/families/cred_mad/experiments/cred_mad_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_mad validate-run --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
uv run research_cli experiment --family cred_mad render-report --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
```
