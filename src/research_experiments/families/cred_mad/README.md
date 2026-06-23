# cred_mad

`cred_mad` implements CRED-MAD v5: a shrunken Contractual Refutation Evidence
Debate line kept for mechanism validation.

The framework separates three effects that are usually entangled in MAD:

- initial ensemble gain from five independent contract-bearing agents;
- router gain from debating only weak/split Stage A votes;
- debate gain from targeted refutation, defense, and survival-score aggregation.

Current CRED runs use `json_object_answer_v3`: one compact JSON answer card
containing reasoning, answer, confidence, key evidence, and risk fields. CRED
turns use provider JSON-object response formatting and explicit completion caps.
Router risk and evidence quality are diagnostics in v5; they no longer trigger
debate by themselves.

Prompt version `cred_mad_selective_verify_v5` makes every Stage A agent produce a
short answer card. The main debate method is lock-only survival aggregation
(`cred_refute_queue_v1_lock`) with one refutation target and a stricter verified
override gate.

Future SOTA work should go through the separate `cred_v` family. This CRED-MAD
line is intentionally narrow so old debate-heavy logic does not leak back into
the main verifier-centric research path.

Run:

```powershell
uv run research_cli experiment --family cred_mad inspect-experiment --experiment configs/families/cred_mad/experiments/cred_mad_main.toml
uv run research_cli experiment --family cred_mad run --experiment configs/families/cred_mad/experiments/cred_mad_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_mad validate-run --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
uv run research_cli experiment --family cred_mad render-report --run-dir local/runs/cred_mad/cred_mad_main/count20/<run_id>
```
