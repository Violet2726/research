# cred_mad

`cred_mad` implements CRED-MAD v6: a shrunken Contractual Refutation Evidence
Debate line kept for mechanism validation.

The framework separates three effects that are usually entangled in MAD:

- initial ensemble gain from five SC-aligned strong CoT candidates;
- router gain from debating only weak/split Stage A votes;
- debate gain from targeted refutation, defense, and survival-score aggregation.

Current CRED runs split generation from verification. Stage A uses
`free_text_answer_v1` and the same strong CoT prompt family as `sc_5`, so the
candidate pool is not weakened by the verifier protocol. Refutation, defense,
and judge turns use `json_object_answer_v3` as compact verification
certificates. Router risk and evidence quality are diagnostics; they no longer
trigger debate by themselves.

Prompt version `cred_mad_sc_aligned_selective_verify_v6` aligns Stage A with
self-consistency and keeps structure only for verifier turns. The main debate method is lock-only survival aggregation
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
