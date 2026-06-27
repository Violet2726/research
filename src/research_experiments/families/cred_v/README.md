# cred_v

`cred_v` is now the CRED-ACS line: adaptive candidate search with verifiable
aggregation. Stage A uses five structured role contracts, then the router sends
only risky samples to `mimo-v2.5-pro` as a candidate generator. The pro model is
not allowed to act as a one-shot judge.

Initial methods:

- `cred_v_vote_5`: five SC-aligned strong CoT candidates with family voting.
- `cred_v_task_verify_v3`: runs one task verifier for split votes, then promotes
  a challenger only when the verifier certificate passes score, confidence, and
  concrete-evidence gates. This is retained as a legacy self-verifier baseline.
- `cred_verify_safe_v1`: promotes challengers only through deterministic repairs,
  rule/tool verification, or a verifier model variant such as `mimo-v2.5-pro`.
  Same-model verifier promotion is disabled by default. This is retained as a
  legacy safety baseline.
- `cred_acs_v1`: expands candidates with math repair, Hotpot span extraction,
  multiple-choice option shuffling, or StrategyQA dual-polarity checks. Final
  promotion requires deterministic repair or at least two independent expansion
  supports with a positive aggregation margin.

Run the screening phase:

```bash
uv run research_cli experiment --family cred_v inspect-experiment --experiment configs/families/cred_v/experiments/cred_v_main.toml
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_v validate-run --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
uv run research_cli experiment --family cred_v render-report --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
```
