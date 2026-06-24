# cred_v

`cred_v` is the verifier-centric successor line for CRED. It uses strong
free-text CoT candidates for Stage A and JSON verification certificates for
selective refutation, rather than open-ended multi-agent debate.

Initial methods:

- `cred_v_vote_5`: five SC-aligned strong CoT candidates with family voting.
- `cred_v_selective_verify_v1`: runs a verifier queue only for split votes, then
  allows overrides only when refutation and judge evidence support the challenger.

Run the screening phase:

```bash
uv run research_cli experiment --family cred_v inspect-experiment --experiment configs/families/cred_v/experiments/cred_v_main.toml
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_v validate-run --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
uv run research_cli experiment --family cred_v render-report --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
```
