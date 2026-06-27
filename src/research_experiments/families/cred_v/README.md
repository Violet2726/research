# cred_v

`cred_v` is now the CRED-RFS line: reasoning-first selective compute.
Stage A uses five free-text role-guided solvers, then only weak-split samples
receive adaptive extra candidates. `mimo-v2.5-pro` is used as a candidate
generator for multiple-choice shuffle checks, never as a one-shot judge.

Main methods:

- `cred_rfs_vote_5`: five free-text role-guided candidates with family voting.
- `cred_rfs_adaptive_sc_v1`: weak-split selective compute with extra free-text
  solvers, conservative MC shuffle support, deterministic repair, and strong
  majority locking.

Legacy methods are retained in `configs/families/cred_v/experiments/cred_v_legacy.toml`
for failure analysis only: `cred_v_vote_5`, `cred_v_task_verify_v3`,
`cred_verify_safe_v1`, and `cred_acs_v1`.

Run the screening phase:

```bash
uv run research_cli experiment --family cred_v inspect-experiment --experiment configs/families/cred_v/experiments/cred_v_main.toml
uv run research_cli experiment --family cred_v run --experiment configs/families/cred_v/experiments/cred_v_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli experiment --family cred_v validate-run --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
uv run research_cli experiment --family cred_v render-report --run-dir local/runs/cred_v/cred_v_main/count20/<run_id>
```
