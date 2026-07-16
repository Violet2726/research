# BRD-MAD: Blind Review--Reconstruct Debate

`blind_reconstructive_mad` is the active MAD innovation line. It replaces further versioning of A-SMAD, CRED-MAD, and CRED-V; those families remain historical, reproducible controls.

## Frozen V1 mechanism

1. Run five Stage-A free-CoT samples with the exact `sc_5` prompt, temperature, and seed rule.
2. Exit on a 5-0 consensus. For disagreement, group normalized final answers and show one representative rationale per group.
3. Hide support counts, source order, and majority identity. Independently permute anonymous labels for each of three reviewers.
4. BRD reviewers falsify each candidate's first decisive step, then reconstruct the answer from scratch. `conditional_resample_3` has no candidate board; `gsa_quorum_3` sees the board but does not receive the explicit falsification instruction.
5. A 4-1 Stage-A split can change only with 3/3 review support for an existing minority. All other disagreement patterns use a 2/3 existing-candidate quorum. Novel review answers remain `shadow` diagnostics.

The empirical target is net safe coverage:

`ΔAcc = P(anchor wrong and correct override) − P(anchor correct and wrong override)`.

The report estimates reviewer error correlation, effective reviewer count, actual quorum error, candidate-oracle gap, corrected/harmed counts, token cost, latency, and paired statistics. The IID 2-of-3 expression `3e² − 2e³` is never treated as an empirical guarantee.

## Phases

- `brd_mad_pilot.toml`: Qwen-Flash pilot on the shared `count100_seed42` prefixes of Omni-MATH-2 and BBEH. It includes the visible-support ablation.
- `brd_mad_locked.toml`: frozen full confirmation (`full1000_seed42` Omni-MATH-2, `full4520_seed42` BBEH) plus `full198_seed42` GPQA transfer. It excludes the visible-support pilot ablation.

Do not start the locked run unless the pilot has zero request/protocol failures, at least a 3pp candidate-oracle gap on both primary datasets, positive net BRD corrections versus both `sc_5` and `conditional_resample_3`, at least 20 overrides, and override precision at least 2/3.

```powershell
uv run research_cli experiment --family blind_reconstructive_mad inspect-experiment --experiment configs/families/blind_reconstructive_mad/experiments/brd_mad_pilot.toml
uv run research_cli experiment --family blind_reconstructive_mad run --experiment configs/families/blind_reconstructive_mad/experiments/brd_mad_pilot.toml --phase pilot --model dashscope/qwen-flash
uv run research_cli experiment --family blind_reconstructive_mad validate-run --run-dir local/runs/blind_reconstructive_mad/<experiment>/<phase>/<run_id>
uv run research_cli experiment --family blind_reconstructive_mad render-report --run-dir local/runs/blind_reconstructive_mad/<experiment>/<phase>/<run_id>
```
