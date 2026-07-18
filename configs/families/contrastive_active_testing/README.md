# CATCH-ICV

CATCH now uses a best-effort execution policy. Scientific thresholds, provider
audits, human audits, and prior phase results are descriptive evidence only;
they do not authorize or block a run. The indexed-contrast selector, blinded
witnesses, abstention rules, and fixed v3 decoder are unchanged.

## Running experiments

Inspect a configuration without network access:

`research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_cross_domain_boundary_audit.toml`

Run the four-dataset experiment:

`research_cli experiment --family contrastive_active_testing run --experiment configs/families/contrastive_active_testing/experiments/catch_cross_domain_boundary_audit.toml --phase boundary_audit`

The runner screens BBEH, MuSR, seqBench, and GPQA independently. A failed
request, sample, or dataset is recorded and the remaining work continues. The
configured network-attempt count is a visible warning threshold; the 18 RPM
limiter remains the actual admission control. Re-running creates a new run and
reuses only exact successful cache entries.

Development, heldout, and confirmation can also be invoked directly with
`catch_gate.toml`. Missing preflight, frozen decoder, human audit, or earlier
phase result is recorded as a warning. For v3 the built-in fixed decoder is
used; legacy v1/v2 default to `d_min=2, margin=1` when no decoder file exists.

## Result files

New runs keep a compact result contract:

- `manifest.json` and terminal `progress.json`;
- raw `turns/agent_turns.jsonl` and `turns/router_decisions.jsonl`;
- `views/predictions.jsonl`, `views/metrics.json`, and `views/run_summary.json`;
- researcher-facing `report.md`;
- compatibility-only, non-blocking `run_validation.json`.

The report gives planned, attempted, evaluable, and missing denominators; both
complete-case and missing-as-wrong accuracy; paired method comparisons; request
and parse failures with a Wilson 95% interval; mechanism diagnostics; and
cache/network/token costs. No `gate.json`, preflight artifact, manual-audit
artifact, or archive-integrity package is required for a new run.

Historical CATCH-v1/v2/v3 runs and their old audit artifacts remain read-only.
The optional `canonicalization-replay` command is retained only for reproducing
those archived results.
