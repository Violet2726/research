# CATCH

CATCH (Contrastive Active Testing for Candidate Hypotheses) is the independent successor to DGCR. It treats the
valid Stage-A answer classes as finite hypotheses, asks a designer to commit those hypotheses to discrete diagnostic
outcomes, hides every candidate-side field from two witnesses, and performs deterministic candidate-restricted
decoding.

`catch_v1` is frozen as a failed-futility predecessor: on 87 completed dev samples it produced no override and could
not possibly reach either the code-coverage or structured-output gate. `catch_v2` remains in this same family and run
directory. It replaces model-generated numeric offsets with uniquely aligned evidence quotes, treats bad witness rows
as erasures, and requires a gold-free 20-question structural preflight before the full development run.

The registered primary runner never invokes the unblinded or vote-aware ablation prompts.  It also never makes a live
request unless the user explicitly runs `provider-audit` or `run`.

## Reproducible order

1. Inspect without network access:

   `research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_gate.toml`

2. Explicitly run the uncached provider audit.  A failed audit blocks every phase.
3. Run `development`; the same run first writes `diagnostics/preflight.json` and stops before dev100 if the structural
   channel is infeasible. Inspect `progress.json`, `run_validation.json`, `diagnostics/gate.json`, and
   `diagnostics/frozen_decoding_candidate.json`.
4. Only for a passing run, invoke `freeze-development`.  This is the sole command that copies a development candidate
   into the configured frozen decoder path.
5. Run `heldout` once.  Its first response fixes the held-out protocol and cache namespace.
6. Complete the two-annotator audit using `human_audit_schema.example.json`.
7. Run `confirmation` only when development, held-out, validation, and human-audit gates all pass.

The confirmation BBEH selection is the version-controlled `full4520_seed42` population minus the disjoint
`dgcr_dev100_seed42` and `dgcr_holdout200_seed42` manifests, producing exactly 4,220 samples.

## Artifact contract

Every run starts with a live `progress.json`.  Successful and failed terminal paths write `run_validation.json`.
Turns contain the real payload, namespace, cache source/key, seed, actual usage, reasoning tokens, finish reason,
retries/network attempts, test codebook, witness permutations/vectors, and decoder diagnostics.  A performance-gate
failure remains a scientifically inspectable completed run; request/usage or artifact violations fail run validation.

Baseline roles may read exact-payload hits from the read-only `catch-dev-v1` predecessor namespace. All misses and all
v2 designer/witness responses are written only to `catch-dev-v2`; every turn records both lookup and write namespaces.
Use `finalize-partial --run <path>` to recover a hard-stopped historical run into an explicit failed terminal artifact.
