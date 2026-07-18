# CATCH-ICV

CATCH-ICV is the active `catch_v3` protocol in the existing
`contrastive_active_testing` family. It converts the plurality anchor and at
most two strongest Stage-A challengers into pair-local indexed reasoning
contrasts. A selector may choose IDs only; two blinded witnesses compare the
selected statements; the runner applies the fixed two-of-three, two-panel,
unique-challenger decoder.

`catch_v1` and `catch_v2` are immutable failed predecessors. They remain in
the same family and run tree for audit, but cannot be rerun, frozen, or used as
active gate candidates. Their exact failure registrations are in
`versions.toml`. The frozen v3 scientific contract is in
`PREREGISTRATION_V3.md`.

## Reproducible one-shot order

1. Inspect the active experiment without network access:

   `research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_gate.toml`

2. Run the cache-bypassed live provider audit. Any payload, usage, finish,
   retry, or attempt-timeline failure blocks all subsequent commands:

   `research_cli experiment --family contrastive_active_testing provider-audit --experiment configs/families/contrastive_active_testing/experiments/catch_gate.toml`

3. Run the one-shot 20-disagreement structural preflight as a separate
   terminal process:

   `research_cli experiment --family contrastive_active_testing structural-preflight --experiment configs/families/contrastive_active_testing/experiments/catch_gate.toml`

   It always stops after the preflight and writes terminal `progress.json`,
   `run_validation.json`, `diagnostics/preflight.json`, and the 40-coordinate
   blind-audit sample. A failed or already-attempted v3 preflight permanently
   blocks another v3 attempt; it never falls through into dev100.

4. Only if the machine preflight passes, complete the two-annotator blind
   audit. Write the adjudicated summary to
   `configs/families/contrastive_active_testing/frozen/catch_v3_preflight_human_audit.json`
   using `preflight_human_audit_schema.example.json`. The source run ID and
   full config hash must exactly match the preflight artifacts. The gate also
   requires the exact 40 coordinate hashes, two complete item-level boolean
   label sets, and adjudication of every disagreement; it recomputes all rates
   and pooled non-leakage Cohen's kappa instead of trusting entered summaries.

5. Only after that audit passes, run dev100 once:

   `research_cli experiment --family contrastive_active_testing run --experiment configs/families/contrastive_active_testing/experiments/catch_gate.toml --phase development`

6. Only for a fully passing dev100, freeze its exact protocol/config/split
   candidate with `freeze-development`, then run heldout200 once with
   `--phase heldout`. Run confirmation only if heldout passes; v3 inherits the
   exact-config 40-coordinate record-level audit completed before dev100 and
   does not substitute the retired v1/v2 100-item summary audit.

Do not monitor, poll, or auto-resume any of these commands. Each process is
finite and terminal; inspect its artifacts once after the user reports that it
has completed.

## Cache and compute contract

The active intervention namespace is `catch-dev-v3` (then
`catch-heldout-v3`/`catch-confirm-v3`). Only byte-identical Stage-A, adaptive
resample, and DirectJudge payloads may read exact hits from the read-only
`catch-dev-v1` fallback. ICV selector/witness and PairJudge responses cannot
cross protocol namespaces.

On a disagreement, each reported comparison method uses the same five shared
Stage-A calls and at most three method-specific calls. CATCH uses selector plus
two witnesses only for an eligible packet and abstains early otherwise.
Adaptive-SC8, DirectJudge-3, and PairJudge-3 each use three calls. Scientific
cost gates use actual input, output, and reported reasoning tokens.

## Artifact contract

Every run starts with `progress.json` and ends with terminal `progress.json`
plus `run_validation.json`, even on scientific failure, validator exception,
packaging failure, cancellation, or futility. Turns record real payloads,
cache namespace/source, logical and physical attempts, retry timing, provider
request ID, finish reason, actual usage, evidence/codebook, permutations,
witness vectors, and deterministic decoder diagnostics. Archived v3 turns
must independently reproduce predictions, target oracle, overrides, calls,
tokens, and gate results.

The confirmation BBEH population is the version-controlled
`full4520_seed42` population minus the disjoint `dgcr_dev100_seed42` and
`dgcr_holdout200_seed42` manifests, yielding 4,220 samples.
