# CATCH-ICV

The active registered study is now the non-confirmatory
`catch_cross_domain_boundary_audit`. The preregistered BBEH CATCH-v3 structural
preflight is terminally recorded as `failed_structural_preflight`; the new study
maps whether that failure is BBEH-specific and cannot authorize heldout or full
confirmation. See `PREREGISTRATION_BOUNDARY_AUDIT.md`.

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

1. Restore the two new pinned public assets if they are absent:

   `research_cli tools dataset-assets download-used`

2. Inspect the non-confirmatory experiment without network access:

   `research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_cross_domain_boundary_audit.toml`

3. The one-shot runner reuses the already passing MiMo provider audit. If the
   audit file is absent on a fresh server, the same command first performs the
   required ten live, cache-bypassed checks and stops before the scientific run
   if any provider-contract condition fails.

4. Run the four-dataset audit once:

   `research_cli experiment --family contrastive_active_testing run --experiment configs/families/contrastive_active_testing/experiments/catch_cross_domain_boundary_audit.toml --phase boundary_audit`

   The command screens 100 items per dataset, selects at most twenty Stage-A
   disagreements without gold, runs all matched methods, writes a checkpoint
   after every dataset, and exits. It never dispatches heldout or confirmation.

5. After completion, fill `diagnostics/human_audit_sample.json` with two blind
   annotations if the manuscript will discuss coordinate validity. The labels
   explain mechanism validity and cannot select prompts, samples, or thresholds.

   Install and independently recompute the completed annotations with:

   `research_cli experiment --family contrastive_active_testing boundary-human-audit --run <run-directory> --input <completed-human-audit.json>`

Do not monitor, poll, or auto-resume any of these commands. Each process is
finite and terminal; inspect its artifacts once after the user reports that it
has completed.

## Cache and compute contract

The audit writes to four isolated `catch-boundary-v3-*` namespaces. BBEH may
read byte-identical v3 selector/witness results and v1/v3 shared-solver results
from read-only predecessor namespaces. MuSR, seqBench, and GPQA never reuse
intervention responses across datasets or study versions.

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
