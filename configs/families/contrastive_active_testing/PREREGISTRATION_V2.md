# CATCH v2 preregistration

## Structural preflight

Development begins inside the ordinary `catch_gate/development/<run_id>` run. Stage-A is evaluated for the frozen
dev100 split, after which 20 disagreement samples are selected without gold by seed-42 task-stratified hashing. The
designer stage must achieve 100% top-level schema parsing, at least 95% unique evidence-quote alignment, zero leakage,
and at least 60% `d_min >= 2` code-packet coverage. Only then are two blinded witnesses run on eligible packets. Their
top-level parse rate must be 100%, valid-coordinate rate at least 95%, and usable double-panel rate at least 90%.

Preflight does not inspect gold labels. Failure terminates the run and forbids dev100, freezing, and held-out execution.
Successful preflight responses are reused through their exact cache keys by the full development stage.

## Fixed inference protocol

- Five shared Stage-A solver calls and three adaptive-SC8 resamples use the unchanged v1 payloads.
- A designer emits at most six pair-targeted finite contrast atoms. Every non-null commitment contains a unique exact
  `evidence_quote`; the runner computes and hashes source spans after NFKC and newline normalization.
- The program selects at most four coordinates by the frozen max-min objective. A packet below the current `d_min`
  abstains without witness calls.
- A legal packet invokes two independently permuted blinded witnesses. Candidate data, votes, traces, commitments, and
  mappings remain hidden. Invalid rows are erasures; malformed top-level JSON fails the panel.
- CATCH consumes at most three intervention calls. The decoder remains candidate-restricted and requires one identical
  challenger to pass both panels under `d_min in {2,3,4}` and `margin in {1,2}`.
- DirectJudge-3 and the existing accuracy, precision, net-correction, token, held-out, and human-audit gates remain
  unchanged. Actual tokens and physical network attempts are the cost endpoints.

## Version and stopping policy

CATCH-v2 reuses the `contrastive_active_testing` family and `catch_gate` experiment directory. Manifests, configuration
hashes, prompt/schema versions, and v2 cache namespaces identify the active protocol. CATCH-v1 artifacts remain
immutable and cannot authorize held-out execution. A v2 preflight or dev failure is reported as a mechanism boundary;
it is not repaired using held-out labels.
