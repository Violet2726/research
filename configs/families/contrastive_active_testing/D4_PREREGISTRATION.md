# CATCH-Kernel D4 preregistration

Method name: **Risk-Calibrated Proof-Carrying Candidate Completion**.

The sole executable D4 mainline is `catch_kernel_d4_mainline_v3`, implemented
by `protocols/catch_kernel_d4_v3.toml`. It fixes tagged-text Stage-A and the
65,536/65,536/32,768 solver/compiler/judge completion caps. Retired D4 JSON,
answer-first, A/B, high-cap diagnostic, and v1/v2 protocol configurations are
not executable fallbacks.

## Executable state machine

1. **Current terminal state:** the single allowed source-only compiler smoke is
   hash-linked as `failed_blocking_downstream`; it cannot be rerun in a new
   directory.
2. Selection inspection and the live provider audit both require a passing,
   current-mainline smoke artifact before they can read sealed rows or construct
   a provider.
3. The 300-record tagged validation additionally requires all three sealed
   manifests and exact 100/100/100 selection hashes. It uses the project-wide
   `global_validated_response_v3` cache; protocol and selection hashes remain
   bound in the run manifest instead of the cache path.
4. Calibration, semantic IR audit, risk freeze, and one-shot confirmation are
   strictly downstream. No failure permits cap changes, row deletion,
   resampling, or fallback to a retired protocol.

## Primary hypothesis

Under one frozen backbone, no training, one shared Stage-A of five samples, and at most eight logical model calls per method, D4 should expand executable coverage while preserving a calibrated low-harm override policy. Candidate completion may emit a locally solved answer that was absent from the Stage-A candidate pool.

The proof language is strictly conditional: `source -> SourceIRv3 -> local answer`. It is not a proof of natural-language/IR equivalence and never a proof of gold correctness.

## Main methods

The main table contains exactly `SC5`, `fixed-SC8`, `D3-exact-only`, `SSV-raw`, and `D4-full`. Adaptive-SC8 and the historical generic judges are supplementary/development-only, respectively. DirectJudge-3 and PairJudge-3 are forbidden in large confirmation.

## Jurisdiction

- `SequenceTraceKernel`: frozen D3 Dyck/sort/spatial foundation; new shuffled-state exact shadow; word-sort error and temporal trace semantic shadow.
- `EventStateKernel`: typed shadow interface for BBEH structured state plus MuSR object-belief/team-constraint ledgers. No EventState override is authorized until a concrete local operator, per-item metamorphic relation, and independent audit all pass. Murder mysteries remain unsupported for override.
- `ConstraintCalculatorKernel`: explicit truth constraints and closed local calculations. Retrieval, knowledge APIs, and name-to-structure inference are forbidden.

Task names may rule out an open-world task but do not authorize an executable route. Authorization requires a source signature, a closed query operator, a valid answer contract, and a passing risk snapshot.

For a semantic route, SourceIR v3 is the only compiler contract. Trusted host code binds the capability, query operator, answer contract, complete reversible span map, mandatory-span set, and IR hash. The model supplies only candidate-blind entities, facts, events, constraints, query, and `uncovered_span_ids`. Every compiler output must independently parse, solve uniquely, pass its reference checker and applicable metamorphic checks; the three canonical answers must agree. Their surface IR hashes need not match, and all three hashes, solver traces, and audits are retained in the proof package.

## Gates

The one permitted 45-by-3 public-development SourceIR v3 compiler smoke completed on 2026-08-01 and failed its frozen stopping rules: 91/135 outputs passed the SourceIR v3 parser versus the required 122, and none of the nine capabilities produced a sample with three independently verified agreeing proof chains. All 135 requests ended with `stop`, there were no request errors or leakage findings, and the largest completion was 19,193 tokens under the 65,536 cap; the failure is therefore structural rather than a completion-cap shortage. Semantic activation and every downstream formal phase remain stopped. The immutable result and offline diagnosis are retained in `local/runs/contrastive_active_testing/catch_kernel_d4/source_compiler_smoke_v3_20260801` and `local/analysis/D4_SOURCE_IR_V3_SMOKE_20260801.md`; no prompt retry or resampling is authorized under this revision. The runner, selection inspector, and live provider-audit entry point all enforce this terminal smoke gate before data materialization or provider construction.

The first completed D4 output-protocol A/B run on 2026-07-29 failed the original 0.2% parse gate for all three arms (tagged 7/1500, reasoning-first JSON 136/1500, answer-first JSON 219/1500). A later high-cap short-reasoning rerun also failed: reasoning-first JSON had 34/1500 parse failures and lost 6.67 percentage points of SC5 accuracy, while answer-first JSON had 66/1500 failures and lost 16 points. The recomputed evidence is retained in `local/analysis/D4_OUTPUT_PROTOCOL_AB_20260729.md`; neither JSON protocol can authorize calibration or confirmation. The evidence rejects answer-first JSON as the D4 default.

The frozen Stage-A default is therefore the byte-for-byte legacy tagged-text prompt with a 65,536 completion-token safety cap; source-compiler/resample turns use the same cap and judges use 32,768. Cache identity is `request_identity_without_completion_cap_v2`: it excludes only the two completion-cap field names and retains every other generation-semantic field. Requested and origin caps, usage, and finish reason remain in the turn audit. Only successful `stop` responses are reusable; length/repetition truncations, request errors, empty outputs, soft rejections, and D4 contract failures are not shared-cache entries. Each frozen D4 run separately keeps an append-only completion ledger, so an interrupted resume replays an already completed failure rather than making a new request. Deterministic parsing may accept case/space variants of the explicit final-answer label and a unique one-based option ordinal under a known single-choice contract; conflicts, missing answers, duplicate incompatible answers, and label/text mismatches remain fail-closed. Selective cache deletion and re-sampling of observed failures is forbidden as evidence.

The original 0.2% per-turn threshold was an unsupported engineering constant and selected a protocol objective that was empirically anti-correlated with task accuracy. The revised operational contract targets the actual SC5 failure mode: on a fresh, frozen, independent protocol-validation sample, tagged text must have a per-turn parse/request-failure one-sided 95% Clopper-Pearson upper bound below 1%, and the probability that a sample has fewer than three valid answers out of five must also have an upper bound below 1%. Every sample must still have agent IDs 1-5 exactly once, and all failures are abstentions. The validation design is fixed at 300 samples: 100 custodian-sealed records each from a new BBEH extension, MuSR-X, and a deduplicated SuperGPQA Science subset. Its three selection hashes, manifest split, counts, and manifest expectations must be frozen before `sealed_data_ready=true`; the runner rejects missing or mismatched values before any task API call. The development result (7/1500 failures, zero quorum failures among 300 samples; bounds about 0.875% and 0.994%) motivated this revision and is not itself confirmation because the threshold was revised after inspection. A new hash-linked independent validation artifact is mandatory before risk calibration.

Development uses a fixed preregistered family of nine new capabilities. To protect the family of data-selected activations, each capability uses a Bonferroni one-sided Clopper-Pearson bound with alpha `0.05/9`. Consequently, the zero-error minimum for a 0.90 precision lower bound is 50 shadow overrides, not the single-capability value of 29. All applicable, actually executed metamorphic relations must pass. Semantic capabilities additionally require at least 60 independently double-annotated, unique IRs per activated semantic kernel, two distinct raters on every item, Gwet's AC1 at least 0.80, observed critical semantic error at most 2%, observed adjudicated validity at least 95%, independent third-person adjudication of every disagreement, and zero unexplained high-severity false pass. Cohen's kappa and raw agreement are reported, but kappa is not the activation gate because the expected high prevalence of valid IRs can make it undefined or paradoxically low. The two audit error-rate thresholds are quality-control point estimates, not confidence guarantees. A metamorphic field that is absent, hard-coded, or entirely `NOT_APPLICABLE` cannot activate a new route.

The activation artifact must be generated from one completed, error-free, independent post-freeze calibration run, hash-link its manifest, turns, predictions, and any audit file, match the frozen capability registry, and pass count-consistency and source-recomputation checks. Hand-written capability counts or summaries are not accepted by the freeze or confirmation gates. The current capability-stratified engineering sample contains 540 BBEH records, 120 MuSR records, and the 47 locally routed GPQA compatibility records; because these public pools were used while designing and gold-checking the parsers, they are diagnostics only and cannot generate activation evidence. New data must be split into calibration and confirmation before either split is exposed, and the calibration role is fixed as `d4_independent_calibration_after_method_freeze`.

Confirmation activation uses only frozen development evidence. Confirmation gold is not used to tune or activate a route. After the one-shot run, the preregistered pooled claim requires at least 59 overrides, a one-sided precision lower bound at least 0.95, and a one-sided harm upper bound at most 0.05. Fifty-nine is only the zero-error minimum; it is not sufficient when any error is observed. Failure permits only a `shadow` or `high-precision narrow-coverage` interpretation.

Confirmation is a hard gate: `confirmatory=true`, exact selection hashes, a valid component freeze, a passing output-protocol assessment, validated hash-linked development risk evidence, any required semantic blind audit, schema-valid sealed manifests for exactly every preregistered benchmark, a passing live provider audit, and `sealed_data_ready=true` are all required before sample materialization or any API call. Every sealed benchmark must resolve without loader errors to a non-empty sample set. The repository now implements the BBEH-extension, MuSR-X, and SuperGPQA Science loaders plus text-manifest and MuSR-X manifest-v3 validators; real assets, preregistered counts/strata, independent audits, and source hashes are still required before confirmation can launch.

The provider-specific `xiaomimimo_75x95_validated_v1` runtime profile records the previously validated 75-concurrent/95-RPM operating point. It is an execution profile, not part of the logical-call budget, and confirmation still requires a fresh live provider audit.

Before independent validation or later calibration/confirmation, `kernel-d4-provider-audit` must pass ten cache-bypassed live requests, including the 32K and 64K payload caps. This is a transport and payload-contract preflight, not evidence about task accuracy. A failed preflight blocks the larger run; transport failures with zero completion tokens must never be counted as output-protocol parse failures.

## Data independence

- Local BBEH Full 4520 has been used for aggregate route/parser validation and is therefore development/compatibility only. No subset of those same records may be relabeled as calibration or sealed confirmation. A primary BBEH result requires a genuinely new upstream release or an independently generated and custodian-sealed extension that is split into post-freeze calibration and one-shot confirmation before disclosure.
- Existing MuSR 756 and GPQA Diamond 198 are development/compatibility only.
- MuSR-X requires the official repository URL, a full 40-character generator commit, a hashed generation-environment lock, a pinned narrative-generator identity, a hashed quality-validation protocol, and an independent custodian. New latent graphs are split by latent graph hash before rendering (post-freeze calibration 1200, audit 600, confirmation 1200). The manifest must also hash-link the rendered narrative/question/gold asset and its independent render audit; manifest v3 rejects the earlier latent-only v2 contract. The official repository uses model-assisted generation, so changing latent seeds alone is not sufficient evidence of a valid new benchmark.
- Science calibration/confirmation requires previously unanalysed, disjoint text-only, single-choice physics/chemistry/biology SuperGPQA subsets, with SHA-256/MinHash de-duplication against GPQA and SciBench before disclosure. String/hash similarity is only a duplicate screen, not evidence that model-training contamination is absent.
- Confirmation text, gold, and latent graphs remain unreadable before method freeze. A manifest must use an exact allow-listed schema with unique record, question, source-record, and latent hashes; the runner must also verify every linked source/rendered asset and audit hash. Self-declared `contains_text=false` or `contains_gold=false` flags alone are insufficient.
- SuperGPQA is composite and includes transformed material from named prior datasets; source provenance, licenses, and near-duplicate exclusions are reported rather than treating the pool as automatically uncontaminated.

## Statistics

Only D4 vs SC5 and D4 vs fixed-SC8 are preregistered. Across BBEH, MuSR, and science this gives six McNemar tests under one Holm correction; old GPQA is not added as a fourth confirmatory family when SuperGPQA science is present. BBEH uses task-stratified micro accuracy as primary, with adjusted harmonic as a secondary Full-compatibility metric. MuSR uses three-task macro accuracy, and science uses domain-macro accuracy. Bootstrap resampling is within task/domain strata. Report correction/harm, jurisdiction coverage, solver-executable coverage, authorized coverage, selective risk, calls, tokens, latency, cache/network calls, and parse/request failures.
