# CATCH-ICV

## CATCH-Cert

`catch_cert_v1` 是独立于冻结 CATCH-v3 的问题条件证书协议。它复用五个
Stage-A 候选，先构建 `TaskContract` 与带 provenance 的 `ClaimGraph`，再使用
一次 certificate designer 和两个 blinded verifier 验证候选的必要条件与反例。
证书只允许在现有候选中选择，确定性 adapter 不读取 gold、不搜索新答案，也不
调用外部模型。GPQA、BBEH、MuSR 和 seqBench 的任务结构分别映射到 equation、
proof-state、state-transition、set/count 或 semantic adapter。

配置检查：

`research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_cert_development.toml`

运行开发阶段：

`research_cli experiment --family contrastive_active_testing run --experiment configs/families/contrastive_active_testing/experiments/catch_cert_development.toml --phase development`

同一候选池上的冻结 CATCH-v3 对照使用
`configs/families/contrastive_active_testing/experiments/catch_cert_v3_baseline.toml`；其
Stage-A 读取 CATCH-Cert 的只读 cache namespace，便于做同题、同候选集合的 offline replay
和 paired comparison。

证书方法的干预预算固定为 3 次（designer + 两个 verifier）；adaptive-SC8、
DirectJudge-3 和 PairJudge-3 的调用作为独立基线计费。报告固定输出中文的平均每题
token、每千 token 准确率、token/correct、调用数、wrong→correct、
correct→wrong、wrong→wrong、correct→correct、证书覆盖率和 headroom utilization。
seqBench 同时报告 exact match、progress ratio、precision、recall、合法动作率、
执行前缀比例和首个非法动作原因。

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

## CATCH-Kernel D3

The revised source-blind implementation is registered in
`experiments/catch_kernel_d3.toml`. It routes every item deterministically,
keeps exact/semantic/soft jurisdictions exclusive, and reports solver-direct
and candidate-completion decisions separately. Development uses nested
task/domain-stratified count50/count100 manifests; the primary confirmation is
unseen BBEH/MuSR/GPQA. The official BBEH Mini and full MuSR/GPQA compatibility
run is deliberately separate because the official Mini overlaps 45 of the 400
previously inspected BBEH items.

Use `experiments/catch_kernel_d3_count50.toml` to freeze the first development
slice, then `catch_kernel_d3.toml` for count100/heldout and unseen confirmation.
Use `catch_kernel_d3_benchmark_compat.toml` only for benchmark-compatible
secondary reporting. See `D3_PREREGISTRATION.md` for metric, risk, and claim
boundaries.

## CATCH-Kernel

`catch_kernel_v1` treats the language model as an untrusted proof producer.
The local task-semantics registry owns query semantics and typed obligation
meanings; verifier capability manifests determine which verifier may decide
each operation. Executable `CONFLICT` and `UNSUPPORTED` results are terminal
per-case abstentions and can never fall back to a model verifier. This is a
per-case proof rule, not a global experiment gate.

Kernel v3 components compile every pair, finite outcome, answer hash,
candidate commitment, and refutation locally. Bounded-semantic obligations
use independent TRUE/FALSE/UNKNOWN proposition tests rather than pairwise
candidate outcomes. Operations whose source semantics cannot yet be compiled
locally are not granted executable jurisdiction. Reports call structural
success `typed_compilation_validity`; semantic and contract accuracy remain
pending until human or deterministic adjudication instead of being inferred
from a successful parse.

New Kernel verifier calls use a 16384-token completion ceiling. Historical
DirectJudge and PairJudge calls retain their exact 4096-token request identity
and read through the frozen CATCH-Cert-v2 cache.

The confirmation phase excludes both inspected splits. MuSR is balanced by
subtask, seqBench is interleaved by backtracking count, noise ratio, and
logical-depth decile, and BBEH uses a frozen seed-42 hash. D1 records selected
ID hashes; after D2, materialize the exact components and IDs before any
confirmation call:

`research_cli experiment --family contrastive_active_testing freeze-kernel-d2 --experiment configs/families/contrastive_active_testing/experiments/catch_kernel_d1.toml --output local/analysis/catch_kernel_d2_freeze.json`

Inspect D1 without network access:

`research_cli experiment --family contrastive_active_testing inspect-experiment --experiment configs/families/contrastive_active_testing/experiments/catch_kernel_d1.toml`

Build the 832-case offline causal ledger and the frozen 232-case intensive
audit set:

`research_cli experiment --family contrastive_active_testing kernel-causal-ledger --run <development-run> --run <heldout-run> --output-dir local/analysis/catch_kernel_causal_ledger`

Create the predeclared representation, contract/verifier 2x2, jurisdiction,
and proof-completeness matrix:

`research_cli experiment --family contrastive_active_testing kernel-mechanism-template --audit local/analysis/catch_kernel_causal_ledger/intensive_audit.json --output local/analysis/catch_kernel_causal_ledger/kernel_mechanism_matrix.json`

Arm runners write JSON/JSONL results that are merged only when the frozen
case, arm, and candidate-set hash match:

`research_cli experiment --family contrastive_active_testing ingest-kernel-mechanism-results --matrix <matrix.json> --result <arm-results.jsonl> --output <merged.json>`

After D1, materialize the capability-routed arm and all three proof-decoder
ablations from the same proof objects without API calls:

`research_cli experiment --family contrastive_active_testing kernel-run-mechanism-results --matrix <matrix.json> --run <development-kernel-run> --run <heldout-kernel-run> --output <kernel-arm-results.jsonl>`

After separate frozen v3 and Cert-v2 runs reuse the same Stage-A candidates,
merge their predictions with an exact five-candidate signature check:

`research_cli experiment --family contrastive_active_testing merge-kernel-comparators --primary-run <kernel-run> --comparator-run <v3-run> --comparator-run <cert-v2-run> --output <comparison.jsonl>`

Completed counterfactual annotations are summarized separately; pending
annotations never appear as a causal decomposition:

`research_cli experiment --family contrastive_active_testing summarize-kernel-counterfactuals --ledger <causal_ledger.jsonl> --output <counterfactual_summary.json>`

Kernel reports keep accuracy and cost metrics while adding syntax, schema,
semantic validity, verifier-jurisdiction coverage, proof completeness, proof
status counts, and first-failure layers. Existing CATCH-Cert v2 artifacts and
decoder behavior are unchanged and remain replayable.
