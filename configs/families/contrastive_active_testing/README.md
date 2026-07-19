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
