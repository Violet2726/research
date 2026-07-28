# CATCH-Kernel D3 预注册边界

## 研究问题

D3 检验的是：在同一 MiMo-v2.5 backbone、共享 Stage-A cache、每个方法至多 8 次模型调用的条件下，确定性可执行 jurisdiction 是否能以可审计的 conditional certificate 改善风险–准确率 Pareto 前沿。D3 不把 solver 输出称为原题 gold 的证明。

## 固定数据流

1. 每题先运行 5 次 Stage-A；确定性 capability registry 选择 `EXACT_EXECUTABLE`、`SEMANTIC_COMPILABLE` 或 `SOFT_UNSUPPORTED`，不跨 jurisdiction fallback。
2. Exact route 只使用 source-only parser 和本地 solver；候选验证在 solver 之后，`solver_direct` 与 `candidate_completion` 分开记录。
3. Semantic route 最多 3 次 candidate-blind compiler 调用；IR 必须闭合、span 完整、canonical IR 一致、reference checker 通过，否则保留 anchor。默认 semantic override 关闭。
4. Soft route 不做 generic self-judge，也不把额外 resample 当作 D3 干预；冻结默认 `soft_fallback = "stage_a_anchor"`，直接保留 Stage-A anchor。fixed-SC8 与 adaptive-SC8 只作为独立 comparator，不进入 D3 主方法。

DirectJudge-3、PairJudge-3、fixed-SC8 和 adaptive-SC8 与 D3 共用 Stage-A 结果，但各自的实际方法成本均不得超过 8 次调用。运行器可以在同一批次中物理收集 comparator rows；统计时按方法分别计费。

独立 confirmation 的必需 paired comparators 为 SC5、fixed-SC8、adaptive-SC8 和 D3。fixed/adaptive-SC8 共用三次 resample，因而不额外重复采样。DirectJudge-3、PairJudge-3 只保留为开发阶段的负对照：它们在 count100 已呈净 harm，故不在 916 题确认集中再消耗每题六次额外调用。

## 数据角色

- D3 development：BBEH/MuSR 使用从已查看开发+heldout池构造的 nested task-stratified count50/count100；GPQA 使用已查看 dev98 的 domain-stratified count50。
- D3 primary mechanism confirmation：BBEH 460、MuSR 356、GPQA 100，均与既有查看池 disjoint；用于独立机制确认，不包装为公开 benchmark SOTA。
- secondary benchmark compatibility：`catch_kernel_d3_benchmark_compat.toml` 使用官方 BBEH Mini 460、MuSR 全量756、GPQA Diamond全量198。官方 Mini 与既有400题重合45题，因此只作 benchmark-compatible 次级结果。

BBEH Mini 主指标为 micro accuracy；只有完整 BBEH Full 才使用官方 adjusted harmonic mean。MuSR 主报三任务 macro accuracy，GPQA 报 accuracy及领域分层。

## 启用门槛

Exact route 在 source parser、unique solver、canonicalization 通过时启用；候选池外的答案只能走 `candidate_completion`，不能由 Stage-A 触发。Semantic route 只有在开发集 route-specific precision 单侧95%下界 > 0.5、metamorphic audit 和至少60个 IR 双人盲审通过后，才允许将 `semantic_override_enabled` 从 false 改为 true；当前配置保持 false，且 semantic shadow 也关闭。Soft auditor 同样默认关闭。

## 必报审计

每个 run 保存 SourceIR、candidate evaluation、solver certificate、KernelDecision、request/parse failure、first-failure layer、API/token/latency/cache hit，以及 candidate blindness、选项置换、实体重命名、无关文本插入、多解/无解测试。所有 override 需要同时报告 correction precision、harm 和 Clopper–Pearson 单侧区间。

## 结论措辞

若结果成立，最强可支持的表述是：D3 在可形式化且可审计的 jurisdiction 内提供 same-backbone、no-training、equal-call-budget 的 risk–accuracy Pareto improvement。不得声称 solver 自动解决了自然语言到形式语义的 gap；semantic 结果只能称为 compiler-backed conditional certificate。
