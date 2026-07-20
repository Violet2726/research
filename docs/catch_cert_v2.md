# CATCH-Cert v2：答案连接的全局义务证书

## 方法不变量

- v1 artifact、prompt 和 decoder 保持只读可重放；v2 使用独立协议、prompt、schema 和 cache namespace。
- Stage-A 仍为5个 MiMo-v2.5、thinking-off solver；最多增加1次 designer 和2次 blinded verifier。
- 所有 Stage-A 候选均与 anchor 配对，最多4个 challenger；不读取 gold、不使用票数生成证书。
- designer 可见匿名候选答案的真实语义，但 verifier 不可见候选身份、答案或 commitments。
- 每张证书必须绑定 `answer_hash` 并覆盖题目的全部 mandatory obligations。
- `earliest` 必须同时证明候选步骤错误和完整前缀有效；`final_state`、`argmax`、`exact_set`、`exact_sequence` 必须覆盖相应全局义务。
- refutation tests 由 compiler 从不相容 candidate commitments 中推导，模型不再输出自由文本 `refutation_condition`。
- verifier 只能引用预先编号的原文 span；`UNDERDETERMINED` 永不触发覆盖。
- 可执行 adapter 不搜索答案，只执行 certificate 明确给出的候选计划、状态更新、方程或约束 witness。

## 开发工作流

1. 从冻结的 v1 development run 生成120题机制审计队列：

```powershell
uv run research_cli experiment --family contrastive_active_testing cert-v1-audit `
  --run local/runs/contrastive_active_testing/catch_cert_development/development/<run_id> `
  --output-dir local/analysis/catch_cert_v1_mechanism_audit
```

2. 双人完成审计 JSON，并把 `seqbench_executor_golden_tests_passed` 设为 `true`。

3. 生成120题×4 cell 的2×2机制实验模板：

```powershell
uv run research_cli experiment --family contrastive_active_testing cert-v2-factorial-template `
  --audit local/analysis/catch_cert_v1_mechanism_audit/catch_cert_v1_mechanism_audit.json `
  --output local/analysis/catch_cert_v1_mechanism_audit/catch_cert_v2_factorial_template.json
```

4. 先运行24题分层 pilot，并依据失败样本决定完整 development 的分析重点；任何推荐条件未满足都不会自动停止实验：

```powershell
uv run research_cli experiment --family contrastive_active_testing run `
  --experiment configs/families/contrastive_active_testing/experiments/catch_cert_v2_development.toml `
  --phase development --model xiaomimimo/mimo-v2.5
```

5. development 后生成非阻断式 readiness assessment：

```powershell
uv run research_cli experiment --family contrastive_active_testing assess-cert-v2-readiness `
  --run local/runs/contrastive_active_testing/catch_cert_v2_development/development/<run_id> `
  --audit local/analysis/catch_cert_v1_mechanism_audit/catch_cert_v1_mechanism_audit.json `
  --output local/analysis/catch_cert_v2_readiness_assessment.json
```

该 assessment 只用于解释证据，不阻止任何阶段启动。缺失、未满足、损坏或配置 hash 不匹配时，运行仍会继续，但 manifest 和中文报告会将结果标为探索性诊断证据并列出未满足项。单个失败样本始终保留在 artifact 中，可用于错误归因和后续假设生成；推荐条件不能作为删除样本或终止实验的依据。

## 报告口径

中文报告同时给出 micro accuracy、Wilson 区间、McNemar、Holm、四类转移、平均/中位/P90 token、调用数、每千 token 正确题数、token/correct、candidate/target oracle、答案连接覆盖、问题义务覆盖、adapter/verifier 漏斗和 headroom utilization。

BBEH 的样本级 micro accuracy 用于 McNemar；任务级 harmonic mean 单独报告，禁止混用。
