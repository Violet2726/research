# Vanilla MAD Clean Smoke 实验报告

## 1. 实验目的

本次实验的目标不是追求最优成绩，而是验证一条最小、干净、可复现的传统 debate 基线是否已经跑通，并初步回答：

- 在 `2 agent + 1 round all-to-all debate + final majority vote` 的最小设置下，Vanilla MAD 相比等预算 `mv_4` / `sc_4` 是否有稳定收益。
- 在多智能体协议下，当前项目的独立配置链、运行链、日志链和验证链是否完整可用。

本次结论仅适用于 `smoke20` 规模，不应直接外推到 `pilot100` 或主实验。

## 2. 实验配置

- 实验配置：[vanilla_mad_clean_smoke.toml](/d:/user/research/configs/multi_agent/experiments/vanilla_mad_clean_smoke.toml)
- Phase：`smoke20`
- Benchmark：`GSM8K + StrategyQA + HotpotQA`
- 主方法：`mad_2a_r1`
- 等预算对照：
  - `cot_1`
  - `sc_4`
  - `mv_4`
- 每个 backbone 总题量：`60`
- 每个 backbone 总预测条数：`240`
- 每个 backbone 计划总调用数：`780`

本次实际运行的 backbone：

- `dashscope/qwen2.5-14b-instruct`
- `dashscope/deepseek-r1-distill-qwen-14b`

对应运行目录：

- [qwen2.5-14b run](/d:/user/research/runs/multi_agent/20260418T095701Z-vanilla-mad-clean-smoke-smoke20-dashscope-qwen2.5-14b-instruct)
- [deepseek-r1-distill-qwen-14b run](/d:/user/research/runs/multi_agent/20260418T100039Z-vanilla-mad-clean-smoke-smoke20-dashscope-deepseek-r1-distill-qwen-14b)

## 3. 完整性与有效性

两次运行都完整通过验证：

- `passed = true`
- `missing_files = []`
- `request_failures = 0`
- `prediction_rows = 240`

验证文件：

- [qwen2.5-14b run_validation.json](/d:/user/research/runs/multi_agent/20260418T095701Z-vanilla-mad-clean-smoke-smoke20-dashscope-qwen2.5-14b-instruct/run_validation.json)
- [deepseek-r1-distill-qwen-14b run_validation.json](/d:/user/research/runs/multi_agent/20260418T100039Z-vanilla-mad-clean-smoke-smoke20-dashscope-deepseek-r1-distill-qwen-14b/run_validation.json)

因此，本次 `smoke20` 可以视为“实验链路已跑通”的有效结果，而不是半途中断或脏产物。

## 4. 运行效率

### 4.1 qwen2.5-14b-instruct

- 运行时长：`212.36s`
- 网络请求数：`551`
- 缓存命中：`229`
- 观测网络速率：`155.68 RPM`
- 估算样本级有效并发：约 `12.65`

来源：

- [progress.json](/d:/user/research/runs/multi_agent/20260418T095701Z-vanilla-mad-clean-smoke-smoke20-dashscope-qwen2.5-14b-instruct/progress.json)

### 4.2 deepseek-r1-distill-qwen-14b

- 运行时长：`1164.03s`
- 网络请求数：`780`
- 缓存命中：`0`
- 观测网络速率：`40.21 RPM`
- 估算样本级有效并发：约 `7.37`

来源：

- [progress.json](/d:/user/research/runs/multi_agent/20260418T100039Z-vanilla-mad-clean-smoke-smoke20-dashscope-deepseek-r1-distill-qwen-14b/progress.json)

### 4.3 速度结论

- 样本级并发已经生效；`qwen2.5-14b-instruct` 这轮不再是此前那种明显串行的速度。
- 两个模型在同样配置下速度差异较大，说明真实吞吐不仅受本地并发和限流影响，还强依赖模型侧生成长度与服务端吞吐。
- `deepseek-r1-distill-qwen-14b` 的响应明显更长、更慢，因此后续若预算敏感，不适合作为大规模主跑 backbone。

## 5. Token 开销

按 `agent_turns.jsonl` 中真实网络调用统计：

- `dashscope/qwen2.5-14b-instruct`
  - 新发网络 token：约 `485,237`
  - 缓存 token：约 `66,172`
- `dashscope/deepseek-r1-distill-qwen-14b`
  - 新发网络 token：约 `831,020`
  - 缓存 token：`0`

这意味着：

- 单模型 `100万 token` 可以支持一次 `smoke20`，但余量并不充裕。
- 若升级到 `pilot100`，按本次规模粗估约为 `5` 倍，绝大多数模型的 `100万 token` 都不够。

## 6. 核心结果

### 6.1 qwen2.5-14b-instruct

| Dataset | Method | Accuracy | Avg Tokens / Q |
| --- | --- | ---: | ---: |
| GSM8K | `cot_1` | 0.95 | 262.60 |
| GSM8K | `sc_4` | 0.95 | 1018.40 |
| GSM8K | `mv_4` | 1.00 | 1011.50 |
| GSM8K | `mad_2a_r1` | 1.00 | 1424.90 |
| StrategyQA | `cot_1` | 0.60 | 173.25 |
| StrategyQA | `sc_4` | 0.55 | 696.40 |
| StrategyQA | `mv_4` | 0.60 | 700.00 |
| StrategyQA | `mad_2a_r1` | 0.60 | 1003.45 |
| HotpotQA | `cot_1` | 0.75 | 1613.00 |
| HotpotQA | `sc_4` | 0.80 | 6422.05 |
| HotpotQA | `mv_4` | 0.85 | 6424.70 |
| HotpotQA | `mad_2a_r1` | 0.60 | 6820.20 |

MAD 相对等预算对照：

- GSM8K：
  - 相对 `mv_4`：`+0.00`
  - 相对 `sc_4`：`+0.05`
- StrategyQA：
  - 相对 `mv_4`：`+0.00`
  - 相对 `sc_4`：`+0.05`
- HotpotQA：
  - 相对 `mv_4`：`-0.25`
  - 相对 `sc_4`：`-0.20`

### 6.2 deepseek-r1-distill-qwen-14b

| Dataset | Method | Accuracy | Avg Tokens / Q |
| --- | --- | ---: | ---: |
| GSM8K | `cot_1` | 0.95 | 598.15 |
| GSM8K | `sc_4` | 0.95 | 2275.70 |
| GSM8K | `mv_4` | 0.95 | 2449.00 |
| GSM8K | `mad_2a_r1` | 0.95 | 2855.60 |
| StrategyQA | `cot_1` | 0.75 | 478.25 |
| StrategyQA | `sc_4` | 0.65 | 2181.85 |
| StrategyQA | `mv_4` | 0.75 | 2301.90 |
| StrategyQA | `mad_2a_r1` | 0.65 | 2630.10 |
| HotpotQA | `cot_1` | 0.60 | 1950.50 |
| HotpotQA | `sc_4` | 0.55 | 7919.10 |
| HotpotQA | `mv_4` | 0.65 | 7890.70 |
| HotpotQA | `mad_2a_r1` | 0.60 | 8020.15 |

MAD 相对等预算对照：

- GSM8K：
  - 相对 `mv_4`：`+0.00`
  - 相对 `sc_4`：`+0.00`
- StrategyQA：
  - 相对 `mv_4`：`-0.10`
  - 相对 `sc_4`：`+0.00`
- HotpotQA：
  - 相对 `mv_4`：`-0.05`
  - 相对 `sc_4`：`+0.05`

## 7. Debate 诊断

### 7.1 qwen2.5-14b-instruct

- GSM8K：
  - 初始分歧率：`0.00`
  - debate 后一致率：`1.00`
  - vote flip rate：`0.00`
  - wrong consensus rate：`0.00`
- StrategyQA：
  - 初始分歧率：`0.05`
  - debate 后一致率：`0.95`
  - vote flip rate：`0.10`
  - wrong consensus rate：`0.35`
- HotpotQA：
  - 初始分歧率：`0.30`
  - debate 后一致率：`0.70`
  - vote flip rate：`0.10`
  - wrong consensus rate：`0.15`

### 7.2 deepseek-r1-distill-qwen-14b

- GSM8K：
  - 初始分歧率：`0.10`
  - debate 后一致率：`0.95`
  - vote flip rate：`0.00`
  - wrong consensus rate：`0.00`
- StrategyQA：
  - 初始分歧率：`0.20`
  - debate 后一致率：`0.80`
  - vote flip rate：`0.10`
  - wrong consensus rate：`0.25`
- HotpotQA：
  - 初始分歧率：`0.35`
  - debate 后一致率：`0.90`
  - vote flip rate：`0.25`
  - wrong consensus rate：`0.30`

诊断解读：

- GSM8K 上两模型本来分歧就少，debate 的额外收益非常有限，更像“高成本确认”。
- StrategyQA 和 HotpotQA 上，debate 确实能提高一致率，但“一致”并不等于“更正确”。
- 特别是 HotpotQA，错误共识比例不低，说明 peer exchange 有时会把系统推向更稳的错误答案，而不是更好的答案。

## 8. 本次 smoke 的主要结论

### 8.1 跑通层面

- `Vanilla MAD clean smoke` 已经跑通。
- 配置链、运行链、产物链、验证链都正常。
- 新增的样本级并发已经显著改善了运行效率。

### 8.2 方法层面

在这次最小 smoke 上，`mad_2a_r1` 没有展示出稳定优于等预算 `mv_4` 的证据。

更具体地说：

- 在 `qwen2.5-14b-instruct` 上：
  - GSM8K 与 `mv_4` 持平
  - StrategyQA 与 `mv_4` 持平
  - HotpotQA 明显落后于 `mv_4`
- 在 `deepseek-r1-distill-qwen-14b` 上：
  - GSM8K 与 `mv_4` 持平
  - StrategyQA 落后于 `mv_4`
  - HotpotQA 也落后于 `mv_4`

因此，本次 smoke 支持的更谨慎结论是：

- `Vanilla MAD` 在最小 2-agent / 1-round 设置下，至少在这两个 14B backbone 上，并没有显著优于等预算直接投票。
- 如果只是为了做“传统 debate baseline”，当前设置是合格的。
- 如果目标是“在相同预算下提升性能”，这次 smoke 没有给出积极证据。

## 9. 局限与注意事项

- 这只是 `smoke20`，每个数据集只有 `20` 题，不能做显著性结论。
- 本次只跑了 `mad_2a_r1`，还没有覆盖 `mad_3a_r1` 或 `mad_3a_r2`。
- 只比较了两个 DashScope 14B 同梯度 backbone，结论不能直接外推到其他模型族。
- `deepseek-r1-distill-qwen-14b` 需要关闭 `response_format` 才能稳定返回可解析内容，这一兼容性已经体现在模型目录覆盖里。
- `qwen3-*` 暂未纳入本次正式跑，因为兼容模式下还涉及额外协议参数。

## 10. 建议的下一步

- 保留当前 `mad_2a_r1 + cot_1 + sc_4 + mv_4` 作为最小传统 debate 基线。
- 后续如果继续验证 debate 是否值得，优先加：
  - `mad_3a_r1`
  - `mad_3a_r2`
- 但在进入更大实验前，建议先加入实验预算估算器，避免 `100万 token` 的免费额度在单次 `pilot100` 中被直接打空。

