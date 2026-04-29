# DALA-lite Smoke20 结果汇总

## 1. 文档目的

本文档汇总 `budget_comm` 实验线下 DALA-lite 在 `same-context` 与 `split-context` 两条 smoke20 轨道上的最终有效结果。这里仅纳入**最终通过校验**的运行，不纳入中间调试 run。

## 2. 最终有效运行

### 2.1 same-context

- 运行目录：
  `local/runs/budget_comm/dala_lite_same_context_v1/smoke20/20260424T040321Z-dala_lite_same_context_v1-smoke20-dashscope-qwen-turbo-1101`
- 校验结果：
  - `passed = true`
  - `request_failures_total = 0`
  - `schema_failures_total = 0`

### 2.2 split-context

- 运行目录：
  `local/runs/budget_comm/dala_lite_split_context_v1/smoke20/20260424T043649Z-dala_lite_split_context_v1-smoke20-dashscope-qwen-turbo-1101`
- 校验结果：
  - `passed = true`
  - `request_failures_total = 0`
  - `schema_failures_total = 0`

## 3. 实验设置摘要

- Backbone：`dashscope/qwen-turbo-1101`
- 方法集合：
  - `mv_3`
  - `all_to_all_full`
  - `budget_random`
  - `budget_confidence`
  - `dala_lite`
- same-context 数据集：
  - `gsm8k`
  - `strategyqa`
  - `hotpotqa`
- split-context 数据集：
  - `strategyqa`
  - `hotpotqa`
- 预算冻结规则：
  - 先做 5 题 `all_to_all_full` 校准
  - `round_budget_tokens = floor(0.4 * p50(comm_tokens))`

## 4. 主结果

### 4.1 same-context overall

| Method | Accuracy | Comm Tokens | Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.5333 | 0.00 | 2443.62 | 0.218256 |
| `all_to_all_full` | 0.5667 | 149.08 | 5227.85 | 0.108394 |
| `budget_random` | 0.5833 | 43.25 | 4939.47 | 0.118096 |
| `budget_confidence` | 0.5667 | 41.00 | 4937.85 | 0.114760 |
| `dala_lite` | 0.5667 | 34.10 | 4919.50 | 0.115188 |

### 4.2 split-context overall

| Method | Accuracy | Comm Tokens | Total Tokens | Acc / 1K Tokens |
| --- | ---: | ---: | ---: | ---: |
| `mv_3` | 0.4250 | 0.00 | 1793.45 | 0.236973 |
| `all_to_all_full` | 0.5000 | 134.50 | 3882.00 | 0.128800 |
| `budget_random` | 0.4250 | 35.62 | 3623.93 | 0.117276 |
| `budget_confidence` | 0.4500 | 34.23 | 3623.65 | 0.124184 |
| `dala_lite` | 0.4500 | 25.98 | 3605.88 | 0.124796 |

## 5. 关键信息提炼

### 5.1 same-context

- `dala_lite` 在 overall accuracy 上追平 `all_to_all_full`：
  - `0.5667 vs 0.5667`
- `dala_lite` 的通信 token 只有 `all_to_all_full` 的约 `22.9%`：
  - `34.10 / 149.08 = 0.2287`
- `dala_lite` 比 `budget_confidence` 更省通信，且 `acc_per_1k_tokens` 略高：
  - `34.10 < 41.00`
  - `0.115188 > 0.114760`
- 但 `dala_lite` 没有超过 `budget_random` 的 `acc_per_1k_tokens`：
  - `0.115188 < 0.118096`

### 5.2 split-context

- `dala_lite` 在 overall 上优于 `budget_random`，并追平 `budget_confidence` 的 accuracy：
  - `0.4500 > 0.4250`
  - `0.4500 = 0.4500`
- `dala_lite` 的 `acc_per_1k_tokens` 是预算方法里最高：
  - `0.124796 > 0.124184 > 0.117276`
- `dala_lite` 的通信 token 只有 `all_to_all_full` 的约 `19.3%`：
  - `25.98 / 134.50 = 0.1931`
- 但 `dala_lite` 相对 `all_to_all_full` 仍有 `5pp` accuracy gap：
  - `0.4500 vs 0.5000`

## 6. 分数据集观察

### 6.1 same-context

- `gsm8k`：
  - `dala_lite = 0.55`
  - `budget_random = 0.60`
  - 当前 DALA-lite 在数学题上没有超过随机预算基线
- `strategyqa`：
  - `dala_lite = 0.60`
  - 与其余通信方法持平，但通信成本最低
- `hotpotqa`：
  - `dala_lite = 0.55`
  - 与 `mv_3`、`budget_random`、`budget_confidence`、`all_to_all_full` 全部持平，但通信成本最低

### 6.2 split-context

- `strategyqa`：
  - `dala_lite = 0.70`
  - 与 `budget_confidence` 持平，并高于 `all_to_all_full = 0.65`
  - 说明在结构化事实分片场景中，预算感知选择性通信是有效的
- `hotpotqa`：
  - `dala_lite = 0.20`
  - 明显低于 `all_to_all_full = 0.35`
  - 是当前 split-context 主瓶颈

## 7. Full DALA 进入门槛检查

### 7.1 same-context

- 未通过
- 原因：
  - `dala_lite` 没有超过 `budget_random` 的 `acc_per_1k_tokens`
- 已满足条件：
  - `dala_lite` 相比 `all_to_all_full` 的 accuracy gap 不超过 3pp
  - `dala_lite` 相比 `all_to_all_full` 的 communication tokens 不高于 60%
  - `dala_lite` 优于 `budget_confidence` 的 `acc_per_1k_tokens`

### 7.2 split-context

- 未通过
- 原因：
  - `dala_lite` 相比 `all_to_all_full` 的 accuracy gap 为 `5pp`
- 已满足条件：
  - `dala_lite` 的 `acc_per_1k_tokens` 优于 `budget_random`
  - `dala_lite` 的 `acc_per_1k_tokens` 优于 `budget_confidence`
  - `dala_lite` 的 communication tokens 显著低于 `all_to_all_full`

## 8. 当前结论

- 当前 smoke20 结果支持下面这条更谨慎的结论：
  - `DALA-lite` 已经显示出**稳定的通信节省能力**
  - 但还没有足够证据支持“现在就值得投入 full DALA / MAPPO”
- 更细一点地说：
  - 在 `same-context` 下，`DALA-lite` 已经证明自己可以用更低通信成本追平 `all_to_all_full`
  - 在 `split-context` 下，`DALA-lite` 对 `StrategyQA` 有明显正向信号，但在 `HotpotQA` 上仍然明显不够强

## 9. 建议的下一步

### 9.1 优先修 split-context HotpotQA

- 优先检查 paragraph shard 的信息覆盖是否仍然过碎
- 检查 `keyword_clues / claim_span` 是否对多跳证据过弱
- 重点分析 `dala_lite` 在 HotpotQA 的失败案例，尤其是：
  - 关键证据未进入 winner set
  - winner set 太小导致遗漏第二跳证据
  - `full / keywords / silence` 分层对多跳 QA 不够友好

### 9.2 不建议立刻进入 full DALA

- 当前最合理的顺序仍然是：
  1. 先把 `split-context HotpotQA` 修强
  2. 再看 `dala_lite` 能否在 communication-necessary setting 下稳定逼近 `all_to_all_full`
  3. 只有在 smoke / pilot 阶段满足 gate 后，再考虑 full DALA / 学习版

### 9.3 可以继续补的实验

- `pilot100` 版本的 same-context 与 split-context
- `dala_lite` 在 HotpotQA 上的 message-mode 分析
- `dala_lite` vs `local_auditing` 的组合实验
- `winner_set_size` 与正确率关系图

## 10. 一句话总结

当前 DALA-lite 的 smoke20 结果更适合被写成：

> 在 same-context 下，DALA-lite 已能以显著更低的通信成本追平全通信上界；在 split-context 下，它在 StrategyQA 上表现出较好的预算效率，但 HotpotQA 仍暴露出多跳证据整合不足的问题，因此现阶段更适合继续推进 DALA-lite 机制打磨，而不是直接转入 full DALA 学习版。
