# Debate vs Vote 受控实验报告

## 1. 实验概览

- 实验名：`debate-vs-vote-controlled`
- Phase：`smoke20`
- Prompt Version：`v2-controlled-span-strict`
- Backbone：`dashscope/qwen-turbo-1101`
- 运行目录：`runs/multi_agent/20260418T112020Z-debate-vs-vote-controlled-smoke20-dashscope-qwen-turbo-1101`

本实验比较的是同一批初始候选答案上的两种聚合方式：

- `initial vote`：3 个 agent 的初始独立答案直接多数投票
- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，再对修订答案投票

因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。

## 2. 结果摘要

### gsm8k

- 题量：`20`
- initial vote 准确率：`0.9000`
- debate vote 准确率：`0.9500`
- 准确率变化：`+0.0500`
- debate 修正题数：`1`
- debate 改坏题数：`0`
- 保持正确题数：`18`
- 保持错误题数：`1`
- 初始一致率：`0.8000`
- debate 后一致率：`0.9000`
- debate 翻票率：`0.1000`
- debate 增量 token/题：`1568.45`
- debate 增量时延/题：`5939.68 ms`
- 每 1k debate token 的准确率增益：`+0.031879`
- 统计检验：未计算（statistics are only reported for pilot100）

### hotpotqa

- 题量：`20`
- initial vote 准确率：`0.6500`
- debate vote 准确率：`0.7000`
- 准确率变化：`+0.0500`
- debate 修正题数：`1`
- debate 改坏题数：`0`
- 保持正确题数：`13`
- 保持错误题数：`6`
- 初始一致率：`0.7000`
- debate 后一致率：`0.8000`
- debate 翻票率：`0.2000`
- debate 增量 token/题：`5434.95`
- debate 增量时延/题：`4987.90 ms`
- 每 1k debate token 的准确率增益：`+0.009200`
- 统计检验：未计算（statistics are only reported for pilot100）

## 3. 解读注意事项

- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。
- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度/表述噪音，应在结论里单独说明。
- 若 `accuracy_delta` 接近 0 且 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。

