# Debate vs Vote 对照实验报告

## 1. 实验概览

- 实验名：`debate_vs_vote_controlled`
- Phase：`smoke20`
- Prompt Version：`multi_agent_controlled_json`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/multi_agent/debate_vs_vote_controlled/smoke20/20260505T084627Z-debate_vs_vote_controlled-smoke20-xiaomimimo-mimo-v2.5`

本实验比较的是同一批初始候选答案上的两种聚合方式：

- `initial vote`：3 个 agent 的初始独立答案直接多数投票。
- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，对修订答案再次投票。

因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。

## 2. 结果摘要

### gsm8k

- 题量：`20`
- initial vote 准确率：`0.7000`
- debate vote 准确率：`0.9500`
- 准确率变化：`+0.2500`
- debate 修正题数：`5`
- debate 改坏题数：`0`
- 保持正确题数：`14`
- 保持错误题数：`1`
- 初始一致率：`0.5500`
- debate 后一致率：`0.7500`
- debate 翻票率：`0.2500`
- debate 增量 token / 题：`1596.25`
- debate 增量时延 / 题：`6157.01 ms`
- 每 1k debate token 的准确率增益：`+0.156617`
- 统计检验：未计算（statistics are only reported for pilot100）

### hotpotqa

- 题量：`20`
- initial vote 准确率：`0.7500`
- debate vote 准确率：`0.8000`
- 准确率变化：`+0.0500`
- debate 修正题数：`2`
- debate 改坏题数：`1`
- 保持正确题数：`14`
- 保持错误题数：`3`
- 初始一致率：`0.7000`
- debate 后一致率：`0.9000`
- debate 翻票率：`0.1500`
- debate 增量 token / 题：`5611.40`
- debate 增量时延 / 题：`6414.56 ms`
- 每 1k debate token 的准确率增益：`+0.008910`
- 统计检验：未计算（statistics are only reported for pilot100）

## 3. 解读注意事项

- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。
- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度表述噪音，应在结论里单独说明。
- 若 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。

