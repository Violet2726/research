# Debate vs Vote 对照实验报告

## 1. 实验概览

- 实验名：`vanilla_mad_clean_smoke`
- Phase：`smoke20`
- Prompt Version：`multi_agent_debate_json`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/multi_agent/vanilla_mad_clean_smoke/smoke20/20260505T094004Z-vanilla_mad_clean_smoke-smoke20-xiaomimimo-mimo-v2.5`

本实验比较的是同一批初始候选答案上的两种聚合方式：

- `initial vote`：3 个 agent 的初始独立答案直接多数投票。
- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，对修订答案再次投票。

因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。

## 2. 结果摘要

### gsm8k

- 题量：`20`
- initial vote 准确率：`0.6000`
- debate vote 准确率：`0.9000`
- 准确率变化：`+0.3000`
- debate 修正题数：`6`
- debate 改坏题数：`0`
- 保持正确题数：`12`
- 保持错误题数：`2`
- 初始一致率：`0.6000`
- debate 后一致率：`0.9000`
- debate 翻票率：`0.3000`
- debate 增量 token / 题：`870.30`
- debate 增量时延 / 题：`4151.82 ms`
- 每 1k debate token 的准确率增益：`+0.344709`
- 统计检验：未计算（statistics are only reported for pilot100）

### hotpotqa

- 题量：`20`
- initial vote 准确率：`0.7000`
- debate vote 准确率：`0.8000`
- 准确率变化：`+0.1000`
- debate 修正题数：`2`
- debate 改坏题数：`0`
- 保持正确题数：`14`
- 保持错误题数：`4`
- 初始一致率：`0.8500`
- debate 后一致率：`0.9500`
- debate 翻票率：`0.1500`
- debate 增量 token / 题：`3491.15`
- debate 增量时延 / 题：`3687.97 ms`
- 每 1k debate token 的准确率增益：`+0.028644`
- 统计检验：未计算（statistics are only reported for pilot100）

### strategyqa

- 题量：`20`
- initial vote 准确率：`0.7500`
- debate vote 准确率：`0.7500`
- 准确率变化：`+0.0000`
- debate 修正题数：`0`
- debate 改坏题数：`0`
- 保持正确题数：`15`
- 保持错误题数：`5`
- 初始一致率：`0.8500`
- debate 后一致率：`0.9000`
- debate 翻票率：`0.0000`
- debate 增量 token / 题：`635.20`
- debate 增量时延 / 题：`4380.62 ms`
- 每 1k debate token 的准确率增益：`+0.000000`
- 统计检验：未计算（statistics are only reported for pilot100）

## 3. 解读注意事项

- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。
- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度表述噪音，应在结论里单独说明。
- 若 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。

