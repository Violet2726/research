# Debate vs Vote 对照实验报告

## 1. 实验概览

- 实验名：`debate_vs_vote_controlled`
- Phase：`pilot100`
- Prompt Version：`multi_agent_controlled_json`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/multi_agent/debate_vs_vote_controlled/pilot100/20260506T140639Z-debate_vs_vote_controlled-pilot100-xiaomimimo-mimo-v2.5`

本实验比较的是同一批初始候选答案上的两种聚合方式：

- `initial vote`：3 个 agent 的初始独立答案直接多数投票。
- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，对修订答案再次投票。

因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。

## 2. 结果摘要

### gsm8k

- 题量：`100`
- initial vote 准确率：`0.6300`
- debate vote 准确率：`0.9100`
- 准确率变化：`+0.2800`
- debate 修正题数：`28`
- debate 改坏题数：`0`
- 保持正确题数：`63`
- 保持错误题数：`9`
- 初始一致率：`0.4400`
- debate 后一致率：`0.8600`
- debate 翻票率：`0.2900`
- debate 增量 token / 题：`1663.44`
- debate 增量时延 / 题：`8926.13 ms`
- 每 1k debate token 的准确率增益：`+0.168326`
- McNemar exact p：`0.0`
- Bootstrap 95% CI：`[0.19, 0.37]`

### hotpotqa

- 题量：`100`
- initial vote 准确率：`0.6700`
- debate vote 准确率：`0.7600`
- 准确率变化：`+0.0900`
- debate 修正题数：`9`
- debate 改坏题数：`0`
- 保持正确题数：`67`
- 保持错误题数：`24`
- 初始一致率：`0.6800`
- debate 后一致率：`0.8500`
- debate 翻票率：`0.1200`
- debate 增量 token / 题：`5579.74`
- debate 增量时延 / 题：`6415.11 ms`
- 每 1k debate token 的准确率增益：`+0.016130`
- McNemar exact p：`0.003906`
- Bootstrap 95% CI：`[0.04, 0.15]`

## 3. 解读注意事项

- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。
- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度表述噪音，应在结论里单独说明。
- 若 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。

