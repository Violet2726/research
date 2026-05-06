# Debate vs Vote 对照实验报告

## 1. 实验概览

- 实验名：`vanilla_mad_clean_smoke`
- Phase：`pilot100`
- Prompt Version：`multi_agent_debate_json`
- Backbone：`xiaomimimo/mimo-v2.5`
- 运行目录：`runs/multi_agent/vanilla_mad_clean_smoke/pilot100/20260506T140702Z-vanilla_mad_clean_smoke-pilot100-xiaomimimo-mimo-v2.5`

本实验比较的是同一批初始候选答案上的两种聚合方式：

- `initial vote`：3 个 agent 的初始独立答案直接多数投票。
- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，对修订答案再次投票。

因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。

## 2. 结果摘要

### gsm8k

- 题量：`100`
- initial vote 准确率：`0.5300`
- debate vote 准确率：`0.8500`
- 准确率变化：`+0.3200`
- debate 修正题数：`32`
- debate 改坏题数：`0`
- 保持正确题数：`53`
- 保持错误题数：`15`
- 初始一致率：`0.5700`
- debate 后一致率：`0.8800`
- debate 翻票率：`0.3300`
- debate 增量 token / 题：`898.66`
- debate 增量时延 / 题：`4503.02 ms`
- 每 1k debate token 的准确率增益：`+0.356086`
- McNemar exact p：`0.0`
- Bootstrap 95% CI：`[0.23, 0.42]`

### hotpotqa

- 题量：`100`
- initial vote 准确率：`0.6700`
- debate vote 准确率：`0.7100`
- 准确率变化：`+0.0400`
- debate 修正题数：`5`
- debate 改坏题数：`1`
- 保持正确题数：`66`
- 保持错误题数：`28`
- 初始一致率：`0.8100`
- debate 后一致率：`0.9200`
- debate 翻票率：`0.1000`
- debate 增量 token / 题：`3473.46`
- debate 增量时延 / 题：`3972.04 ms`
- 每 1k debate token 的准确率增益：`+0.011516`
- McNemar exact p：`0.21875`
- Bootstrap 95% CI：`[0.0, 0.09]`

### strategyqa

- 题量：`100`
- initial vote 准确率：`0.6900`
- debate vote 准确率：`0.7400`
- 准确率变化：`+0.0500`
- debate 修正题数：`7`
- debate 改坏题数：`2`
- 保持正确题数：`67`
- 保持错误题数：`24`
- 初始一致率：`0.8600`
- debate 后一致率：`0.9500`
- debate 翻票率：`0.0900`
- debate 增量 token / 题：`641.28`
- debate 增量时延 / 题：`4532.49 ms`
- 每 1k debate token 的准确率增益：`+0.077969`
- McNemar exact p：`0.179688`
- Bootstrap 95% CI：`[-0.01, 0.11]`

## 3. 解读注意事项

- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。
- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度表述噪音，应在结论里单独说明。
- 若 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。

