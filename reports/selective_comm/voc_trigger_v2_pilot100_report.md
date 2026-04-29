# MiMo `voc_trigger_v2` Pilot100 实验报告

## 1. 实验目标

本次实验聚焦评估 `voc_trigger_v2` 在 `xiaomimimo/mimo-v2.5` 上的实际效果，核心问题是：

- 相比不通信基线，`voc_trigger_v2` 是否能稳定提升准确率。
- 相比同预算基线，`voc_trigger_v2` 是否仍然具备成本收益优势。
- 相比总是通信的上界方法，`voc_trigger_v2` 是否能在较小精度损失下显著节省 token。

本轮报告基于以下正式确认 run：

- 运行目录：
  [20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5](/d:/user/research/local/runs/selective_comm/trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy/pilot100/20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5)
- 主配置：
  [trigger_voc_v2_mimo_v2.5_equal_budget_gsm_strategy.toml](/d:/user/research/configs/selective_comm/experiments/trigger_voc_v2_mimo_v2.5_equal_budget_gsm_strategy.toml)

## 2. 实验设置

- Backbone：`xiaomimimo/mimo-v2.5`
- Prompt version：`selective_comm_voc_json_v2`
- Protocol：`3 agents + 1 debate round`
- Phase：`pilot100`
- Dataset：
  `gsm8k`
  `strategyqa`
  `hotpotqa`
- Method：
  `always_communicate`
  `disagreement_triggered`
  `voc_trigger_v2`
  `mv_3`
  `mv_6`
  `sc_6`

本轮总计划规模：

- `5400` calls
- `1800` 题级预测

## 3. 运行质量

运行验证结果见：
[run_validation.json](/d:/user/research/local/runs/selective_comm/trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy/pilot100/20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5/run_validation.json)

关键结论：

- `passed = true`
- `request_failures_total = 0`
- `output_success_rate = 1.0`
- `invalid_confidence_ratio = 0.0`
- `shared_hash_check = true`
- `early_exit_zero_comm_check = true`

这说明本轮结果可用于正式分析，不存在工程性失败对结论的系统性污染。

## 4. Overall 结果

| Method | Accuracy | Total Tokens | Comm Tokens | Trigger Rate | Early Exit Rate |
|---|---:|---:|---:|---:|---:|
| `always_communicate` | 0.750 | 3929.85 | 1461.20 | 1.000 | 0.000 |
| `voc_trigger_v2` | 0.733 | 3219.68 | 751.03 | 0.500 | 0.500 |
| `disagreement_triggered` | 0.723 | 2977.02 | 508.37 | 0.323 | 0.677 |
| `mv_3` | 0.630 | 2468.65 | 0.00 | 0.000 | 0.000 |
| `mv_6` | 0.640 | 4941.03 | 0.00 | 0.000 | 0.000 |
| `sc_6` | 0.647 | 4938.75 | 0.00 | 0.000 | 0.000 |

围绕 `voc_trigger_v2` 的核心结论：

- 对比 `mv_3`：
  准确率提升 `+10.33` 个百分点，但 token 增加约 `30.4%`。
- 对比 `mv_6`：
  准确率提升 `+9.33` 个百分点，同时 token 降低约 `34.8%`。
- 对比 `sc_6`：
  准确率提升 `+8.67` 个百分点，同时 token 降低约 `34.8%`。
- 对比 `always_communicate`：
  准确率下降 `1.67` 个百分点，但 token 降低约 `18.1%`。
- 对比 `disagreement_triggered`：
  准确率提升 `1.0` 个百分点，但 token 增加约 `8.15%`。

解释：

- `voc_trigger_v2` 显著优于不通信基线，尤其优于同预算的 `mv_6 / sc_6`。
- 它没有达到 `always_communicate` 的最高准确率，但用明显更低的通信开销换取了较小精度差。
- 相比更保守的 `disagreement_triggered`，`voc_trigger_v2` 更积极触发通信，带来小幅精度增益，也带来一定成本上升。

## 5. 数据集拆分结果

### 5.1 GSM8K

| Method | Accuracy | Total Tokens | Comm Tokens |
|---|---:|---:|---:|
| `always_communicate` | 0.780 | 2904.58 | 1678.08 |
| `voc_trigger_v2` | 0.760 | 2227.93 | 1001.43 |
| `disagreement_triggered` | 0.730 | 2156.67 | 930.17 |
| `mv_3` | 0.480 | 1226.50 | 0.00 |
| `mv_6` | 0.490 | 2456.54 | 0.00 |
| `sc_6` | 0.470 | 2452.09 | 0.00 |

结论：

- `voc_trigger_v2` 对 `mv_6 / sc_6` 的优势非常明确：
  准确率高约 `27` 到 `29` 个百分点，token 还更低。
- 相比 `disagreement_triggered`，`voc_trigger_v2` 额外带来 `+3` 个百分点准确率，代价是约 `3.3%` token 增长。
- 相比 `always_communicate`，只损失 `2` 个百分点准确率，却节省约 `23.3%` token。

GSM8K 是 `voc_trigger_v2` 的明确受益场景。

### 5.2 StrategyQA

| Method | Accuracy | Total Tokens | Comm Tokens |
|---|---:|---:|---:|
| `always_communicate` | 0.770 | 2254.70 | 1294.71 |
| `sc_6` | 0.760 | 1921.99 | 0.00 |
| `disagreement_triggered` | 0.750 | 1271.95 | 311.96 |
| `voc_trigger_v2` | 0.750 | 1744.66 | 784.67 |
| `mv_6` | 0.730 | 1923.98 | 0.00 |
| `mv_3` | 0.710 | 959.99 | 0.00 |

结论：

- `voc_trigger_v2` 相比 `mv_6` 仍有 `+2` 个百分点优势，同时 token 下降约 `9.3%`。
- 但相比 `disagreement_triggered`，`voc_trigger_v2` 没有精度优势，却多花了约 `37.2%` token。
- 相比 `always_communicate`，`voc_trigger_v2` 精度下降 `2` 个百分点，但节省约 `22.6%` token。

StrategyQA 上，`voc_trigger_v2` 是“能用”的，但不是最优的 trigger；更保守的 `disagreement_triggered` 性价比更强。

### 5.3 HotpotQA

| Method | Accuracy | Total Tokens | Comm Tokens |
|---|---:|---:|---:|
| `sc_6` | 0.710 | 10442.17 | 0.00 |
| `mv_3` | 0.700 | 5219.47 | 0.00 |
| `always_communicate` | 0.700 | 6630.28 | 1410.81 |
| `mv_6` | 0.700 | 10442.58 | 0.00 |
| `disagreement_triggered` | 0.690 | 5502.45 | 282.98 |
| `voc_trigger_v2` | 0.690 | 5686.45 | 466.98 |

结论：

- `voc_trigger_v2` 没有打赢任何代表性 baseline。
- 它相对 `mv_3` 准确率下降 `1` 个百分点，token 反而增加约 `8.95%`。
- 相对 `disagreement_triggered`，精度持平，但 token 更高。
- 相对 `always_communicate`，虽然节省了约 `14.2%` token，但没有精度收益。

HotpotQA 是 `voc_trigger_v2` 的弱场景，说明“基于 claim/uncertainty 的触发”并不天然适合所有多跳问答任务。

## 6. 触发行为分析

基于
[oracle_trigger_eval.json](/d:/user/research/local/runs/selective_comm/trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy/pilot100/20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5/oracle_trigger_eval.json)
和
[policy_diagnostics.json](/d:/user/research/local/runs/selective_comm/trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy/pilot100/20260429T053620Z-trigger_voc_v2_mimo_v2_5_equal_budget_gsm_strategy-pilot100-xiaomimimo-mimo-v2.5/policy_diagnostics.json)
，`voc_trigger_v2` 的行为特征是：

- `overall trigger_rate = 0.50`
- `overall early_exit_rate = 0.50`
- `helpful_recall = 0.868`
- `false_trigger_rate = 0.39`

与 `disagreement_triggered` 对比：

- `voc_trigger_v2` 更积极：
  `trigger_rate` 从 `0.323` 提高到 `0.500`
- `voc_trigger_v2` 的 beneficial recall 更高：
  从 `0.789` 提高到 `0.868`
- 但代价是 neutral/harmful 触发更多：
  `false_trigger_rate` 从 `0.223` 升到 `0.390`

这说明 `voc_trigger_v2` 的核心价值，不是“更省”，而是“愿意为了多救回一部分应通信样本而支付更多通信成本”。

## 7. 对 `voc_trigger_v2` 的总体判断

如果目标是“相对同预算基线证明 trigger 通信有价值”，本实验给出的结论是正面的：

- `voc_trigger_v2` 明显优于 `mv_6` 和 `sc_6`
- `overall` 上分别提升约 `9.33` 和 `8.67` 个百分点准确率
- 同时 token 还下降约 `34.8%`

如果目标是“在 trigger 家族中选择最优默认策略”，结论就更细：

- `voc_trigger_v2` 比 `disagreement_triggered` 略强一点：
  `overall accuracy +1.0` 个百分点
- 但也更贵：
  `total_tokens +8.15%`
- 且这种增益主要来自 `gsm8k`
- 在 `strategyqa` 上没有拉开差距
- 在 `hotpotqa` 上没有收益

因此，`voc_trigger_v2` 更适合被解释为：

- 一个“偏积极通信”的高召回 trigger
- 在数理推理场景中更有价值
- 但不是跨所有任务都最稳妥的默认策略

## 8. 汇报建议

汇报时建议采用以下表述：

1. `voc_trigger_v2` 已经通过完整 `pilot100` 工程验证，运行质量可靠。
2. 相比同预算 baseline，`voc_trigger_v2` 能显著提高准确率，并降低总 token 开销。
3. 相比总是通信，它用约 `18%` token 节省换来了约 `1.67` 个百分点 accuracy 损失，整体属于可接受的效率折中。
4. 相比更保守的 `disagreement_triggered`，`voc_trigger_v2` 的精度略高，但成本更高，是否采用取决于业务更看重“准确率上限”还是“成本稳健性”。
5. `voc_trigger_v2` 的收益具有任务依赖性：
   `gsm8k` 明显受益，
   `strategyqa` 收益有限，
   `hotpotqa` 不建议优先采用。

## 9. 最终建议

- 如果业务目标是“尽量稳妥、省通信”，优先 `disagreement_triggered`
- 如果业务目标是“在同预算下尽量提高整体准确率，尤其偏向数理推理”，可以考虑 `voc_trigger_v2`
- 如果业务流量中 `hotpotqa` 类多跳问答占比很高，不建议把 `voc_trigger_v2` 直接作为全局默认策略

就本轮 `pilot100` 而言，我对 `voc_trigger_v2` 的结论是：

它是一个**有效但非全局最优**的 trigger，
更像“高召回、适合数理任务”的通信策略，
而不是“所有任务统一默认”的最优答案。
