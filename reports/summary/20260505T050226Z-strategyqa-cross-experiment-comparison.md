# `strategyqa` 跨实验统一对比说明

## 1. 先给结论

`StrategyQA` 和 `GSM8K` 很不一样。

这轮 `mimo-v2.5` 的 `smoke20` 结果表明：

- 单智能体 direct reasoning 并不封顶，`cot_1 = 0.70`
- 无通信自洽已经很强，`single_agent/main_baselines` 里的 `sc_5 = 0.90`
- 很多通信方法可以把 `0.65~0.70` 的 baseline 拉到 `0.75~0.80`
- 但它们通常很难稳定超过强无通信基线 `sc_5/sc_6`

所以在 `StrategyQA` 上，真正的问题不是“通信能不能比弱 baseline 好”，而是：

> 通信能否在不显著增加成本的情况下，接近或超过强自洽基线。

## 2. 怎么读这页

和 `gsm8k` 专项报告一样，这里也需要区分两类比较：

- 同一 family 内部的比较：更能反映方法本身
- 不同 family 横向比较：更多只能做方向性解释，因为 prompt、agent role、输出协议、消息格式都不相同

本页主要回答三个问题：

1. `StrategyQA` 在各实验里大概是什么水平？
2. 哪些方法真的在这个数据集上有效？
3. 为什么通信方法常常赢不了最强的 `sc_5/sc_6`？

## 3. 代表性结果总表

| Family | Experiment | Representative Methods | Accuracy | Total Tokens | Calls/Q | 直接结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `single_agent` | `main_baselines` | `cot_1 / mv_5 / sc_5` | `0.70 / 0.75 / 0.90` | `171.40 / 845.35 / 856.55` | `1 / 5 / 5` | 强无通信自洽是最难打的对照 |
| `single_agent` | `robustness` | `cot_1` | `0.70` | `171.40` | `1` | 与主基线一致 |
| `multi_agent` | `vanilla_mad_clean_smoke` | `cot_1 / mad_2a_r1 / mv_4 / sc_4` | `0.70 / 0.80 / 0.70 / 0.75` | `206.05 / 1124.60 / 821.70 / 830.25` | `1 / 4 / 4 / 4` | debate 有帮助，但 `sc_4` 也不弱 |
| `free_mad_lite` | `free_mad_lite_v1` | `mv_3_initial / vanilla_mad / anti_conformity / trajectory_judge` | `0.70 / 0.80 / 0.55 / 0.70` | `605.70 / 1759.85 / 1929.35 / 2634.30` | `3 / 6 / 6 / 7` | 反从众在 `StrategyQA` 上明显有害 |
| `budget_comm` | `dala_lite_same_context_v1` | `mv_3 / all_to_all / budget_random / budget_confidence / dala_lite` | `0.60 / 0.60 / 0.65 / 0.65 / 0.70` | `1083.90 / 2623.80 / 2245.30 / 2249.70 / 2257.10` | `3 / 6 / 6 / 6 / 6` | `dala_lite` 是该 family 内最优 |
| `budget_comm` | `dala_lite_split_context_v1` | `mv_3 / all_to_all / budget_random / budget_confidence / dala_lite` | `0.85 / 0.85 / 0.85 / 0.85 / 0.90` | `1191.70 / 2827.65 / 2474.30 / 2470.55 / 2460.80` | `3 / 6 / 6 / 6 / 6` | 在这条线里 `dala_lite` 反而最强 |
| `sid_lite` | `sid_lite_v1` | `mv_3 / always_full / compression_only / sid_lite` | `0.70 / 0.75 / 0.70 / 0.70` | `950.70 / 2953.65 / 2307.15 / 1378.30` | `3 / 6 / 6 / 3.9` | 早退省成本，但没有涨点 |
| `selective_comm` | `trigger_early_exit_v1` | `mv_3 / always / disagreement / confidence / sc_6` | `0.70 / 0.75 / 0.75 / 0.70 / 0.80` | `807.00 / 2206.50 / 1093.25 / 807.00 / 1644.15` | `3 / 6 / 3.6 / 3 / 6` | 轻触发能接近 always，但 `sc_6` 更强 |
| `selective_comm` | `trigger_voc_v2` | `mv_3 / always / disagreement / voc_v2 / sc_6` | `0.65 / 0.80 / 0.80 / 0.80 / 0.85` | `961.40 / 2257.15 / 1553.50 / 1872.15 / 1925.40` | `3 / 6 / 4.35 / 5.1 / 6` | selective 通信能明显提分，但还没赢 `sc_6` |
| `cue` | `cue_v1` | `mv_3 / always / disagreement / cue_v1` | `0.70 / 0.75 / 0.75 / 0.75` | `1028.50 / 3009.50 / 1509.05 / 1167.20` | `3 / 6.25 / 3.8 / 3.2` | `cue_v1` 在这题上很像效率最优解 |
| `sparc` | `sparc_v1_smoke` | `mv_3 / hybrid_trigger_baseline / sparc_v1 / always` | `0.65 / 0.65 / 0.60 / 0.60` | `915.70 / 1541.05 / 1768.85 / 2354.00` | `3 / 4.5 / 4.95 / 6.45` | 这轮 `StrategyQA` 上 `SPARC` 没有形成优势 |
| `sparc` | `content_ablation_v1` | `mv_3 / full_cot / disagreement_step_only / critical_evidence_only` | `0.65 / 0.70 / 0.65 / 0.65` | `915.70 / 2656.70 / 2126.20 / 2221.65` | `3 / 6 / 6 / 6` | full content 只带来有限收益 |
| `sparc` | `auditing_ablation_v1` | `majority_vote / final_round_vote / local_auditing / single_judge` | `0.65 / 0.65 / 0.60 / 0.75` | `915.70 / 2126.20 / 2354.00 / 2624.25` | `3 / 6 / 6.45 / 7` | 强 judge 能救，但局部审计这轮反而不稳 |

## 4. 对 `StrategyQA` 最重要的观察

### 4.1 强无通信基线非常硬

`single_agent/main_baselines` 给出的最重要信号是：

- `cot_1 = 0.70`
- `mv_5 = 0.75`
- `sc_5 = 0.90`

这意味着：

- 仅靠多次独立采样和自洽聚合，模型已经能拿到很强结果
- 很多通信方法即使“有用”，也只是从 `0.65~0.70` 拉到 `0.75~0.80`
- 它们必须面对一个事实：`sc_5/sc_6` 本来就是更难打的对照

### 4.2 通信是有帮助的，但通常不是压倒性的

比较典型的正面例子：

- `multi_agent/vanilla_mad_clean_smoke` 中 `mad_2a_r1 = 0.80`，高于 `mv_4 = 0.70`
- `selective_comm/trigger_voc_v2` 中 `disagreement / voc_v2 / always = 0.80`，高于 `mv_3 = 0.65`
- `budget_comm/dala_lite_same_context_v1` 中 `dala_lite = 0.70`，高于 `mv_3 = 0.60`

这说明：

- `StrategyQA` 不是“完全不值得通信”
- 但通信收益通常是中等幅度的，而不是像 split-context 那样具有必要性

### 4.3 反从众在 `StrategyQA` 上风险很大

`free_mad_lite_v1` 在这个数据集上很值得警惕：

- `vanilla_mad_r1_final_vote = 0.80`
- `anti_conformity_final_vote = 0.55`

这几乎是整轮实验里最清晰的负例之一，说明：

- `StrategyQA` 这类很多题的答案空间本来就不大
- 人为诱导“反着想”很容易把正确共识打散

### 4.4 更轻、更保守的选择性通信反而更合适

这一点 `CUE` 表现得最明显：

- `always_communicate = 0.75`
- `disagreement_triggered = 0.75`
- `cue_v1 = 0.75`

但成本差很多：

- `always total_tokens_mean = 3009.50`
- `disagreement total_tokens_mean = 1509.05`
- `cue_v1 total_tokens_mean = 1167.20`

也就是说：

- `CUE` 在 `StrategyQA` 上没有掉点
- 同时把通信量压得很低

这很符合该任务的结构：值得通信的样本有，但并不多。

## 5. 哪些比较最值得信

### 5.1 同 family 内部最可信

#### `single_agent`

- `cot_1 -> mv_5 -> sc_5` 从 `0.70 -> 0.75 -> 0.90`

这是真正可靠的结论：

- 对 `StrategyQA`，多样独立采样是明显有效的

#### `multi_agent/vanilla_mad_clean_smoke`

- `mad_2a_r1 = 0.80`
- `mv_4 = 0.70`
- `sc_4 = 0.75`

这里可以比较放心地说：

- debate 在这个协议下确实有帮助
- 但没有形成对 `sc_4` 的压倒性优势

#### `free_mad_lite`

- `anti_conformity` 从 `0.80` 打到 `0.55`

这里几乎可以直接定性：

- 当前反从众策略不适合作为 `StrategyQA` 默认机制

#### `budget_comm`

same-context：

- `dala_lite = 0.70`
- `budget_random / budget_confidence = 0.65`
- `all_to_all_full = 0.60`

split-context：

- `dala_lite = 0.90`
- 其他预算方法大多是 `0.85`

这里说明：

- `dala_lite` 在 `StrategyQA` 上是稳健有效的
- 而且并不是靠“多发消息”取胜

#### `selective_comm/trigger_voc_v2`

- `mv_3 = 0.65`
- `always / disagreement / voc_v2 = 0.80`
- `sc_6 = 0.85`

这说明：

- selective communication 的确能让性能上台阶
- 但仍没有跨过强 self-consistency 基线

#### `cue/cue_v1`

- `mv_3 = 0.70`
- `cue_v1 = 0.75`
- `always / disagreement = 0.75`

这是 `CUE` 在 `StrategyQA` 上最漂亮的局部结果之一：

- 它不像 `VoC` 那样必须靠更高 trigger 才到位
- 也不像 `always` 那样成本极重

### 5.2 跨 family 横比只能做方向性解释

例如：

- `single_agent/main_baselines sc_5 = 0.90`
- `selective_comm/trigger_voc_v2 sc_6 = 0.85`
- `trigger_early_exit_v1 sc_6 = 0.80`

这些不能简单推导成：

- “family A 比 family B 差 5 个点”

更合理的解释是：

- 虽然名字都叫 `sc`
- 但 prompt contract、结构化输出、共享前缀、阶段化控制都不同

## 6. 为什么很多通信方法赢不了 `sc_5/sc_6`

这背后有三个结构性原因。

### 6.1 `StrategyQA` 的很多收益来自“多样答案”，不一定来自“相互说服”

对于是非判断题，多个独立答案的分布本身就很有信息量。  
这会让 `self-consistency` 天生占便宜。

### 6.2 交流带来的信息增量有限

和 split-context 任务不同，`StrategyQA` 大多数样本并不是“我缺了一块证据，必须靠别人告诉我”。  
很多时候所有 agent 都看的是同一个问题，只是推理偏好不同。

这时通信不一定提供新信息，只是在重新组织已有信息。

### 6.3 通信可能引入不必要噪声

一旦交流格式复杂、阶段变多、输出字段增多，就会出现：

- 多余解释干扰原本正确的直觉
- 错误但自信的理由被采纳
- 本来简单的是非题被过度推理

这也是为什么：

- `anti_conformity` 会明显伤害 `StrategyQA`
- 轻量 trigger/early-exit 往往比重 debate 更合适

## 7. 这个数据集对论文意味着什么

`StrategyQA` 给你的论文主线提出的是一个很重要的约束：

> 不是所有任务都值得重通信，很多任务更需要的是“少量、精准、低打扰”的通信。

从这个角度看，最有价值的实验不是：

- “always communicate 能不能再高 1 个点”

而是：

- `CUE` 能否用更少通信达到和 `always/disagreement` 一样的准确率
- `DALA-lite` 能否证明预算控制不是随便省 token，而是能守住效果

## 8. 最值得记住的 5 条结论

1. `StrategyQA` 上最强的单项结果之一仍然是强无通信自洽，`single_agent/main_baselines sc_5 = 0.90`。
2. 通信不是没用，但多数时候只能把弱 baseline 拉到 `0.75~0.80` 左右。
3. `anti_conformity` 在该任务上是明显负例，不适合作为默认设计。
4. `CUE` 在 `StrategyQA` 上表现很像“最省且不掉点”的效率最优解。
5. 这个数据集最支持的论文观点不是“多交流更好”，而是“交流应当轻量、节制、选择性发生”。

## 9. 相关结果路径

- 单智能体主基线：`runs/single_agent/main_baselines/smoke20/20260505T050226Z-main_baselines-smoke20-xiaomimimo-mimo-v2.5`
- Vanilla MAD clean smoke：`runs/multi_agent/vanilla_mad_clean_smoke/smoke20/20260505T050948Z-vanilla_mad_clean_smoke-smoke20-xiaomimimo-mimo-v2.5`
- Free-MAD-lite：`runs/free_mad_lite/free_mad_lite_v1/smoke20/20260505T050228Z-free_mad_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- DALA-lite same-context：`runs/budget_comm/dala_lite_same_context_v1/smoke20/20260505T050230Z-dala_lite_same_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- DALA-lite split-context：`runs/budget_comm/dala_lite_split_context_v1/smoke20/20260505T054055Z-dala_lite_split_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- SID-lite：`runs/sid_lite/sid_lite_v1/smoke20/20260505T050232Z-sid_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- Trigger early-exit：`runs/selective_comm/trigger_early_exit_v1/smoke20/20260505T050233Z-trigger_early_exit_v1-smoke20-xiaomimimo-mimo-v2.5`
- VoC v2：`runs/selective_comm/trigger_voc_v2/smoke20/20260505T053920Z-trigger_voc_v2-smoke20-xiaomimimo-mimo-v2.5`
- CUE：`runs/cue/cue_v1/smoke20/20260505T053917Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- SPARC：`runs/sparc/sparc_v1_smoke/smoke20/20260505T050235Z-sparc_v1_smoke-smoke20-xiaomimimo-mimo-v2.5`
