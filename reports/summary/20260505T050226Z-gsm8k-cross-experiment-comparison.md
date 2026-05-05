# `gsm8k` 跨实验统一对比说明

## 1. 先回答核心问题

不是。

在这轮 `mimo-v2.5` 的 `smoke20` 实验里，`gsm8k` 上拿到 `1.0` 的情况只出现在少数实验/方法上：

- `single_agent/main_baselines` 的 `cot_1 = 1.0`
- `single_agent/main_baselines` 的 `mv_5 = 1.0`
- `single_agent/main_baselines` 的 `sc_5 = 1.0`
- `single_agent/robustness` 的 `cot_1 = 1.0`
- `selective_comm/trigger_early_exit_v1` 的 `sc_6 = 1.0`

绝大多数其他方法并不是 `1.0`，而且跨度很大，最低可以到 `0.45`。

这说明：

- `MiMo-V2.5` 在某些 direct single-agent / self-consistency 协议下确实能把这轮 `gsm8k smoke20` 做满。
- 但“底座模型在某个协议下是 `1.0`”不等于“任何多智能体、通信、审计、选择性触发框架在同一子集上也会自动是 `1.0`”。

## 2. 这一页怎么读

本页的目标不是给出一个“所有实验混在一起的最终排名”，而是回答两个更准确的问题：

1. `gsm8k` 在各实验里到底是多少分？
2. 这些差异里，哪些更像方法本身，哪些更像 prompt/protocol 差异？

需要强调：

- 这些实验大多使用的是同一个 `smoke20_seed42` 子集。
- 但它们并不共享同一个 prompt contract、agent role、消息格式、调用预算和聚合流程。
- 所以跨 family 横比只能做方向性解释，不能直接视为“同一条件下的纯算法优劣”。

## 3. 代表性结果总表

下面只列每个实验中最有代表性的 `gsm8k` 方法，不把所有行全部铺开。

| Family | Experiment | Representative Methods | Accuracy | Total Tokens | Calls/Q | 直接结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `single_agent` | `main_baselines` | `cot_1 / mv_5 / sc_5` | `1.00 / 1.00 / 1.00` | `238.10 / 1186.45 / 1198.15` | `1 / 5 / 5` | 这是当前最强、最干净的 `gsm8k` 结果 |
| `single_agent` | `robustness` | `cot_1` | `1.00` | `238.10` | `1` | 与主单智能体结果一致 |
| `multi_agent` | `debate_vs_vote_controlled` | `mad_3a_r1` | `0.80` | `2594.80` | `6` | 单轮三 agent debate 不是 `1.0` |
| `multi_agent` | `vanilla_mad_clean_smoke` | `cot_1 / mad_2a_r1 / mv_4 / sc_4` | `0.55 / 0.80 / 0.70 / 0.80` | `315.45 / 1530.40 / 1179.00 / 1171.95` | `1 / 4 / 4 / 4` | 同 family 内 debate 有收益，但远未到 `1.0` |
| `free_mad_lite` | `free_mad_lite_v1` | `mv_3_initial / vanilla_mad / anti_conformity / trajectory_judge` | `0.50 / 0.90 / 0.95 / 0.90` | `894.05 / 2591.45 / 2758.55 / 3671.85` | `3 / 6 / 6 / 7` | 反从众在这轮 `gsm8k` 上有效，但仍没追到 `1.0` |
| `budget_comm` | `dala_lite_same_context_v1` | `mv_3 / all_to_all / budget_random / budget_confidence / dala_lite` | `0.50 / 0.70 / 0.80 / 0.80 / 0.75` | `1300.60 / 3170.10 / 2696.00 / 2697.05 / 2675.05` | `3 / 6 / 6 / 6 / 6` | 预算通信有效，但这轮 `gsm8k` 上没打满 |
| `sid_lite` | `sid_lite_v1` | `mv_3 / always_full / compression_only / sid_lite` | `0.55 / 0.80 / 0.60 / 0.60` | `1182.40 / 3501.15 / 2741.20 / 2102.80` | `3 / 6 / 6 / 4.65` | 早退压缩在 `gsm8k` 上明显掉点 |
| `selective_comm` | `trigger_early_exit_v1` | `mv_3 / always / disagreement / confidence / sc_6` | `0.95 / 0.95 / 0.95 / 0.95 / 1.00` | `996.55 / 2717.95 / 1104.25 / 996.55 / 1981.15` | `3 / 6 / 3.15 / 3 / 6` | 这组协议下 `gsm8k` 几乎全线很高 |
| `selective_comm` | `trigger_voc_v2` | `mv_3 / always / disagreement / voc_v2 / confidence` | `0.45 / 0.85 / 0.85 / 0.85 / 0.50` | `1202.45 / 2819.65 / 2227.75 / 2227.75 / 1304.10` | `3 / 6 / 4.8 / 4.8 / 3.15` | 这组协议下基线掉很多，但 trigger 线能拉回到 `0.85` |
| `cue` | `cue_v1` | `mv_3 / always / disagreement / cue_v1` | `0.60 / 0.80 / 0.80 / 0.75` | `1262.50 / 3593.55 / 2669.85 / 1804.05` | `3 / 6.3 / 4.95 / 3.75` | `cue_v1` 更省，但在 `gsm8k` 上略 under-trigger |
| `sparc` | `sparc_v1_smoke` | `mv_3 / hybrid_trigger_baseline / sparc_v1 / always` | `0.45 / 0.60 / 0.65 / 0.65` | `1136.15 / 2028.95 / 2265.35 / 2861.40` | `3 / 4.65 / 5 / 6.35` | 端到端结构比纯投票好，但离 `1.0` 很远 |
| `sparc` | `content_ablation_v1` | `mv_3 / full_cot / disagreement_step_only / critical_evidence_only` | `0.45 / 0.80 / 0.60 / 0.60` | `1136.15 / 3192.35 / 2625.00 / 2671.50` | `3 / 6 / 6 / 6` | `gsm8k` 上内容完整性非常重要 |
| `sparc` | `auditing_ablation_v1` | `majority_vote / final_round_vote / local_auditing / single_judge` | `0.45 / 0.60 / 0.65 / 0.90` | `1136.15 / 2625.00 / 2861.40 / 3263.45` | `3 / 6 / 6.35 / 7` | 强 judge 确实能救，但方法类别已明显变重 |

## 4. 为什么会出现“单智能体 1.0，但别的方法不是 1.0”

根因不是模型突然“不会做 `gsm8k` 了”，而是实验协议发生了变化。

主要有 5 个原因：

### 4.1 Prompt contract 变了

不同 family 使用的 prompt version 并不一样，例如：

- `single_agent_reasoning_json_v1`
- `multi_agent_controlled_json`
- `multi_agent_debate_json`
- `free_mad_lite_v1_json`
- `budget_comm_dala_lite_v1`
- `sid_lite_v1_json`
- `selective_comm_trigger_json`
- `selective_comm_voc_json_v2`
- `cue_v1_json`
- `sparc_v1_json`

这些 prompt 不是简单的“换个包装”，而是把模型放进了完全不同的行为约束里：

- 有的要求直接单次解题
- 有的要求先提出候选再 debate
- 有的要求输出结构化 trigger 字段
- 有的要求审计、复议、局部裁决

在 `gsm8k` 这种本来单次解题就很强的任务上，协议越复杂，越容易额外引入噪声。

### 4.2 调用预算和 agent 数不同

同样是 `gsm8k`，实际比较对象可能是：

- `cot_1`
- `mv_3`
- `mv_4`
- `mv_5`
- `mv_6`
- `sc_4`
- `sc_5`
- `sc_6`
- `mad_2a_r1`
- `mad_3a_r1`

这些本来就不是同一种问题设置。

例如：

- `single_agent/main_baselines` 的 `cot_1 = 1.0`
- `multi_agent/vanilla_mad_clean_smoke` 的 `cot_1 = 0.55`

这不应该被解释成“同一个 `cot_1` 差了 45 个点”，更合理的解释是：

- 两边虽然名字都叫 `cot_1`
- 但提示协议、输出约束和运行流程已经不是同一个实验条件

### 4.3 通信本身会引入额外错误路径

在 `gsm8k` 上，很多题本来单独做就能做对。此时多一轮通信可能带来：

- 错误中间结论被他人采纳
- 正确答案被不必要地推翻
- 结构化通信字段本身成为新的失败点

所以在这个数据集上，通信收益必须先战胜“通信噪声”。

### 4.4 结构化输出负担会吃掉一部分原始解题能力

例如 `VoC`、`CUE`、`SPARC`、`SID-lite` 等方法都要求模型同时完成：

- 解题
- 提取 claim / confidence / evidence / uncertainty
- 再做 trigger 或 message formatting

这类多头任务在复杂任务上可能值得，但在 `gsm8k` 这种 direct reasoning 极强的题上，常常会变成额外负担。

### 4.5 某些方法的 baseline 不是“最强单智能体基线”

例如：

- `VoC v2` 里的 `mv_3 = 0.45`
- `SPARC` 里的 `mv_3 = 0.45`

这不能直接拿来和 `single_agent/main_baselines` 的 `mv_5 = 1.0` 做“方法优劣”判断，因为它们不是同一提示协议下的同一控制组。

## 5. 哪些差异更像“方法本身”，哪些更像“协议差异”

### 5.1 更像方法本身的差异

这类差异通常应该只在同一个 experiment 或同一家族内解释。

#### `multi_agent/vanilla_mad_clean_smoke`

- `mad_2a_r1 = 0.80`
- `mv_4 = 0.70`
- `sc_4 = 0.80`

这里可以较放心地说：

- 在这个协议下，debate 确实比 `mv_4` 好
- 但没有明显超过 `sc_4`

#### `free_mad_lite/free_mad_lite_v1`

- `vanilla_mad_r1_final_vote = 0.90`
- `anti_conformity_final_vote = 0.95`
- `free_mad_lite_llm_trajectory = 0.90`

这里可以解释成：

- 反从众在这轮 `gsm8k` 上是有益的
- 但 trajectory judge 没有继续把准确率往上推

#### `budget_comm/dala_lite_same_context_v1`

- `budget_random = 0.80`
- `budget_confidence = 0.80`
- `dala_lite = 0.75`

这里更像真实方法信号：

- 在这轮 `gsm8k` 上，`dala_lite` 没有战胜更简单的预算启发式

#### `sid_lite/sid_lite_v1`

- `always_full = 0.80`
- `sid_lite = 0.60`

这里也很像真实方法结论：

- 对 `gsm8k` 来说，SID 风格的早退/压缩过于保守，掉点明显

#### `selective_comm/trigger_voc_v2`

- `always = 0.85`
- `disagreement = 0.85`
- `voc_v2 = 0.85`

这说明在这个协议里：

- `VoC v2` 至少没有比 `disagreement` 更强
- 它更多像是“没掉点，但也没额外涨点”

#### `cue/cue_v1`

- `always = 0.80`
- `disagreement = 0.80`
- `cue_v1 = 0.75`

这里可以合理解释为：

- `CUE` 在 `gsm8k` 上有轻度 under-trigger
- 它用更少通信换了一点点准确率损失

#### `sparc/content_ablation_v1`

- `full_cot = 0.80`
- `disagreement_step_only = 0.60`
- `critical_evidence_only = 0.60`

这很像真实内容机制信号：

- 在 `gsm8k` 上，过度压缩消息内容会直接伤害性能

#### `sparc/auditing_ablation_v1`

- `majority_vote = 0.45`
- `local_auditing = 0.65`
- `single_judge = 0.90`

这表明：

- 强裁决器在这个子集上确实有效
- 但它也明显变成了更重、更中心化的方法

### 5.2 更像协议差异的部分

下面这些对比不要直接解释成“方法更强”：

#### `single_agent/main_baselines` 的 `cot_1 = 1.0` 对比 `multi_agent/vanilla_mad_clean_smoke` 的 `cot_1 = 0.55`

这更像：

- 单智能体 direct reasoning prompt 和多智能体 controlled/debate prompt 不是一回事

#### `single_agent/main_baselines` 的 `mv_5 = 1.0` 对比 `selective_comm/trigger_voc_v2` 的 `mv_3 = 0.45`

这更像：

- `mv_5` 与 `mv_3` 本来预算不同
- 还叠加了不同 prompt version 和 shared-stage 协议

#### `single_agent/main_baselines` 的 `sc_5 = 1.0` 对比 `trigger_early_exit_v1` 的 `sc_6 = 1.0`

这两个结果都很好，但也不能简单合并说“所有 SC 都是 1.0”，因为：

- 它们对应的协议与控制环境不同

## 6. 一个更准确的结论框架

如果把问题重新表述成三个更精确的问题，答案会更清楚：

### 问题 1

`MiMo-V2.5` 在这轮 `gsm8k smoke20` 上，单智能体 direct reasoning 强不强？

答案：很强，而且在 `single_agent_reasoning_json_v1` 协议下是 `1.0`。

### 问题 2

在多智能体/通信/触发/审计框架下，`gsm8k` 会不会自动保持 `1.0`？

答案：不会。很多框架只到 `0.75~0.90`，也有一些只有 `0.45~0.60`。

### 问题 3

那这些框架是不是都没价值？

答案：也不是。

更合理的理解是：

- `gsm8k` 对它们不一定是最公平的主战场
- 因为模型本体已经很强，协议噪声很容易淹没通信收益
- 所以这类任务更适合考察“能不能少花成本不掉点”，而不是“还能不能继续涨很多点”

## 7. 对你后续写报告或论文的建议

### 7.1 不要直接写成

“MiMo-V2.5 在 `gsm8k` 上是 `1.0`，所以其他方法没到 `1.0` 说明方法有问题。”

这个说法太粗，会被 prompt/protocol 差异反驳。

### 7.2 更好的写法

可以写成：

“在 `single_agent_reasoning_json_v1` 下，MiMo-V2.5 在 `gsm8k smoke20` 上达到 `1.0`，说明该子集对底座模型并不构成显著难度。后续多智能体与通信方法在 `gsm8k` 上的差异因此应主要解读为协议负担、通信噪声与资源效率的变化，而非底座模型本身的上限变化。”

### 7.3 如果要做严谨对比

建议优先在同 family 内解释：

- `VoC v2` 对比 `disagreement`
- `CUE` 对比 `disagreement / always`
- `SPARC` 对比自己的 ablation
- `DALA-lite` 对比 `budget_random / budget_confidence / all_to_all`

而不是直接跨 family 横比。

## 8. 最后一句判断

这轮 `gsm8k` 结果最值得记住的不是“有没有到 `1.0`”，而是：

- `1.0` 证明了底座模型在 direct reasoning 协议下非常强
- 其他 family 的掉点，很多时候是在测“协议和通信设计有没有干扰原本很强的解题能力”
- 因此 `gsm8k` 更适合作为“效率与干扰分析集”，而不是所有通信方法的唯一主证明场

## 9. 相关结果路径

- 单智能体主基线：`runs/single_agent/main_baselines/smoke20/20260505T050226Z-main_baselines-smoke20-xiaomimimo-mimo-v2.5`
- 多智能体 clean smoke：`runs/multi_agent/vanilla_mad_clean_smoke/smoke20/20260505T050948Z-vanilla_mad_clean_smoke-smoke20-xiaomimimo-mimo-v2.5`
- Free-MAD-lite：`runs/free_mad_lite/free_mad_lite_v1/smoke20/20260505T050228Z-free_mad_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- DALA-lite：`runs/budget_comm/dala_lite_same_context_v1/smoke20/20260505T050230Z-dala_lite_same_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- SID-lite：`runs/sid_lite/sid_lite_v1/smoke20/20260505T050232Z-sid_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- Trigger early-exit：`runs/selective_comm/trigger_early_exit_v1/smoke20/20260505T050233Z-trigger_early_exit_v1-smoke20-xiaomimimo-mimo-v2.5`
- VoC v2：`runs/selective_comm/trigger_voc_v2/smoke20/20260505T053920Z-trigger_voc_v2-smoke20-xiaomimimo-mimo-v2.5`
- CUE：`runs/cue/cue_v1/smoke20/20260505T053917Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- SPARC：`runs/sparc/sparc_v1_smoke/smoke20/20260505T050235Z-sparc_v1_smoke-smoke20-xiaomimimo-mimo-v2.5`
