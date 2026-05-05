# `hotpotqa` 跨实验统一对比说明

## 1. 先给结论

`HotpotQA` 是这轮实验里最容易把不同方法真正拉开的数据集之一，但要先分清两种完全不同的设定：

1. `same-context HotpotQA`
2. `split-context HotpotQA`

这两者不能混为一谈。

如果只看 one-line 结论：

- 在 `same-context` 下，很多通信方法并没有稳定优于强无通信基线，甚至经常更差。
- 在 `split-context` 下，通信是必要的，但当前消息协议只能部分修复信息缺失，远未恢复到 full-context 上限。

也就是说：

> `HotpotQA` 不是在告诉我们“通信总是有用”，而是在告诉我们“信息缺失时通信很重要；信息已经共享时，低质量通信反而可能坏事”。

## 2. 先分设定，再比较

### 2.1 `same-context`

这类实验里，各 agent 基本面对同一个完整问题上下文。

相关 family：

- `single_agent/main_baselines`
- `single_agent/main_table_same_context`
- `multi_agent`
- `free_mad_lite`
- `budget_comm/dala_lite_same_context_v1`
- `sid_lite`
- `selective_comm`
- `cue`
- `sparc`

### 2.2 `split-context`

这类实验里，信息被切分给不同 agent，通信承担了“补齐证据缺口”的职责。

相关 family：

- `comm_necessary/hotpotqa_split_evidence_v1`
- `comm_necessary/hotpotqa_split500_main`
- `budget_comm/dala_lite_split_context_v1`

这类结果与 same-context 的方法含义不同，必须单独看。

## 3. same-context 代表性结果总表

| Family | Experiment | Representative Methods | Accuracy | Total Tokens | Calls/Q | 直接结论 |
| --- | --- | --- | ---: | ---: | ---: | --- |
| `single_agent` | `main_baselines` | `cot_1 / mv_5 / sc_5` | `0.65 / 0.85 / 0.75` | `1597.15 / 7998.60 / 8009.15` | `1 / 5 / 5` | 强无通信多数投票已经很强 |
| `single_agent` | `main_table_same_context` | `cot_1 / mv_3 / sc_5` | `0.65 / 0.75 / 0.75` | `1597.15 / 4794.55 / 8009.15` | `1 / 3 / 5` | `mv_3` 已经明显优于 `cot_1` |
| `multi_agent` | `debate_vs_vote_controlled` | `mad_3a_r1` | `0.75` | `10627.85` | `6` | 一轮三 agent debate 没有压过强无通信基线 |
| `multi_agent` | `vanilla_mad_clean_smoke` | `cot_1 / mad_2a_r1 / mv_4 / sc_4` | `0.70 / 0.70 / 0.70 / 0.80` | `1626.50 / 6796.65 / 6457.55 / 6485.15` | `1 / 4 / 4 / 4` | debate 没占优，`sc_4` 最强 |
| `free_mad_lite` | `free_mad_lite_v1` | `mv_3_initial / vanilla_mad / anti_conformity / trajectory_judge` | `0.70 / 0.65 / 0.65 / 0.70` | `4898.65 / 10463.50 / 10578.55 / 12732.95` | `3 / 6 / 6 / 7` | 反从众没有带来帮助 |
| `budget_comm` | `dala_lite_same_context_v1` | `mv_3 / all_to_all / budget_random / budget_confidence / dala_lite` | `0.70 / 0.70 / 0.70 / 0.70 / 0.70` | `5467.30 / 11513.05 / 11052.95 / 11052.45 / 11018.50` | `3 / 6 / 6 / 6 / 6` | same-context 上通信基本没带来额外收益 |
| `sid_lite` | `sid_lite_v1` | `mv_3 / always_full / compression_only / sid_lite` | `0.75 / 0.75 / 0.80 / 0.80` | `5306.70 / 11841.95 / 10970.45 / 7245.55` | `3 / 6 / 6 / 4.05` | 压缩和早退反而更合适 |
| `selective_comm` | `trigger_early_exit_v1` | `mv_3 / always / disagreement / confidence / sc_6` | `0.75 / 0.80 / 0.75 / 0.75 / 0.80` | `5137.00 / 10999.00 / 6800.95 / 5137.00 / 10320.75` | `3 / 6 / 3.75 / 3 / 6` | 重通信不稳，`sc_6` 仍强 |
| `selective_comm` | `trigger_voc_v2` | `mv_3 / always / disagreement / voc_v2 / confidence` | `0.80 / 0.75 / 0.75 / 0.75 / 0.80` | `5271.45 / 6694.50 / 5709.65 / 5850.55 / 5271.45` | `3 / 6 / 3.9 / 4.2 / 3` | `VoC` 在 `HotpotQA` 上没有打赢不通信 |
| `cue` | `cue_v1` | `mv_3 / always / disagreement / cue_v1` | `0.80 / 0.75 / 0.75 / 0.80` | `5342.50 / 11976.60 / 6940.55 / 5650.00` | `3 / 6.2 / 3.75 / 3.15` | `CUE` 通过少通信守住了强 baseline |
| `sparc` | `sparc_v1_smoke` | `mv_3 / hybrid_trigger_baseline / sparc_v1 / always` | `0.70 / 0.70 / 0.80 / 0.80` | `5237.20 / 7464.85 / 7976.90 / 11305.30` | `3 / 4.2 / 4.45 / 6.25` | `SPARC` 比 always 更省，但并没有超越 `0.80` |
| `sparc` | `content_ablation_v1` | `mv_3 / full_cot / disagreement_step_only / critical_evidence_only` | `0.70 / 0.80 / 0.70 / 0.70` | `5237.20 / 11408.70 / 10699.75 / 10793.25` | `3 / 6 / 6 / 6` | full message 才有明显收益 |
| `sparc` | `auditing_ablation_v1` | `majority_vote / final_round_vote / local_auditing / single_judge` | `0.70 / 0.70 / 0.80 / 0.80` | `5237.20 / 10793.25 / 11305.30 / 12677.55` | `3 / 6 / 6.25 / 7` | 审计有帮助，但代价很高 |

## 4. split-context 代表性结果总表

### 4.1 `comm_necessary`

| Method | Ans EM | Joint F1 | Total Tokens | Comm Tokens | Calls/Q | 直接结论 |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `full_context_single` | `0.60` | `0.5239` | `1947.10` | `0.00` | `1` | 上限参考 |
| `split_no_comm_mv3` | `0.25` | `0.1976` | `2965.15` | `0.00` | `3` | 不通信时明显塌陷 |
| `answer_only_exchange` | `0.40` | `0.3266` | `6535.85` | `78.70` | `6` | 答案交换能部分修复 |
| `evidence_exchange` | `0.40` | `0.3115` | `6858.45` | `426.50` | `6` | 证据交换也能修复，但不够 |
| `full_packet_exchange` | `0.45` | `0.3792` | `7502.65` | `1076.50` | `6` | 消息更全，恢复更多，但离上限仍远 |

### 4.2 `budget_comm/dala_lite_split_context_v1`

| Method | Accuracy | Total Tokens | Comm Tokens | Calls/Q | 直接结论 |
| --- | ---: | ---: | ---: | ---: | --- |
| `mv_3` | `0.25` | `2809.65` | `0.00` | `3` | 不通信失效 |
| `all_to_all_full` | `0.65` | `6445.45` | `370.35` | `6` | 全通信能恢复 |
| `budget_random` | `0.40` | `5870.10` | `112.95` | `6` | 预算乱花会失败 |
| `budget_confidence` | `0.65` | `5834.20` | `103.90` | `6` | 简单启发式已能恢复大部分 |
| `dala_lite` | `0.60` | `5822.05` | `101.05` | `6` | 成本低，但这轮略输 `budget_confidence` |

## 5. same-context 上最关键的结论

### 5.1 很多通信方法没有系统性优于不通信

`HotpotQA same-context` 的一个核心现象是：

- 很多强无通信方法已经不差
- 很多通信方法并没有把分数继续推高
- 有时还会因为错误共识或无效讨论而掉点

最典型的例子：

- `single_agent/main_baselines mv_5 = 0.85`
- `multi_agent/vanilla_mad_clean_smoke sc_4 = 0.80`
- `selective_comm/trigger_voc_v2 mv_3 = 0.80`
- `cue/cue_v1 mv_3 = 0.80`

但：

- `VoC v2` 的 `always/disagreement/voc_v2` 大多只有 `0.75`
- `MAD` 也没有超过 `sc_4`
- `DALA-lite same-context` 几乎全线都在 `0.70`

### 5.2 这不是说通信没价值，而是说“答案级通信不够”

`HotpotQA` 的难点主要在：

- 证据定位
- 多跳关联
- 支持事实组合

因此，如果通信只是在交换答案、置信度、简略分歧，很容易出现：

- 大家更快达成共识
- 但共识并不更正确

这也是为什么 `debate_vs_vote_controlled` 里会出现：

- `post_debate_consensus_rate = 0.90`
- `wrong_consensus_rate = 0.20`

## 6. split-context 上最关键的结论

### 6.1 通信必要性被非常清楚地证明了

`comm_necessary` 的最强信号是：

- `full_context_single = 0.60`
- `split_no_comm_mv3 = 0.25`

也就是说，只要把信息拆开，不通信时准确率会直接塌掉。

这说明：

- 在 `HotpotQA split-context` 里，通信不是锦上添花，而是雪中送炭

### 6.2 但当前通信协议还远没有恢复到 full-context 上限

即使最好的 `full_packet_exchange`：

- `Ans EM = 0.45`
- `Joint F1 = 0.3792`

仍然明显低于：

- `full_context_single Ans EM = 0.60`
- `full_context_single Joint F1 = 0.5239`

这说明现在真正缺的不是“有没有发消息”，而是：

- 消息有没有把最关键的证据组织清楚
- 接收方有没有真正把证据融合好

### 6.3 `DALA-lite split-context` 的启发

这条线也很有意思：

- `all_to_all_full = 0.65`
- `budget_confidence = 0.65`
- `dala_lite = 0.60`
- `budget_random = 0.40`

它说明：

- split-context 下，预算控制确实可以做
- 但如果预算分配不准，代价会很大

## 7. 哪些比较最可信

### 7.1 最可信的 same-context 结论

#### `multi_agent/vanilla_mad_clean_smoke`

- `mad_2a_r1 = 0.70`
- `mv_4 = 0.70`
- `sc_4 = 0.80`

说明：

- debate 在这个协议下没有提供净增益

#### `free_mad_lite`

- `vanilla_mad = 0.65`
- `anti_conformity = 0.65`
- `trajectory_judge = 0.70`

说明：

- 反从众在 `HotpotQA` 上没有帮助
- judge 只能部分补回损失

#### `sid_lite`

- `mv_3 = 0.75`
- `always_full = 0.75`
- `sid_lite = 0.80`

说明：

- 在这个任务上，“少说一点、早点停”反而优于“全部展开说”

#### `VoC v2`

- `mv_3 = 0.80`
- `always = 0.75`
- `disagreement = 0.75`
- `voc_v2 = 0.75`

这是本轮非常重要的负面结果：

- `HotpotQA same-context` 上，当前 trigger 并没有找到真正有用的 communication opportunity

#### `CUE`

- `mv_3 = 0.80`
- `always = 0.75`
- `disagreement = 0.75`
- `cue_v1 = 0.80`

说明：

- `CUE` 在这个任务上最关键的优点不是“更会通信”
- 而是“更知道什么时候别通信”

#### `SPARC`

- `hybrid_trigger_baseline = 0.70`
- `sparc_v1 = 0.80`
- `always = 0.80`

说明：

- `SPARC` 能通过结构化内容和局部审计把 selective 路线拉回到 always 水平
- 但没有超越无通信最强前沿

### 7.2 最可信的 split-context 结论

#### `comm_necessary`

这是整轮最可信的 `HotpotQA` 机制实验之一：

- 不通信会塌
- 轻通信能救一部分
- 证据与完整 packet 比纯答案更有恢复力
- 但离真正恢复还有很大差距

#### `budget_comm/dala_lite_split_context_v1`

也是可信的，因为控制组很清楚：

- `mv_3 = 0.25`
- `all_to_all_full = 0.65`
- `budget_random = 0.40`

说明预算是否“花在对的地方”极其重要。

## 8. 为什么 `HotpotQA` 会给出这么强的负面反馈

### 8.1 任务难点在证据而不在答案本身

很多通信框架默认认为：

- 只要 agent 把最终答案和理由互相交换一下，就能相互纠正

但 `HotpotQA` 更像是：

- 关键在于证据链是否完整
- 支持事实是否匹配
- 多跳推理是否把两个实体正确接上

### 8.2 same-context 下的“重复沟通”容易变成浪费

如果每个 agent 原本就看到了同样上下文，那么很多通信其实是在：

- 重复已经存在的信息
- 放大错误解释
- 让系统更快收敛到错误答案

### 8.3 split-context 下的“消息太弱”又不够

而一旦信息真的被拆开，当前这些通信内容又往往不够强，导致：

- 明知道必须通信
- 但发出去的消息仍然不足以恢复 full-context 推理能力

## 9. 这个数据集对论文最重要的启发

如果你的论文最终目标是提出较通用的方法论，那么 `HotpotQA` 对主张的约束非常强：

### 9.1 不能把“是否通信”当成唯一问题

`VoC` 已经说明：

- 即使 trigger 做得还行
- 只要消息内容不对，结果照样不提升

### 9.2 必须把“传什么”纳入方法定义

`SPARC content ablation` 和 `comm_necessary` 一起说明：

- content design 是第一类问题，不是附属实现细节

### 9.3 必须允许“有些 same-context 任务不该频繁通信”

`CUE` 在 `HotpotQA` 上最好的地方，就是它不像 `always` 那样盲目交流。

因此更适合的论文表述应该是：

> 我们的方法不假设通信总是有益，而是在可观测冲突和证据缺口足够明显时，才启动定向通信，并根据任务结构决定消息内容与验证方式。

## 10. 最值得记住的 6 条结论

1. `HotpotQA same-context` 上，通信不是天然增益，很多方法不如强无通信基线。
2. `HotpotQA split-context` 上，通信是必要条件，不通信会明显塌陷。
3. `VoC v2` 在 `HotpotQA same-context` 上几乎是最清楚的负例之一，说明当前 trigger 语义与真正有益通信不匹配。
4. `CUE` 在 `HotpotQA same-context` 上的优势不是更会沟通，而是更会克制。
5. `SPARC` 说明“选择性触发 + 结构化内容 + 局部审计”这条路线比单纯 trigger-only 更接近正确方向。
6. 如果后续想把论文做成通用方法论，`HotpotQA` 会逼着你把证据感知消息设计纳入主框架。

## 11. 相关结果路径

- 单智能体主基线：`runs/single_agent/main_baselines/smoke20/20260505T050226Z-main_baselines-smoke20-xiaomimimo-mimo-v2.5`
- main-table same-context：`runs/single_agent/main_table_same_context/smoke20/20260505T050236Z-main_table_same_context-smoke20-xiaomimimo-mimo-v2.5`
- debate-vs-vote：`runs/multi_agent/debate_vs_vote_controlled/smoke20/20260505T050228Z-debate_vs_vote_controlled-smoke20-xiaomimimo-mimo-v2.5`
- Vanilla MAD clean smoke：`runs/multi_agent/vanilla_mad_clean_smoke/smoke20/20260505T050948Z-vanilla_mad_clean_smoke-smoke20-xiaomimimo-mimo-v2.5`
- Free-MAD-lite：`runs/free_mad_lite/free_mad_lite_v1/smoke20/20260505T050228Z-free_mad_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- DALA-lite same-context：`runs/budget_comm/dala_lite_same_context_v1/smoke20/20260505T050230Z-dala_lite_same_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- DALA-lite split-context：`runs/budget_comm/dala_lite_split_context_v1/smoke20/20260505T054055Z-dala_lite_split_context_v1-smoke20-xiaomimimo-mimo-v2.5`
- comm necessary：`runs/comm_necessary/hotpotqa_split_evidence_v1/smoke20/20260505T050231Z-hotpotqa_split_evidence_v1-smoke20-xiaomimimo-mimo-v2.5`
- SID-lite：`runs/sid_lite/sid_lite_v1/smoke20/20260505T050232Z-sid_lite_v1-smoke20-xiaomimimo-mimo-v2.5`
- Trigger early-exit：`runs/selective_comm/trigger_early_exit_v1/smoke20/20260505T050233Z-trigger_early_exit_v1-smoke20-xiaomimimo-mimo-v2.5`
- VoC v2：`runs/selective_comm/trigger_voc_v2/smoke20/20260505T053920Z-trigger_voc_v2-smoke20-xiaomimimo-mimo-v2.5`
- CUE：`runs/cue/cue_v1/smoke20/20260505T053917Z-cue_v1-smoke20-xiaomimimo-mimo-v2.5`
- SPARC：`runs/sparc/sparc_v1_smoke/smoke20/20260505T050235Z-sparc_v1_smoke-smoke20-xiaomimimo-mimo-v2.5`
