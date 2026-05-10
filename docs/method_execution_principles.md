# 各方法执行原则与实现细节说明

## 1. 文档目的

本文档用于系统说明当前项目中各实验方法的执行原则、阶段结构、关键输入输出、对照组设计和结果解释边界。

这份文档重点回答 4 类问题：

1. 每个方法到底在“做什么”。
2. 每个方法的执行流程是怎样展开的。
3. 不同方法之间哪些地方可以直接比较，哪些地方只能方向性比较。
4. `faithful_matrix` 是如何把这些方法统一编排到同一套 `smoke20 / pilot100` 主矩阵里的。

本文档主要覆盖当前 authoritative 主线中的 9 个 family：

- `single_agent`
- `multi_agent`
- `selective_comm`
- `budget_comm`
- `comm_necessary`
- `sid_lite`
- `free_mad_lite`
- `sparc`
- `cue`

同时补充说明共享基础设施 `experiment_core` 和顶层矩阵编排 `faithful_matrix` 的职责。

## 1.1 当前 authoritative 项目状态

为避免文档和当前代码结构脱节，先固定当前 authoritative 口径：

- `single_agent`
  只保留 `cot_1` 与 `sc_k`；`mv_*` 已从这个 family 中删除。
- `multi_agent`
  只保留一个正式实验入口：`configs/multi_agent/experiments/same_context_controlled_debate.toml`。
- `selective_comm`
  只保留两个正式实验入口：
  - `trigger_early_exit_main`
  - `voc_trigger_main`
- `comm_necessary`
  只保留一个正式实验入口：`configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml`。
  `smoke20 / pilot100 / main(split500)` 统一通过 phase 表达，不再拆成多个 experiment。
- `faithful_matrix`
  当前 authoritative 主矩阵是 `15` 个语义唯一目标，不再保留任何本地开发专用实验入口。

## 2. 全局设计原则

整个项目遵循 3 条全局原则。

### 2.1 方法 faithful

同一 family 内的方法结构应保持 faithful，不通过增加额外 early-exit、外接 fallback、额外裁判或新 message phase 去“修平”结果。

这意味着：

- `Stage A / Stage B / judge / audit` 等阶段是否存在，必须由方法本身决定。
- 触发规则、通信轮数、agent 数、消息模式、聚合规则属于方法定义的一部分。
- 后续允许优化的是 prompt 骨架、结构化输出恢复、评分归一化和报告口径，而不是方法执行图。

### 2.2 共享基础设施统一下沉

所有 family 共享以下能力：

- provider 调用
- 请求缓存
- 限流
- 数据集 split 选择
- 结构化输出校验与恢复
- run 目录组织
- validation 与汇总报告基础设施

这些能力都下沉到 `src/experiment_core/`，各实验包不彼此直接依赖。

### 2.3 运行产物路径统一

所有实验产物统一落到：

- `local/runs/<family>/<experiment>/<phase>/<run_id>/`
- `local/reports/<family>/`

顶层跨实验矩阵分析统一落到：

- `local/runs/faithful_matrix_iterative/...`
- `local/reports/faithful_matrix/...`

## 3. 共享基础设施

## 3.1 `experiment_core`

`experiment_core` 是唯一共享核心层，负责：

- `config.py`
  解析 benchmark、provider 和 model 配置。
- `datasets.py`
  根据冻结 split 选样本。
- `providers/`
  调用 OpenAI-compatible provider。
- `cache.py`
  以请求体为键做响应缓存。
- `rate_limits.py`
  用滑动窗口做 RPM / TPM 限流。
- `structured_output.py`
  对不同 output mode 做统一 schema 校验与恢复。
- `evaluation.py`
  做答案归一化、打分和投票聚合。
- `runtime.py`
  生成 `run_id`、写 `progress.json`。
- `workspace.py`
  统一 `local/runs/`、`local/reports/`、`local/cache/` 默认根目录。

## 3.2 结构化输出约束

当前共享的 output mode 包括：

- `core`
- `selective_comm`
- `budget_solver`
- `budget_belief_update`
- `sparc_solver`
- `sparc_message`
- `sparc_belief_update`
- `sparc_audit`
- `cue_solver`
- `cue_belief_update`
- `cue_audit`
- `comm_necessary_solver`
- `comm_necessary_belief`

统一原则是：

- 优先做严格 schema 校验。
- 校验失败时再走共享恢复逻辑。
- 结构化失败要和“答案内容错误”严格区分。

## 3.3 统一阶段术语

虽然各 family 细节不同，但项目里有一套稳定术语：

- `Stage A`
  独立求解阶段。每个 agent 先在没有显式互相交流的情况下给出答案和中间结构化字段。
- `Stage B`
  通信或 belief update 阶段。agent 基于 peer packet 或 debate message 更新自己的信念。
- `judge`
  在方法定义内负责全局裁决的组件，例如 trajectory judge 或 single judge。
- `audit`
  面向局部冲突的审计阶段，不是所有方法都存在。
- `control`
  与主方法预算匹配的无通信或更简单通信对照组。

## 4. `faithful_matrix` 的角色

`src/experiment_core/matrix/faithful_matrix.py` 不是方法本体，而是**顶层实验编排器**。

它负责：

1. 发现当前 phase 下所有有效 experiment。
2. 读取 `matrix_specs.py`，给每个 experiment 绑定：
   - `evaluation_track`
   - `primary_method_name`
   - `best_no_comm_candidates`
   - `full_comm_reference`
   - `full_context_reference`
3. 对不同 family 应用统一 runtime overrides：
   - `phase_name`
   - `model_ref`
   - `max_concurrent_requests`
   - `requests_per_minute_limit`
   - `tokens_per_minute_limit`
4. 顺序执行主矩阵。
5. 对每个 run 做健康复核：
   - `progress.json` 是否 completed
   - `run_validation.json` 是否 passed
   - summary rows 是否非空
6. 调用：
   - `faithful_analysis.py`
   - `faithful_acceptance.py`
   输出跨实验主表和 acceptance 总结。

需要特别强调的是：

- `faithful_matrix` 只负责**统一比较口径**。
- 它不改变各方法内部的执行图。
- 它不把某个方法修成另外一个方法。

## 5. `single_agent`

## 5.1 目标

`single_agent` 是整个项目的无通信参考系，负责提供：

- 单次 CoT 下界
- 同预算多次采样上界
- 后续所有通信类方法的强无通信比较对象

## 5.2 方法集合

当前主目录 `configs/single_agent/methods/common.toml` 定义了：

  - `cot_1`
  - `sc_3 / sc_5 / sc_7`

含义如下：

- `cot_1`
  单次 Chain-of-Thought，预算为 1 次调用。
- `sc_k`
  Self-Consistency，同题采样 `k` 次，再做多数聚合。

## 5.3 执行原则

`single_agent` 的关键执行原则是：

- 先把 `模型 × benchmark × method × rerun` 展开成 call spec。
- 每条调用彼此独立。
- 所有回答走统一的 `core` output schema。
- 样本级预测再按 method 做聚合和评分。

## 5.4 详细执行流程

1. 读取 experiment 配置和 phase。
2. 加载 method catalog。
3. 为每个 method 生成预算匹配的调用计划。
4. 并发发送请求。
5. 校验和恢复结构化输出。
6. 对同一题下多个 replicate 做：
   - 多数投票
   - 分数统计
7. 汇总出：
   - `predictions.jsonl`
   - `metrics.json`
   - `report.md`
   - `figure_manifest.json`
   - `figures/`
   - `paper_tables.md`
   - `run_validation.json`

## 5.5 它在整个项目中的意义

`single_agent` 不是待优化对象，而是比较锚点。

如果一个通信方法连对应的强无通信控制都守不住，就说明它至少在当前 faithful 结构下缺乏实际价值。

## 6. `multi_agent`

## 6.1 目标

`multi_agent` 负责回答一个更纯粹的问题：

**在相同或匹配预算下，显式 debate 是否比直接 vote 更有额外收益。**

它本质上是一个控制研究 family。

## 6.2 关键配置概念

- `setup`
  一个具体的 debate 配置实例，例如 `mad_3a_r1`。
- `protocol`
  决定 round 数、每轮输出约束等。
- `roster`
  决定 agent 个数和角色编排。
- `matched_controls`
  为当前 debate setup 指定预算匹配的无通信控制。当前正式配置只保留 `mv_k` 语义，不再在 `multi_agent` 内部混入 `sc_k`。

例如：

- `mad_2a_r1`
  对应 `2 agents + 1 debate round`
  匹配控制：
  - `cot_1`
  - `mv_4`
- `mad_3a_r1`
  对应 `3 agents + 1 debate round`
- 匹配控制：
  - `mv_6`

## 6.3 执行原则

它不是“随便多跑几个 agent 看结果”，而是一个严格控制的 paired comparison：

- 同一题使用同一批初始候选。
- 再比较：
  - 直接 vote
  - debate 后再 vote

这样能把 debate 的额外作用从“样本差异”里剥离出来。

## 6.4 详细执行流程

1. 读取 active setups。
2. 读取 matched controls。
3. 每题先跑所有 agent 的初始回答。
4. 如果 setup 带 debate round：
   - 生成显式 debate messages
   - 进入下一轮回答
5. 记录：
   - `agent_turns.jsonl`
   - `debate_messages.jsonl`
6. 题级做最终投票聚合。
7. 统计 debate 是否：
   - 纠错
   - 伤害
   - 保持不变

## 6.5 主要产物

- `final_predictions.jsonl`
- `metrics.json`
- `cost_breakdown.json`
- `debate_diagnostics.json`

## 6.6 解释边界

`multi_agent` 非常适合回答 debate-vs-vote 的机制问题，但不直接等价于“预算型统一方法论”。

### 6.7 最新收口说明

当前 `multi_agent` 的正式配置已经收口为单一入口 `same_context_controlled_debate`，并且：

- `matched_controls` 只保留 `mv_k` 语义，不再在 `multi_agent` 内部混入 `sc_k`；
- 无通信 matched control 的执行逻辑已经迁到共享层 `experiment_core/no_comm_controls.py`；
- 因此 `multi_agent` 现在更干净地表达为：
  - 方法本体负责 debate / vote；
  - 共享层负责无通信对照执行；
  - `faithful_matrix` 负责把两者放到同一比较口径里。

## 7. `selective_comm`

## 7.1 目标

`selective_comm` 关注：

**是否可以通过 trigger / early-exit 只在必要时通信。**

它的核心特色是“共享前缀”。

## 7.2 方法结构

它先运行共享的 `Stage A`，然后复用同一份中间结果给多种 trigger policy。

主实验里常见策略包括：

- `always_communicate`
- `disagreement_triggered`
- `confidence_triggered`
- `hybrid_trigger`
- `voc_trigger_v2`

控制组包括：

- `mv_3`
  直接复用共享 Stage A 的无通信多数票
- `mv_6`
- `sc_6`

这里要注意语义边界：

- `mv_3 / mv_6` 是共享 Stage A 基础上的无通信 vote 对照；
- `sc_6` 是外部 no-comm baseline，用于公平比较，但不属于 `selective_comm` 方法本体。

## 7.3 执行原则

关键原则是：

- 同一题只跑一次共享 Stage A。
- 不同 trigger policy 只是在此基础上复用。
- 只有真正触发通信的 policy 才共享执行 Stage B。

这样做的目的不是偷懒，而是：

- 保证不同策略面对的是同一组初始候选
- 降低重复网络请求
- 保留公平对照

## 7.4 详细执行流程

1. 共享运行 Stage A。
2. 从 Stage A 输出中提取：
   - 答案分歧
   - 置信度
   - claim / evidence 代理特征
3. 对每个 policy 计算是否触发。
4. 触发的策略共享执行 Stage B。
5. 写出：
   - `trigger_decisions.jsonl`
   - `policy_predictions.jsonl`
   - `policy_metrics.json`
   - `policy_diagnostics.json`
   - `oracle_trigger_eval.json`

## 7.5 方法差异

- `hybrid_trigger`
  综合分歧和置信代理做触发。
- `voc_trigger_v2`
  使用 claim-span 与 uncertainty-type 等 black-box proxy。
- `always_communicate`
  作为 full-comm 参照。

## 7.6 解释边界

因为 `mv_3` 复用了共享 Stage A，所以它是实验内无通信基线，但不等于物理上真正独立重新跑一遍的成本。

### 7.7 最新收口说明

当前 `selective_comm` 只保留两个正式 experiment：

- `trigger_early_exit_main`
- `voc_trigger_main`

同时，历史上的裁剪版和变体版配置已经移除，包括：

- `trigger_voc_v2_core_only`
- `trigger_voc_v2_equal_budget_gsm_strategy`
- `confidence_triggered_095`
- `hybrid_trigger_relaxed`
- `claim_divergence_triggered`

因此现在应把 `selective_comm` 理解为两条正式方法线：

- 一条是经典 trigger / early-exit 路线；
- 一条是 VoC v2 的 black-box proxy 路线。

## 8. `budget_comm`

## 8.1 目标

`budget_comm` 关注：

**在明确的通信预算约束下，哪些消息值得被发送。**

当前主线方法是 `dala_lite`。

## 8.2 方法集合

`budget_comm` 的标准方法序列是：

- `mv_3`
- `all_to_all_full`
- `budget_random`
- `budget_confidence`
- `dala_lite`

它们的角色分别是：

- `mv_3`
  无通信基线
- `all_to_all_full`
  full communication 上界
- `budget_random`
  随机预算分配
- `budget_confidence`
  用置信度做预算启发式
- `dala_lite`
  主方法

## 8.3 核心思想

`DALA-lite` 的核心不是简单“谁最不确定就发谁”，而是：

1. 从 Stage A 里提炼可比较候选特征。
2. 估计每个候选的 utility。
3. 将 utility 与消息成本组合成 density。
4. 在预算下做 tier 分配和选择。

## 8.4 关键内部对象

`budget_comm.logic` 里最关键的是：

- `METHOD_ORDER`
- `PACKET_MODE_ORDER`
- `build_shared_candidate_features`
- `build_dala_lite_decision`
- `apply_belief_update`

其中 `build_shared_candidate_features` 会为每个 agent 构造：

- disagreement score
- evidence score
- novelty score
- confidence score
- utility score
- density score

## 8.5 详细执行流程

1. 先做 budget calibration。
2. 共享运行 Stage A。
3. 从 Stage A 提炼候选特征。
4. 构建不同预算策略的决策：
   - random
   - confidence
   - DALA-lite
5. 为中标消息包执行 Stage B belief update。
6. 题级再做投票聚合。
7. 输出：
   - `candidate_packets.jsonl`
   - `auction_decisions.jsonl`
   - `belief_updates.jsonl`
   - `budget_diagnostics.json`

## 8.6 解释边界

`budget_comm` 不回答“通信是否必要”，它回答的是：
**在必须做预算分配时，怎样比随机或简单启发式更合理。**

## 9. `comm_necessary`

## 9.1 目标

`comm_necessary` 是 split-context family，核心问题是：

**当信息被硬切分时，通信是否构成解题能力恢复的必要步骤。**

当前聚焦 HotpotQA。

### 9.1.1 当前正式入口

`comm_necessary` 当前只有一个正式 experiment：

- `hotpotqa_split_context_communication_necessity`

它不是“多个相近实验配置的集合”，而是：

- 一个固定的 split-context HotpotQA 通信必要性实验；
- 通过不同 phase 运行不同规模的数据切片：
  - `smoke20`
  - `pilot100`
  - `main` (`split500_seed42`)

## 9.2 方法集合

主实验方法顺序固定为：

- `full_context_single`
- `split_no_comm_mv3`
- `answer_only_exchange`
- `evidence_exchange`
- `full_packet_exchange`

含义如下：

- `full_context_single`
  完整上下文上界
- `split_no_comm_mv3`
  信息切分但不通信的基线
- `answer_only_exchange`
  只交换答案
- `evidence_exchange`
  只交换证据
- `full_packet_exchange`
  交换完整 packet

## 9.3 核心对象

这个 family 的难点不只是答案，还包括 supporting facts。

因此它在逻辑层维护：

- supporting facts 归一化
- answer / support / joint 指标
- packet mode 压缩

## 9.4 执行原则

它先把原始 HotpotQA 样本切成多个局部视角，然后比较不同强度的通信是否能恢复 full-context 能力。

这意味着：

- 信息缺口是由实验设计显式制造的
- 不通信时的失分是有因果解释的
- 通信带来的恢复也更容易解释

## 9.5 详细执行流程

1. 对样本构造 split-context views。
2. 每个 agent 在自己可见 shard 上做 Stage A。
3. 根据方法构建不同强度的 packet：
   - answer_only
   - evidence
   - full_packet
4. 执行 Stage B belief update。
5. 聚合：
   - 最终答案
   - supporting facts
6. 生成：
   - `hotpot_predictions.json`
   - `metrics.json`
   - `diagnostics.json`

## 9.6 最近契约修正

当前版本已经明确：

- Stage A 可以在没有 grounded answer 时输出空答案作为 abstain。
- Stage B 属于 belief revision；若更新阶段仍为空，但 Stage A 已有 grounded answer，则默认保持原答案。

这保证了 turn-level schema 和 final prediction 语义一致。

### 9.7 最新收口说明

早期配置里曾把：

- `hotpotqa_split_evidence_v1`
- `hotpotqa_split500_main`

拆成两个 experiment 入口。当前版本已经把这两个入口收回到同一个正式 experiment `hotpotqa_split_context_communication_necessity`，原因是：

- 方法本体完全相同；
- 不同之处只是运行 phase 和样本规模；
- 因此差异应放在 phase 层表达，而不是放在 experiment 层表达。

## 10. `sid_lite`

## 10.1 目标

`sid_lite` 关注：

**能否通过高置信一致时 early-exit，以及低成本压缩消息，在预算内逼近 full communication。**

## 10.2 方法集合

- `mv_3`
- `always_full`
- `compression_only`
- `sid_lite`

含义如下：

- `mv_3`
  无通信多数票
- `always_full`
  全量通信上界
- `compression_only`
  一律通信，但只发压缩消息
- `sid_lite`
  先做 early-exit，再决定是否压缩通信

## 10.3 核心机制

`sid_lite.logic` 里最重要的是：

- `decide_early_exit`
- `project_message_packet`
- `apply_belief_update`

其中 early-exit 的判断依赖：

- 是否初始答案一致
- 平均置信度是否够高
- 置信度离散是否够小
- 是否存在无效 confidence

## 10.4 执行原则

`SID-lite` 的原则不是“有共识就停”，而是“高置信共识才停”。

当置信信号不可靠时，它会 fail-open 到通信路径，而不是把结构化失败误判成高置信。

## 10.5 详细执行流程

1. 共享 Stage A。
2. 做 early-exit 判断。
3. 若 early-exit：
   - 直接输出结果
4. 否则：
   - 将 Stage A 输出投影为 `full` 或 `compressed` packet
   - 进入 belief update
5. 最终题级聚合。
6. 输出：
   - `message_packets.jsonl`
   - `belief_updates.jsonl`
   - `diagnostics.json`

## 10.6 解释边界

如果某个 benchmark 上 `stage_ceiling_gap` 很小但 `delta_vs_full_comm` 仍很差，就说明它已经接近 faithful 上限，不应再用结构改写去“救”。

## 11. `free_mad_lite`

## 11.1 目标

`free_mad_lite` 用来验证：

**单轮 anti-conformity debate + LLM trajectory judge 是否比普通 debate 更有信息增益。**

它不是完整 Free-MAD 的 score-model reproduction，而是轻量 faithful 机制版。

## 11.2 方法集合

- `mv_3_initial`
- `vanilla_mad_r1_final_vote`
- `anti_conformity_final_vote`
- `free_mad_lite_llm_trajectory`

含义如下：

- `mv_3_initial`
  初始无通信投票
- `vanilla_mad_r1_final_vote`
  普通单轮 debate 后再 vote
- `anti_conformity_final_vote`
  强化反从众发言后再 vote
- `free_mad_lite_llm_trajectory`
  在 anti-conformity 轨迹上再加 trajectory judge

## 11.3 核心机制

`free_mad_lite.logic` 的关键点是：

- 轨迹裁决 `build_trajectory_decision`
- judge 失败时的确定性 fallback
- trajectory hash

这保证了 judge 即使失败，也有稳定可复现的回退规则。

## 11.4 详细执行流程

1. 所有 agent 先做 Stage A。
2. 进入单轮 anti-conformity debate。
3. 收集 debate 轨迹。
4. 调 trajectory judge。
5. 若 judge 无效：
   - 落回确定性 fallback
6. 生成最终预测和诊断。

## 11.5 解释边界

如果它在准确率上很强，但通信 token 并没有下降，那它更适合作为“性能型机制结果”，而不是“预算型 headline”。

## 12. `sparc`

## 12.1 目标

`sparc` 关注：

**选择何种消息内容、何时通信、如何做局部审计，是否能得到稳定的 selective communication 增益。**

它是当前项目里结构最丰富的 family 之一。

## 12.2 experiment kind

当前主要有 3 类 experiment kind：

- `sparc_v1`
- `content_ablation`
- `auditing_ablation`

另有聚合与审计耦合版：

- `aggregation_auditing_ablation`

## 12.3 消息模式

`sparc.logic` 中定义了稳定的消息模式序列：

- `full_cot`
- `answer_only`
- `answer_confidence`
- `disagreement_step_only`
- `critical_evidence_only`
- `task_adaptive`

`task_adaptive` 会根据 dataset 自动映射：

- `gsm8k / strategyqa / math500 / mmlu_pro / gpqa_diamond`
  倾向 `disagreement_step_only`
- `hotpotqa`
  倾向 `critical_evidence_only`

## 12.4 聚合与审计

`sparc` 内部还有一套显式聚合方式：

- `majority_vote`
- `weighted_vote_fallback`
- `single_judge`
- `final_round_vote`
- `local_auditing`

这让它不仅能研究“是否通信”，还能研究：

- 消息内容是否重要
- 审计方式是否重要
- 聚合方式是否重要

## 12.5 执行原则

`SPARC` 的执行图通常是：

1. 共享 Stage A
2. 选择触发策略
3. 选择消息模式
4. 做 Stage B belief update
5. 视 experiment kind 决定是否 single judge 或 local auditing
6. 做最终聚合

## 12.6 `sparc_v1` 的特点

`end_to_end_main` 的配置里显式声明：

- 默认 trigger 为 `hybrid_trigger`
- fallback trigger 为 `disagreement_triggered`
- dataset 绑定的固定 message mode

因此 `sparc_v1` 不是简单 trigger baseline，而是：
**trigger + content mode + local auditing 的联合方法。**

## 13. `cue`

## 13.1 目标

`cue` 的目标是：

**在黑盒 LLM 环境中估计通信的效用，只对高价值局部冲突触发通信。**

它是“统一方法论”导向最强的 family。

## 13.2 结构化对象

`cue.schemas` 中定义了 3 个核心结构：

- `AgentPacket`
  Stage A 暴露给外部的压缩状态。
- `ConflictObject`
  对局部冲突的结构化表示。
- `UtilityBreakdown`
  通信收益、风险和成本的分解结果。

## 13.3 策略集合

当前主实验包含：

- `always_communicate`
- `disagreement_triggered`
- `consensus_freeze`
- `cue_black_box_utility_main`

## 13.4 核心机制

`cue.logic` 负责：

- 从 Stage A 提炼信号
- 构造 `ConflictObject`
- 计算 utility
- 决定 trigger
- 执行 belief update
- 必要时选 audit pair

当前核心信号包括：

- `answer_entropy`
- `mean_confidence`
- `confidence_spread`
- `claim_conflict_rate`
- `evidence_gap`
- `fragile_consensus`
- `format_conflict_risk`
- `majority_pressure_risk`

## 13.5 执行原则

`CUE` 的原则不是“有分歧就聊”，而是：

1. 先把局部冲突压缩成可计算对象。
2. 再根据收益、风险和成本估计 utility。
3. 只有 utility 过阈值时才触发通信。

因此它比普通 selective trigger 更像一个统一决策框架。

## 13.6 详细执行流程

1. 共享运行 Stage A。
2. 对每题提炼信号。
3. 构建 `ConflictObject`。
4. 计算 `UtilityBreakdown`。
5. 针对每个 policy 决定是否触发。
6. 触发的策略复用共享 communication 结果。
7. 若启用 audit，再做局部冲突审计。
8. 输出：
   - `policy_predictions.jsonl`
   - `policy_metrics.json`
   - `policy_diagnostics.json`
   - `oracle_trigger_eval.json`

## 13.7 解释边界

`CUE` 的价值主要在统一性和可解释性，不保证它在所有 benchmark 上都是经验上最强的 selective 方法。

## 14. Cross-Family 比较时必须注意的边界

## 14.1 同 family 内比较通常更可信

例如：

- `dala_lite` vs `all_to_all_full`
- `voc_trigger_v2` vs `always_communicate`
- `sid_lite` vs `always_full`

这类比较通常共享：

- 相同 Stage A
- 相同 prompt family
- 相同 artifact 契约

因此解释更直接。

## 14.2 跨 family 比较必须谨慎

例如：

- `cue_black_box_utility_main` vs `sparc_v1`
- `trigger_early_exit` vs `dala_lite`

这些方法虽然都在 `faithful_matrix` 里，但不表示它们拥有完全相同的协议结构。

跨 family 更适合比较的是：

- 是否守住强无通信基线
- 相对 full communication 的掉点
- 通信 token frontier
- `stage_ceiling_gap`
- `engineering_noise_gap`

而不是只比较单个 accuracy 数字。

## 15. 如何阅读当前 authoritative 结果

如果你要结合 `smoke20 / pilot100` 结果解读这些方法，建议按下面顺序看：

1. 看 `state.json`
   判断 run 是否完整健康。
2. 看 `run_validation.json`
   判断有没有工程事故。
3. 看 `faithful_analysis.md`
   关注：
   - `delta_vs_best_no_comm`
   - `delta_vs_full_comm`
   - `stage_ceiling_gap`
   - `engineering_noise_gap`
4. 看 `acceptance_summary.md`
   判断是否过当前 faithful gate。
5. 再回到各 family 自己的 report
   看它们的机制分析和失败样例。

## 16. 一句话总结

这个项目不是一套单一算法，而是一组围绕“多智能体推理中的通信价值”展开的 family。

它们的分工大致是：

- `single_agent`
  提供强无通信参考系
- `multi_agent`
  做 debate-vs-vote 控制研究
- `selective_comm`
  研究 trigger / early-exit
- `budget_comm`
  研究预算分配
- `comm_necessary`
  研究 split-context 下通信必要性
- `sid_lite`
  研究早退与压缩通信
- `free_mad_lite`
  研究 anti-conformity 与 trajectory judging
- `sparc`
  研究内容选择、触发和局部审计
- `cue`
  研究黑盒条件下的统一通信效用决策

`faithful_matrix` 则把它们放到同一套 phase-aware 主矩阵中，统一做 faithful 分析与 acceptance 判定。

补一句当前的项目全局理解：

- `single_agent` 提供强无通信参考；
- `multi_agent` 做受控 debate-vs-vote；
- `selective_comm` 做 trigger / early-exit；
- `budget_comm` 做预算分配；
- `comm_necessary` 做 split-context 下的通信必要性；
- `sid_lite`、`free_mad_lite`、`sparc`、`cue` 分别提供压缩通信、反从众、局部审计和统一效用决策视角；
- `faithful_matrix` 只统一比较口径，不改变任何 family 的方法结构。
