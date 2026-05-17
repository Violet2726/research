# 顶会投稿研究约束

## 0. 文档说明

本文档面向顶会投稿前的研究收束阶段，用于统一当前项目的论文定位、实验解释口径和后续研究路线。

文档基于当前 authoritative `count100` faithful matrix 结果，而不是早期探索性运行或已退出正式矩阵的历史配置。

当前项目不应被解释为“多个多智能体方法的结果堆叠”，而应被解释为一个围绕固定预算多智能体推理展开的研究框架。核心问题是：

> 在有限 token 与调用预算下，系统应如何识别高价值分歧、选择性通信、分配通信预算，并对关键分歧进行局部验证。

本文档的作用有三点：

- 明确哪些结论可作为论文主张，哪些只能作为辅助或边界结果。
- 约束后续实验不得为了提升局部结果而破坏 faithful / equal-budget 评估口径。
- 将 same-context 与 split-context 分开解释，避免把不同问题设置混入单一 overall 结论。

## 1. 核心研究约束

## 1.1 不主张多智能体天然优于单智能体

论文主张必须限定在固定预算、明确 setting 和 matched baseline 下。

任何多智能体提升都必须说明相对哪个对照：

- no-comm control；
- full-communication reference；
- full-context reference；
- budget-matched single-agent baseline。

不能把多调用、多 token 或更长推理直接解释成多智能体结构收益。

## 1.2 same-context 与 split-context 必须分开报告

same-context 研究：

- 通信是否值得；
- 何时通信；
- 如何避免无效通信或错误改判。

split-context 研究：

- 通信是否必要；
- 通信能恢复多少信息缺口；
- 什么消息内容最有恢复价值。

两者不能合并成单一主 overall。

## 1.3 所有成本必须计入预算

以下成本都必须进入 total tokens / total calls：

- 初始求解；
- trigger；
- routing；
- communication；
- belief update；
- audit；
- judge；
- aggregation 中的模型调用。

不得把控制器、审计器、路由器或验证器视为免费模块。

## 1.4 方法比较必须 faithful

不允许为了提高某个方法结果而临时加入：

- fallback；
- 额外 judge；
- 额外 early exit；
- 额外通信轮数；
- 额外 agent；
- 额外审计模块；
- 数据集专用阈值或策略。

每个方法必须按固定配置和阶段图运行。

## 1.5 主表只放固定 headline 子集

当前建议的 same-context 主表：

- `trigger_early_exit_main`
- `voc_trigger_main`
- `dala_lite_same_context_main`
- `end_to_end_main`

当前建议的 split-context 主表：

- `hotpotqa_split_context_communication_necessity`
- `dala_lite_split_context_main`

其他 family 应作为 supporting 或 diagnostic evidence，而不是全部并列为主贡献。

## 1.6 accepted 不等于 headline

faithful gate 通过只说明该 experiment 在当前验收规则下可接受。
即便后续补充 `global_total_board` 这类横向景观视图，也只能把它当作扫描入口，
不能把 accepted、headline、supporting、diagnostic 压成同一语义层级。

它不等于：

- 适合成为论文主贡献；
- 在所有 setting 下最强；
- 可以被包装成全面优越。

例如：

- `cue_black_box_utility_main` 更适合作为统一 utility 方向的诊断结果；
- `sid_lite_mechanism_validation` 更适合作为压缩早退的边界结果；
- `content_ablation` 更适合作为消息内容失败风险的分析结果。

## 1.7 必须保留弱结果和边界结果

顶会论文不应只展示有利结果。

较弱结果可以用于说明：

- 哪些设计轴不充分；
- 哪些假设不成立；
- 哪些模块需要后续改进；
- 为什么最终框架需要强调高价值分歧，而不是默认全量 debate。

## 1.8 不随意扩展 family

后续重点不应是无约束地新增方法 family。

优先级应转向：

- 冻结当前主线；
- 补充更大规模 confirmatory phase；
- 统计显著性；
- 成本前沿图；
- 机制分析；
- 更强 equal-budget baseline。

允许的例外只有两类：

- 直接对应高价值顶会论文、且能进入统一 matrix / validation / reporting 契约的正式复现 family；
- 明确服务主结论证据缺口、而不是为了局部分数追涨的 reproduction family。

当前已启用的例外是：

- `dog_graph`：结构化图推理复现支线。`dog_graph_main` 现在代表 DoG 原论文高保真主线，`dog_graph_static_ablation` 只保留为静态候选子图 legacy 对照。它进入 `reproduction_matrix`，但不直接改写 same-context / split-context 主论文口径。
- `table_critic`：结构化表推理复现支线。`table_critic_main` 现在代表 Table-Critic 原论文主复现主线，当前聚焦 `WikiTQ / TabFact`，不纳入 `Binder / Dater`。它进入 `reproduction_matrix`，但不进入 `faithful_matrix`。
- `colmad`：协作监督协议复现支线。`colmad_realmistake_main` 现在代表 ColMAD 的正式复现主线，聚焦 `ReaLMistake` 三类错误检测任务与 `competitive vs collaborative` 协议差异。它进入 `reproduction_matrix`，但不进入 `faithful_matrix`。
- `econ`：低通信协调论文复现支线。`econ_same_context_main` 现在代表 ECON 的正式复现主线，直接进入当前 `faithful_matrix` 的 `same_context / supporting` 分层，用于验证 belief-driven coordination 是否优于无通信投票并以更低成本接近高通信参考线。
- `macnet`：拓扑协作论文复现支线。`macnet_paper_main` 当前暂停在 `count20`，继续保留为平行复现线；`macnet_scaling_study` 负责规模曲线与调用数匹配控制。它们进入 `reproduction_matrix`，但不进入 `faithful_matrix`。

## 1.11 矩阵分层约束

- `faithful_matrix` 只服务主论文问题，不承载图/表/拓扑复现支线。
- `reproduction_matrix` 只允许在各自 `track` 内比较，不生成跨 `graph_reasoning / table_reasoning / oversight_protocol / topology_collaboration` 的统一总榜。
- `scaling` 条目只能进入 reproduction matrix 的 scaling section，不能被压进 canonical 主表。

## 1.9 必须补强 equal-budget 反方基线

为回应“单智能体在同等 thinking-token 下可能更强”的审稿质疑，应增加强单智能体 equal-budget baseline。

建议包括：

- budget-matched long CoT；
- budget-matched self-consistency；
- budget-matched full-context single-agent；
- 与主方法 calls / token 接近的 single-agent baseline。

这些 baseline 的作用不是保证多智能体获胜，而是确认多智能体收益是否仍有独立解释。

## 1.10 结果解释必须以数据为准

所有主张都应对应具体表格或图：

- faithful score；
- delta vs no-comm；
- delta vs full reference；
- token ratio；
- calls per question；
- communication tokens；
- stage ceiling gap；
- helpful / harmful communication rate；
- bootstrap confidence interval。

不得用“更智能”“更协作”“更会推理”等泛化描述替代数据证据。

## 2. 后续研究路线约束

## 2.1 第一阶段：冻结 headline 子集

冻结以下内容：

- headline experiment；
- primary method；
- no-comm control；
- full reference；
- acceptance gate；
- cost accounting；
- phase 命名。

冻结后不再为了局部结果调方法结构。

## 2.2 第二阶段：增加 confirmatory phase

`count100` 适合判断方向，但不应作为顶会最终主实验规模。

建议增加：

- `main300`；
- 或每个核心数据集 `300-500` 样本；
- 或按资源允许扩展到更大确认集。

该阶段必须沿用冻结协议。

## 2.3 第三阶段：加入统计显著性

必须补：

- bootstrap confidence interval；
- paired win/loss；
- McNemar-style test；
- dataset-wise breakdown。

优先比较：

- `hybrid_trigger` vs `always_communicate`；
- `hybrid_trigger` vs `sc_6`；
- `voc_trigger_v2` vs `mv_6`；
- `dala_lite_same_context_main` vs `all_to_all_full`；
- `hotpotqa_split_context_communication_necessity` vs `split_no_comm_mv3`；
- `dala_lite_split_context_main` vs `all_to_all_full`。

## 2.4 第四阶段：补强 single-agent equal-budget baseline

新增 baseline 应作为评估配套，不应改写方法本体。

目标是回答：

> 当单智能体获得近似相同 token / call 预算时，多智能体方法是否仍有独立价值？

## 2.5 第五阶段：绘制机制图

至少绘制：

- budget frontier；
- trigger utility curve；
- stage ceiling gap；
- helpful / harmful communication breakdown；
- same-context 与 split-context 分轨结果图。

这些图用于支撑机制解释，而不是只展示最终分数。

## 2.6 第六阶段：决定主文与 appendix 分配

主文应放：

- headline 方法；
- same-context / split-context 主表；
- 关键成本图；
- 关键统计检验；
- 主要机制分析。

appendix 应放：

- supporting family；
- diagnostic family；
- `global_total_board` 等横向景观视图；
- 失败案例；
- 完整 per-dataset 表；
- prompt 和配置细节。

## 3. 最终路线

最终论文路线应收束为：

> 固定预算下的多智能体高价值分歧处理框架。

而不是：

> 多智能体 debate 的泛化优越性证明。

前者更窄，但更可证、更自洽，也更符合当前结果和最新研究趋势。
