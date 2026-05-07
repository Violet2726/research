# 基于方法草稿与 Pilot100 结果的论文建议

## 1. 文档目的

本文档基于以下三类材料，给出后续论文方向建议：

- 方法草稿：
  - `files/多智能体.md`
  - `files/方法流程_v2.md`
- 当前 authoritative 结果：
  - `reports/summary/20260506T140637Z-pilot100-xiaomimimo-mimo-v2.5-faithful.md`
  - `reports/summary/20260506T140637Z-pilot100-faithful-result-review-and-next-plan.md`
- 统一矩阵口径：
  - `src/experiment_core/faithful_matrix.py`
  - `src/experiment_core/faithful_acceptance.py`

目标不是再重复解释实验结果，而是回答下面 4 个问题：

1. 当前方法草稿与实际结果之间是否匹配。
2. 现在最适合写成什么类型的论文。
3. 应该把哪些结果作为主线，哪些作为对照或诊断。
4. 后续最值得投入的论文化动作是什么。

## 2. 先给结论

### 2.1 当前最自洽的论文方向

当前最适合的方向不是把论文写成：

- “某一个单一新方法在所有条件下都最优”

而是写成：

**一个围绕有限预算下多智能体通信的统一框架与设计原则论文。**

更准确地说，是：

**在固定预算下，多智能体系统的收益主要来自高价值分歧的选择性通信与最小化局部验证，而不是默认全量 debate。**

这和 `files/多智能体.md` 里最初提出的 SPARC 主张是一致的，但需要把它从“一个单点方法一定最强”的叙事，调整为“一个统一设计框架 + 多个实证组件共同支持”的叙事。

### 2.2 当前不建议的论文方向

当前不建议直接把论文写成：

- `SPARC v1` 单方法击败全部 baseline
- `CUE` 是当前项目的唯一主线
- “所有通信方法都在有限预算下同样成立”

原因很简单：

- `pilot100` 已经证明项目里有足够多的正结果；
- 但这些正结果分散在不同 family 中；
- 当前并没有一个单一实现同时在 same-context 和 split-context 两条线都形成统治性优势。

如果此时强行把论文收束成“一个唯一最强方法”，叙事会和数据之间产生张力。

## 3. 方法草稿与结果之间的匹配程度

## 3.1 `files/多智能体.md` 的核心主张

这份草稿最核心的观点其实有 3 条：

1. 在有限预算下，不应默认全量 debate。
2. 应该先独立解题，再按分歧触发通信。
3. 最终不应依赖简单 majority vote，而应做局部审计。

这三条主张从今天的 `pilot100` 结果看，都**没有被推翻**。

相反，它们在不同 family 里分别得到了支撑：

- “不应默认全量 debate”
  得到 `budget_comm`、`selective_comm`、`sid_lite`、`sparc` 的共同支持。
- “应该按分歧触发通信”
  得到 `trigger_early_exit_v1`、`voc_trigger_v2`、`cue_v1`、`sparc_v1` 的支持。
- “局部验证有意义”
  得到 `sparc` auditing 线和 `comm_necessary` split-context 线的间接支持。

所以从大方向上说，`多智能体.md` 的方法观是成立的。

## 3.2 `files/方法流程_v2.md` 的核心主张

这份文档把最初的三段式设计扩成了八阶段正式流程：

1. 协议冻结
2. 独立求解
3. 触发器
4. 分歧对象抽取
5. 预算感知路由
6. 选择性通信
7. 局部审计
8. 最终聚合

从研究方法论上看，这个拆分是对的。

但从当前结果看，有两个地方需要降级处理：

### 第一，分歧对象与预算路由是“框架主张”，不是所有 family 都已完整实现

现在项目里：

- `budget_comm` 把预算路由做得最像样；
- `selective_comm` 把 trigger 做得最完整；
- `sparc` 把内容模式与局部审计做得最完整；
- `comm_necessary` 把 split-context 必要性做得最清楚；
- `cue` 则最接近“统一框架”表达。

换句话说，当前工程结果更像是：

**多个 family 分别验证了八阶段框架里的不同关键环节。**

而不是：

**一个端到端单模型系统已经在所有阶段都达到了最强。**

### 第二，现阶段最强证据不是来自单一 family，而是来自组件组合视角

这也是为什么论文不应直接写成“SPARC v1 单方法已全面胜出”。

目前更自洽的表达是：

- `多智能体.md` 和 `方法流程_v2.md` 提出的，是一个统一框架假设；
- 当前 `pilot100` 已经证明其中多个关键模块是有效的；
- 论文贡献在于提出这个统一视角，并用多个受控实验说明各模块在不同条件下的作用边界。

## 4. 当前结果支持什么，不支持什么

## 4.1 当前结果明确支持的命题

### 命题 A：Same-context 下，有限预算通信是可以成立的

最强证据来自：

- `budget_comm / dala_lite_same_context_v1`
- `selective_comm / trigger_early_exit_v1`
- `selective_comm / trigger_voc_v2*`
- `sparc / sparc_v1_smoke`

这些结果说明：

- 不需要 always-on full communication 才能保持性能；
- selective / budget-aware 设计可以在较低通信成本下接近 full communication；
- 这与 `多智能体.md` 中“不是每个 agent 都需要开口”的命题高度一致。

### 命题 B：Split-context 下，通信不仅有用，而且是必要的

最强证据来自：

- `comm_necessary / full_packet_exchange`
- `budget_comm / dala_lite_split_context_v1`

尤其 `comm_necessary` 的结果非常关键：

- `faithful_score = 0.6100`
- `delta_vs_best_no_comm = +0.2100`
- `delta_vs_full_context = -0.0700`
- `stage_ceiling_gap = 0.0000`

这说明：

- 信息切分造成的能力损失是真实存在的；
- 通信可以恢复其中大部分损失；
- 但即使 full packet exchange，也尚未完全追回 full-context 上界。

这正好支撑论文里最强的因果段落。

### 命题 C：局部验证比“盲目全聊”更值得研究

这条命题的当前证据不是单一分数，而是结构性信号：

- `sparc_v1` 能在较低通信 token 下复现 `always_communicate` 的表现；
- `content_ablation` 明确显示，错误的通信内容会显著伤害效果；
- `auditing_ablation` 和 `aggregation_auditing_ablation` 表明审计层的设计会改变最终性能；
- `trigger_early_exit` 与 `voc_trigger_v2` 的成功也说明“何时说话”比“默认全说”更关键。

### 命题 D：多数收益来自高价值分歧处理，而不是随便增加 agent 数和轮数

这条命题来自跨 family 合读：

- `multi_agent` 说明 debate 可以比 matched vote 更强；
- 但 `same-context` 主结果又说明，不是所有 debate 都值得做满；
- `free_mad_lite` 说明更激进的轨迹机制可以提升准确率，但未必带来预算优势；
- `sid_lite` 说明单靠 early-exit 和压缩并不能自动形成最强方法。

所以更合理的归纳不是“更多 debate 更强”，而是：

**高价值分歧的识别与处理方式，决定了通信是否值得。**

## 4.2 当前结果不支持的命题

### 命题 X：单一统一实现已经统治所有场景

当前没有任何一个 family 同时做到：

- same-context 最强
- split-context 最强
- token frontier 最优
- stage ceiling 仍有明显优势

所以不要把论文写成“一个统一实现已经压过所有其它思路”。

### 命题 Y：`CUE` 已经是当前最强主线

`cue_v1` 当前的结果是：

- `faithful_score = 0.5840`
- `delta_vs_best_no_comm = +0.0300`
- `delta_vs_full_comm = -0.0500`
- `engineering_noise_gap = -0.0760`

这说明：

- `CUE` 仍然有概念价值；
- 但从 empirical headline 角度，它现在不应该被包装成最强主方法。

### 命题 Z：所有 accepted family 都值得写进 headline

虽然当前 `pilot100` 的 acceptance 计数是：

- `accepted_same_context = 17`
- `accepted_split_context = 3`
- `negative_control_family = 0`

但这只是 faithful gate 口径。

它不能直接等价于：

- 所有方法都 equally strong
- 所有方法都 equally novel
- 所有方法都 equally worth emphasizing

## 5. 现在最适合的论文类型

## 5.1 推荐：框架型方法论文

当前最自洽的论文类型是：

**一个围绕“预算约束下的选择性通信与局部验证”的框架型方法论文。**

核心写法应是：

1. 提出统一问题设定。
2. 提出统一设计框架。
3. 指出框架的 4 个关键设计轴：
   - 何时通信
   - 传什么
   - 如何路由
   - 如何验证
4. 用多个 family 的结果证明：
   - 每个轴都重要；
   - 在 same-context 与 split-context 下最重要的轴不同。

这会比写成“一个单点算法 paper”更匹配当前数据。

## 5.2 备选：设计空间论文

如果你不想再额外工程化出一个更统一的 `SPARC_paper_v1`，还有一个次优但也自洽的路线：

**把论文写成多智能体预算通信的 design-space paper。**

这种写法强调：

- 我们系统研究了多智能体通信在有限预算下的关键设计轴；
- 不是所有通信都值得；
- 不是所有 split-context 恢复都容易；
- 局部验证和内容选择是核心。

这种写法弱化“单方法最优”，强化“研究结论与机制洞察”。

### 但从你当前目标看，我更推荐第一种

因为你显然更希望论文有一个清晰的方法名和统一主张。

因此，最佳路径是：

- 保留 `SPARC` 作为统一方法名或统一框架名；
- 但在论文里明确：
  `SPARC` 是一个 framework，不是“所有实验都用同一个代码路径跑出来的唯一实现”。

## 6. 推荐的论文主张

我建议你把主张压缩成下面这 3 条。

### 主张 1

**在 same-context 下，多智能体收益的关键不是默认 debate，而是预算约束下的选择性通信。**

证据：

- `dala_lite_same_context_v1`
- `trigger_early_exit_v1`
- `voc_trigger_v2`
- `sparc_v1_smoke`

### 主张 2

**在 split-context 下，通信是必要条件，但真正困难的是高质量消息设计而不是单纯触发。**

证据：

- `full_packet_exchange` 显著优于 `split_no_comm`
- 但仍落后于 `full_context_single`
- `dala_lite_split_context_v1` 说明预算化通信成立，但会牺牲部分上界

### 主张 3

**多智能体系统的主要收益来自高价值分歧处理与局部验证，而不是简单增加轮数、agent 数或全量消息。**

证据：

- `multi_agent` 说明 debate 有价值，但不是条件充分
- `content_ablation` 说明错误内容会伤害结果
- `auditing_ablation` 和 `sparc_v1` 说明局部验证层值得建模

## 7. 推荐的论文结构

## 7.1 Introduction

引言不要写成：

- “多智能体很强，所以我们也来做一个多智能体方法”

应该写成：

- 多智能体推理常常依赖昂贵的全量 debate；
- 在固定预算下，这种做法并不资源理性；
- 我们关心的是：哪些分歧值得通信，哪些信息值得被传，哪些冲突值得被验证。

## 7.2 Problem Setup

建议正式定义：

- 固定总预算 `B`
- 总成本由：
  - solve cost
  - route/trigger cost
  - communication cost
  - audit cost
  组成

同时明确区分：

- `same_context`
- `split_context`

不要把两类 setting 混在一个总平均故事里。

## 7.3 Method

建议用 `SPARC` 作为总方法名，但写法上要清楚它是四段式或四组件式框架：

1. Independent Solve
2. Triggered Communication
3. Selective Passing / Routing
4. Audited Resolution

然后解释：

- `selective_comm` 系列主要体现了第 2 段；
- `budget_comm` 主要体现了第 3 段；
- `sparc` 主要体现了第 4 段；
- `comm_necessary` 则用 split-context setting 验证通信必要性。

这会让现有工程结果与方法章节自洽。

## 7.4 Experiments

建议拆成两条主表：

### 表 A：Same-context 主表

主方法建议放：

- `dala_lite_same_context_v1`
- `trigger_early_exit_v1`
- `trigger_voc_v2`
- `sparc_v1_smoke`

对照组放：

- `single_agent`
- `multi_agent`
- `free_mad_lite`
- `sid_lite`
- `cue_v1`

### 表 B：Split-context 主表

主方法建议放：

- `full_packet_exchange`
- `dala_lite_split_context_v1`

对照组放：

- `split_no_comm_mv3`
- `full_context_single`

## 7.5 Mechanism Analysis

这里把最有解释力的 supporting experiments 放进去：

- `trigger_early_exit_v1`
  解释“什么时候说话”
- `trigger_voc_v2`
  解释“黑盒 proxy 是否有效”
- `content_ablation_v1`
  解释“传什么”
- `auditing_ablation_v1`
  解释“如何验证”
- `debate_vs_vote_controlled`
  解释“debate 是否真的带来额外收益”

## 7.6 Limitations

这一节建议主动承认：

1. 当前没有单一实现同时统治 same-context 和 split-context。
2. `CUE` 代表的统一框架方向在当前 empirical frontier 上仍偏弱。
3. 本文更像是：
   - 一个统一框架主张
   - 加上多个受控 family 验证其关键设计轴

这反而会让论文更可信。

## 8. 各 family 在论文中的推荐角色

## 8.1 最适合进 headline 主表的 family

### `budget_comm / dala_lite_same_context_v1`

理由：

- same-context 下最干净的预算通信正结果之一；
- 对无通信控制有明显正增益；
- 与 full-comm 非常接近；
- 通信 token 大幅下降。

### `selective_comm / trigger_early_exit_v1`

理由：

- 几乎不损失性能；
- 通信成本极低；
- 很适合支撑“不是每个样本都值得说话”的命题。

### `selective_comm / trigger_voc_v2`

理由：

- 在 `pilot100` 上比 `smoke20` 更稳；
- 比自己的强无通信控制更强；
- 与 full-comm 几乎持平。

### `sparc / sparc_v1_smoke`

理由：

- 它最像你方法草稿中真正想表达的“内容选择 + 局部审计”路线；
- 虽然不是当前数值最强，但最贴近概念主线。

### `comm_necessary / full_packet_exchange`

理由：

- split-context 必须有它；
- 这是你论文里最强的“通信必要性”证据。

### `budget_comm / dala_lite_split_context_v1`

理由：

- 它让 split-context 叙事不至于只剩一个“通信必要”的 yes/no 命题；
- 它把预算化问题也带进了 split-context。

## 8.2 更适合作为 supporting evidence 的 family

### `multi_agent`

理由：

- 它很强，但更像 debate 机制对照；
- 不是你的统一通信预算主线。

### `free_mad_lite`

理由：

- 它能提升准确率；
- 但它不是预算型胜利，更像高性能对照。

### `sparc` 的两条 auditing ablation

理由：

- 非常适合解释“审计层是有用的”；
- 但不适合作为主方法 headline。

## 8.3 更适合作为诊断或负面边界的 family

### `cue_v1`

理由：

- 概念正确；
- empirical 上当前不够强；
- 更适合当作统一框架路线的边界情况。

### `sid_lite_v1`

理由：

- 很省通信；
- 但准确率损失较明显；
- 更像“极限压缩路线的代价示例”。

### `sparc / content_ablation_v1`

理由：

- 最适合作为“错误内容设计会伤害结果”的诊断表。

## 9. 现在最值得做的论文化动作

## 9.1 不建议再做的大动作

当前不建议：

- 再发起新一轮全量 `pilot100`
- 为了让某个 family 更强而改方法结构
- 试图把所有 family 都拉成 headline 级别

原因：

- 当前 authoritative 结果已经足够支撑论文方向判断；
- 再继续全量重跑的边际收益已经很低；
- 更大的风险是把 faithful 边界搞乱。

## 9.2 最值得做的 5 个动作

### 动作 1：给 `faithful_matrix` 增加结果分级

在 accepted / negative 之外，再加：

- `headline_positive`
- `qualified_positive`
- `diagnostic_control`
- `negative_or_weak_signal`

这样可以把当前结果的研究层含义显式化。

### 动作 2：补 bootstrap CI

至少给以下量补 CI：

- `faithful_score`
- `delta_vs_best_no_comm`
- `delta_vs_full_comm`

这样可以避免把 `1-3pp` 级别的小差值讲得过重。

### 动作 3：补预算 frontier 图

same-context 和 split-context 分开画：

- x 轴：`communication_tokens_mean` 或 `total_tokens_mean`
- y 轴：`faithful_score`

这会比只列表格更能支撑“预算下的通信价值”主张。

### 动作 4：补 claim-oriented table

不要再按 family 整体讲，而是按论文命题整理：

- 表 1：same-context budget-preserving methods
- 表 2：split-context communication-necessary methods
- 表 3：local auditing / content ablations
- 表 4：debate vs vote / trajectory / early-exit controls

### 动作 5：把方法章节写成“框架 + 实例化”而不是“唯一代码路径”

这是最关键的一条。

推荐写法：

- `SPARC` 是统一框架；
- 当前项目中的多个 family 实现了其中不同关键模块；
- `sparc_v1` 是最贴近框架完整表达的一个实例；
- 其余 family 是关键设计轴的受控实例与 supporting evidence。

这样写，才能和当前数据完全对齐。

## 10. 论文标题建议

基于当前结果，我建议标题不要写得太像“一个统一实现已经横扫所有 baseline”，而应该更贴近你的真实证据结构。

## 10.1 最推荐

### 方案 A

**Not Every Agent Needs to Speak: Budget-Aware Selective Communication for Multi-Agent Reasoning**

优点：

- 直接表达 same-context 主线；
- 和 `trigger_early_exit`、`voc_v2`、`dala_lite_same_context` 的结果最贴合；
- 标题本身就有很强记忆点。

### 方案 B

**SPARC: Selective Passing and Audited Routing under Communication Budget**

优点：

- 保留你的统一框架命名；
- 更适合方法论文风格；
- 后续好扩展。

但如果用这个标题，正文里必须明确：

- `SPARC` 是 framework；
- 不是“本仓库所有 family 都是同一算法代码路径的不同开关”。

## 10.2 次推荐

**Budgeted Communication and Local Verification in Multi-Agent LLM Reasoning**

这比强调某个单一实现更稳，但品牌感弱一些。

## 11. 最终建议

如果要一句话总结现在最合理的论文路径，我建议是：

**把论文写成一个“预算约束下的选择性通信与局部验证框架”论文，而不是把它写成某一个单一实现已经全面统治的算法论文。**

更具体地说：

- `same-context` 用 `dala_lite_same_context_v1`、`trigger_early_exit_v1`、`voc_trigger_v2`、`sparc_v1_smoke` 支撑预算保真。
- `split-context` 用 `full_packet_exchange` 和 `dala_lite_split_context_v1` 支撑通信必要性与预算化恢复。
- `multi_agent`、`free_mad_lite`、`sid_lite`、`cue` 与 `sparc` ablation 负责填充机制解释和边界情况。

这条路线最符合：

- 你在 `files/多智能体.md` 里提出的初始问题意识；
- `files/方法流程_v2.md` 里搭出来的统一框架；
- 以及当前 `pilot100` authoritative 结果真正支持的结论边界。
