# Pilot100 Faithful 结果复盘与后续计划

## 1. 文档目的

本文基于 `pilot100` 的 authoritative faithful matrix 运行结果，回答两个问题：

1. 当前 `pilot100` 结果是否符合预期。
2. 在不改变方法结构的前提下，后续最合理的研究推进路径是什么。

这里的“符合预期”分三层定义：

- 工程层：实验是否完整、健康、可复现。
- faithful gate 层：是否满足当前矩阵定义的 acceptance 标准。
- 研究层：结果是否足以支撑论文 headline，而不是仅仅“没有明显失败”。

## 2. 分析依据

本次判断主要基于以下产物：

- `runs/faithful_matrix_iterative/20260506T140637Z-pilot100-xiaomimimo-mimo-v2.5/state.json`
- `runs/faithful_matrix_iterative/20260506T140637Z-pilot100-xiaomimimo-mimo-v2.5/faithful_analysis.md`
- `runs/faithful_matrix_iterative/20260506T140637Z-pilot100-xiaomimimo-mimo-v2.5/acceptance_summary.md`
- `src/experiment_core/faithful_matrix.py`
- `src/experiment_core/faithful_acceptance.py`
- 对照基线：
  `runs/smoke20_matrix_iterative/20260505T084626Z-smoke20-mimo-v2.5/faithful_analysis.md`

## 3. 总体结论

### 3.1 工程结论

当前 `pilot100` 结果在工程层面是健康的：

- `20/20` 个语义唯一 experiment 完成。
- `validation_passed = true`，`review_passed = true`。
- 没有遗留 `pending / rerun-needed / failed`。
- 这意味着当前 `pilot100` 可以视为 authoritative 结果，而不是中间态。

### 3.2 Faithful Gate 结论

按当前 `faithful_acceptance.py` 的判定逻辑：

- `accepted_same_context = 17`
- `accepted_split_context = 3`
- `negative_control_family = 0`

也就是说，**在当前 faithful gate 定义下，全部 20 个 experiment 都通过了接受标准。**

### 3.3 研究结论

这里必须做一个关键区分：

- “全部 accepted” 不等于 “全部都值得作为论文主结论”。
- 当前 `pilot100` 的结果更准确的解读是：
  **矩阵已经收口，方法之间的差异可以开始当作研究信号解释；但只有一部分 family 同时具备稳定性、预算意义和 headline 价值。**

换句话说，当前结果**符合工程预期，也符合 faithful 评测预期，但只部分符合论文 headline 预期**。

## 4. 为什么说“全部 accepted”仍然不能直接等同于“全部都很好”

这是当前解读里最容易混淆的一点。

`faithful_matrix` 的 acceptance gate 主要检查的是：

- same-context：
  `primary faithful` 是否对 `best_no_comm_control` 满足 `-2pp` 非劣；
  若存在 `full_comm_reference`，是否在定义允许的 token basis 下满足节省约束。
- split-context：
  是否优于 `split_no_comm`；
  是否至少补回一半 `full_context` gap；
  若有 full-comm 参照，是否没有超出预算上界。

这个 gate 的作用是：

- 排除明显失败的方法。
- 避免把结构上不可比的成本硬算成同一种 penalty。
- 让 faithful 评测保持自洽。

但它**不是**一个“谁最值得写论文 headline”的排序器。

因此，后续必须额外区分 4 类状态：

- `工程健康`
- `faithful gate 通过`
- `论文主线正结果`
- `仅适合作为对照或诊断`

## 5. Pilot100 是否符合预期：分层判断

### 5.1 工程层：符合预期

这一层的答案是明确的：**符合预期**。

原因：

- 所有实验都跑完了。
- 没有新的系统性请求失败、schema 崩坏或 partial run。
- `comm_necessary` 的输出契约问题已经在本轮前被根治，`pilot100` 不再残留结构化失败。

所以现在讨论的重点已经不再是“实验跑没跑坏”，而是“结果代表什么”。

### 5.2 Faithful Gate 层：符合预期

这一层的答案也是：**符合预期**。

因为按当前冻结的规则：

- same-context 共有 `17` 条主线，全部 accepted。
- split-context 共有 `3` 条主线，全部 accepted。
- 没有被判成 `negative_control_family` 的 experiment。

这说明当前 gate 和方法实现之间没有出现逻辑断裂。

### 5.3 研究层：部分符合预期

这一层需要更严格地说：

- `split_context` 结果整体上符合预期，而且很清楚。
- `same_context` 结果不差，但并没有收敛到“单一方法显著统治”的状态。
- 因此，当前结果更支持“形成有层次的论文叙事”，而不是“宣布某一个新方法在所有条件下都是最优”。

这也是本轮最重要的科学判断。

## 6. Cross-Family 深度分析

## 6.1 Same-Context：结论不是“谁都不行”，而是“不同方法回答的是不同问题”

当前 same-context 里最重要的事实不是简单看分数高低，而是看下面三件事是否同时成立：

- 相对强无通信基线是否守住或略优。
- 相对 full communication 是否没有明显掉点。
- 是否真的带来预算上的意义，而不是只是换了一种更贵的实现。

从这个角度看，same-context 方法可以分成 4 档。

### A. 稳定的预算正结果

这类方法最适合支撑“有限预算下能否维持或接近 full communication 能力”。

#### 1. `budget_comm / dala_lite_same_context_v1`

- `faithful_score = 0.7100`
- `delta_vs_best_no_comm = +0.1133`
- `delta_vs_full_comm = -0.0133`
- `communication_token_ratio_vs_full_comm = 0.2007`
- `stage_ceiling_gap = 0.0100`

解读：

- 这是很干净的正结果。
- 它显著优于本 family 的无通信控制。
- 与 `all_to_all_full` 只差 `1.33pp`。
- 通信 token 只用了 full-comm 的约 `20%`。

这组信号说明：**DALA-lite 在 same-context 下已经达到了“准确率基本守住、通信成本显著降低”的强效率结论。**

#### 2. `selective_comm / trigger_early_exit_v1`

- `faithful_score = 0.8400`
- `delta_vs_best_no_comm = -0.0167`
- `delta_vs_full_comm = 0.0000`
- `communication_token_ratio_vs_full_comm = 0.1973`
- `stage_ceiling_gap = 0.0167`

解读：

- 它比最强无通信控制低 `1.67pp`，但仍在 `-2pp` gate 内。
- 与 `always_communicate` 完全打平。
- 通信 token 只用 full-comm 的约 `20%`。

这说明它的意义不是“超过强无通信”，而是：
**用很低的通信负担，复现了 full communication 的表现。**

如果论文的问题是“预算内能不能保真”，它是一个非常强的 same-context 结果。

#### 3. `selective_comm / trigger_voc_v2*`

三条 `voc_v2` 主线在 `pilot100` 上几乎一致：

- `faithful_score = 0.7300`
- `delta_vs_best_no_comm = +0.0800` 或 `+0.1267`
- `delta_vs_full_comm = -0.0100`
- `communication_token_ratio_vs_full_comm = 0.5237`
- `stage_ceiling_gap = 0.0100`

解读：

- 和 `smoke20` 相比，它在 `pilot100` 上更稳，不再像之前那样像负结果。
- 它没有 `trigger_early_exit` 那么省通信，但比自己的强无通信控制更强。
- 与 full-comm 只差 `1pp`。

这说明：**在当前 faithful 口径下，`voc_v2` 不是失败方法，而是一个性能较稳、通信节省中等的 selective family 结果。**

它不一定是最省的，但已经不应再被视为当前版本下的“负结果方法”。

#### 4. `sparc / sparc_v1_smoke`

- `faithful_score = 0.6667`
- `delta_vs_best_no_comm = +0.0400`
- `delta_vs_full_comm = 0.0000`
- `communication_token_ratio_vs_full_comm = 0.3238`
- `stage_ceiling_gap = 0.0200`

解读：

- 相对 `mv_3` 有稳定正增益。
- 与 `always_communicate` 打平。
- 通信成本明显低于 always communication。

它不如 `trigger_early_exit` 那么强势，但作为“审计增强式 selective communication”的代表，它已经足够稳。

### B. 强结果，但更适合作为控制组或机制研究

#### 1. `multi_agent / debate_vs_vote_controlled`

- `faithful_score = 0.8350`
- `delta_vs_best_no_comm = +0.1500`
- `stage_ceiling_gap = 0.0000`

解读：

- 这是高质量正结果。
- 它很清楚地证明了：在当前 paired setup 下，debate 确实可以显著强于 matched vote 控制。

但它的研究角色更像是：

- “debate 是否真的带来额外收益”的对照研究
- 而不是“预算内统一通用方法论”的主线方法

#### 2. `multi_agent / vanilla_mad_clean_smoke`

- `faithful_score = 0.7667`
- `delta_vs_best_no_comm = +0.1000`

解读：

- 它也是稳定正结果。
- 但同样更像是多智能体 debate 机制研究的 supporting evidence。

#### 3. `free_mad_lite / free_mad_lite_v1`

- `faithful_score = 0.7867`
- `delta_vs_best_no_comm = +0.1600`
- `delta_vs_full_comm = +0.0333`
- `token_ratio_vs_full_comm = 1.2875`
- `communication_token_ratio_vs_full_comm = 1.0`

解读：

- 从准确率看，它很强，甚至略高于 `vanilla_mad_r1_final_vote`。
- 但从“有限预算”角度看，它不是一个节省通信的方案。
- 它更像是在固定结构内引入 trajectory judge 后的性能型改进。

因此，它适合做“准确率强化型控制方法”，不适合直接作为“预算型 headline 方法”。

### C. Accepted，但研究上偏边缘或偏弱

#### 1. `cue / cue_v1`

- `faithful_score = 0.5840`
- `delta_vs_best_no_comm = +0.0300`
- `delta_vs_full_comm = -0.0500`
- `communication_token_ratio_vs_full_comm = 0.2021`
- `engineering_noise_gap = -0.0760`
- `stage_ceiling_gap = 0.0180`

解读：

- 它过线了，但不是强正结果。
- 相对 `mv_3` 只有 `+3pp`。
- 相对 `always_communicate` 低 `5pp`。
- 相比 `smoke20` 还明显回落。

这说明目前的 `CUE` 有两个现实问题：

- 它不是坏方法，但当前实证强度不够。
- 它也不能再被自然地视为当前项目里最强的 same-context 主线。

更准确的定位应该是：
**概念上有价值、结构上清晰，但在当前 faithful 实现下还没有站到 empirical frontier。**

#### 2. `sid_lite / sid_lite_v1`

- `faithful_score = 0.6200`
- `delta_vs_best_no_comm = +0.0267`
- `delta_vs_full_comm = -0.0933`
- `communication_token_ratio_vs_full_comm = 0.1372`
- `stage_ceiling_gap = 0.0033`
- `engineering_noise_gap = -0.0633`

解读：

- 它很省通信，但准确率只比 `mv_3` 略高。
- 相对 full-comm 掉了 `9.33pp`。
- 更关键的是 `stage_ceiling_gap` 几乎为 `0`。

这说明：**在当前 faithful 结构下，它几乎已经把自己能回收的空间用完了。**

因此，它更适合作为“早退/压缩路线的边界对照”，而不是继续重投入打磨的主线。

#### 3. `sparc / content_ablation_v1`

- `faithful_score = 0.6233`
- `delta_vs_best_no_comm = -0.0033`
- `delta_vs_full_comm = -0.0967`
- `stage_ceiling_gap = 0.0167`

解读：

- 它 technically 过线，但只是 barely accepted。
- 更重要的是，它是 ablation，不是主方法。
- 这个结果更像在说明：**communication content 的选择非常关键，错误内容会明显伤害效果。**

因此它是很有价值的诊断结果，但不应被当成正主线。

### D. 作为 faithful 参考系的无通信家族

`single_agent` 三条 experiment 的主要作用是：

- 提供强无通信参考点
- 稳定 benchmark 难度锚点
- 帮助判断通信方法到底是在“增益”还是在“补偿”

其中需要注意的一点是：

- `main_baselines` 从 `smoke20` 的 `0.9000` 回落到 `pilot100` 的 `0.8300`

这不是坏事，反而说明：

- `pilot100` 把 `smoke20` 里的乐观偏差压掉了
- 当前比较更接近稳定结论

## 6.2 Split-Context：结果比 Same-Context 更清楚

split-context 这条线的故事目前比 same-context 更干净。

### 1. `comm_necessary / full_packet_exchange`

两条主线一致：

- `faithful_score = 0.6100`
- `delta_vs_best_no_comm = +0.2100`
- `delta_vs_full_context = -0.0700`
- `stage_ceiling_gap = 0.0000`
- `engineering_noise_gap = +0.0600`

解读：

- 相对 split no-comm 有非常明确的收益。
- 已经补回了大部分 full-context gap。
- `stage_ceiling_gap = 0` 很关键，这说明：
  **在当前 faithful 结构内，它已经基本达到了自己的上限。**

这组结果非常适合支撑论文里的强因果叙事：

- 在信息确实被切分时，通信不是“锦上添花”，而是“必要条件”。
- 但即便如此，通信质量仍未完全追平 full-context。

### 2. `budget_comm / dala_lite_split_context_v1`

- `faithful_score = 0.6950`
- `delta_vs_best_no_comm = +0.0850`
- `delta_vs_full_comm = -0.0550`
- `communication_token_ratio_vs_full_comm = 0.2118`
- `stage_ceiling_gap = 0.0300`

解读：

- 它的增益不如 `comm_necessary` 那么“因果强”，但它回答了另一个问题：
  **split-context 下，预算化通信是否还能成立。**
- 当前答案是成立的，但会比 full communication 低 `5.5pp`。

因此，这条线更适合被放在：

- “预算化 split communication”的 supporting claim
- 而不是 split-context 的唯一 headline

## 7. 与 Smoke20 对比：哪些趋势稳定，哪些被放大了

当前 `pilot100` 和 `smoke20` 有一个非常清楚的对照关系：

- acceptance 计数没有变：
  `17 same_context + 3 split_context + 0 negative`
- 但很多 `smoke20` 中看起来很高的分数，在 `pilot100` 上回落了。

这说明：

- `smoke20` 适合快速定位问题
- `pilot100` 才适合作为结论依据

### 7.1 稳定甚至增强的结果

- `dala_lite_same_context_v1`
  从 `0.6833` 升到 `0.7100`
- `voc_trigger_v2*`
  从 `0.7167` 升到 `0.7300`
- `comm_necessary`
  从 `0.5500` 升到 `0.6100`

这类结果说明：

- 它们不是小样本偶然正波动
- 更大样本下反而更稳

### 7.2 明显回落但仍可解释的结果

- `cue_v1`
  `0.6600 -> 0.5840`
- `sid_lite_v1`
  `0.6833 -> 0.6200`
- `sparc` 两条 auditing ablation
  `0.7167 -> 0.6667`
- `main_baselines`
  `0.9000 -> 0.8300`

这类结果说明：

- `smoke20` 的上限感并不可靠
- 如果一个方法在 `pilot100` 上回落，同时 `stage_ceiling_gap` 又接近 `0`，那更像是方法本体限制，而不是工程事故

## 8. 当前结果是否符合“有限预算中还能保持或提升模型解题能力”的研究目标

如果把问题问得非常严格，答案是：

### 8.1 对整体项目来说：符合

因为当前至少有多条主线清楚地证明了这一点：

- `dala_lite_same_context_v1`
- `trigger_early_exit_v1`
- `voc_trigger_v2*`
- `sparc_v1_smoke`
- `dala_lite_split_context_v1`
- `comm_necessary/full_packet_exchange`

这些结果共同说明：

- 在 faithful 约束下，预算化通信是可以成立的。
- 不是所有 gain 都必须依赖 full communication。

### 8.2 对“所有方法都要强”这个目标来说：不符合，也不应该再这样要求

这一点同样必须说清楚：

- 当前结果不支持“所有通信 family 都应该成为强正结果”。
- 更合理的判断是：
  **项目已经有足够多的正结果 family，可以支撑论文主张；其余 family 应当作为控制组、诊断组或负结果忠实保留。**

这比继续尝试把所有方法都磨平，更科学，也更符合 faithful 原则。

## 9. 后续详细计划

## 9.1 总体原则

后续计划建议严格遵守以下 4 条：

- 不再做新的全量 faithful 重跑，除非发现新的工程事故。
- 不再为了拉高分数而修改方法结构。
- 把后续工作重点从“继续跑”切到“继续分层解释”。
- 用论文叙事去选择主线，而不是试图让所有 family 同时成为主线。

## 9.2 Phase A：冻结 authoritative 结果

目标：
正式冻结当前 `pilot100` 为 authoritative 结果集。

建议动作：

- 冻结：
  `runs/faithful_matrix_iterative/20260506T140637Z-pilot100-xiaomimimo-mimo-v2.5`
- 冻结：
  `runs/smoke20_matrix_iterative/20260505T084626Z-smoke20-mimo-v2.5`
- 在 `faithful_matrix` 产物里增加：
  - `reference_state_path`
  - `acceptance_version`
  - `gate_parameters`

原因：

- 现在结果已经具备解释价值，最怕的是后续忘记“这次是按什么口径判的”。

## 9.3 Phase B：补一层“headline 价值分级”

目标：
让 `faithful_matrix` 不只告诉我们“过没过线”，还告诉我们“值不值得主打”。

建议新增 4 个标签：

- `headline_positive`
- `qualified_positive`
- `diagnostic_control`
- `negative_or_weak_signal`

建议当前先按下面的暂定分法执行：

- `headline_positive`
  - `budget_comm/dala_lite_same_context_v1`
  - `selective_comm/trigger_early_exit_v1`
  - `selective_comm/trigger_voc_v2*`
  - `sparc/sparc_v1_smoke`
  - `comm_necessary/hotpotqa_split_evidence_v1`
  - `comm_necessary/hotpotqa_split500_main`
  - `budget_comm/dala_lite_split_context_v1`

- `qualified_positive`
  - `multi_agent/debate_vs_vote_controlled`
  - `multi_agent/vanilla_mad_clean_smoke`
  - `free_mad_lite/free_mad_lite_v1`
  - `sparc/auditing_ablation_v1`
  - `sparc/aggregation_auditing_ablation_v1`

- `diagnostic_control`
  - `cue/cue_v1`
  - `sid_lite/sid_lite_v1`
  - `sparc/content_ablation_v1`
  - `single_agent/*`

这个分层会比单纯的 accepted / negative 更贴近研究含义。

## 9.4 Phase C：加强分析层，而不是再改方法

目标：
让论文里的差异解释更稳，避免把偶然波动讲成理论结论。

建议新增 4 类分析产物：

### 1. `bootstrap confidence interval`

对以下量加 bootstrap CI：

- `faithful_score`
- `delta_vs_best_no_comm`
- `delta_vs_full_comm`

原因：

- 现在很多差值虽然方向明确，但数值上仍是 `1-5pp` 量级。
- 加 CI 之后，哪些差异是稳定信号、哪些只是轻微漂移会更清楚。

### 2. `dataset-wise win/loss matrix`

按 dataset 输出：

- `win_vs_best_no_comm`
- `tie_vs_best_no_comm`
- `loss_vs_best_no_comm`
- `win_vs_full_comm`
- `tie_vs_full_comm`
- `loss_vs_full_comm`

原因：

- overall 分数会掩盖“有些方法是靠某一个数据集吃满收益”的事实。

### 3. `budget frontier figure`

为 same-context 和 split-context 分别画：

- x 轴：`communication_tokens_mean` 或 `total_tokens_mean`
- y 轴：`faithful_score`

原因：

- 当前很多结论其实是 frontier 结论，而不是谁的单点 accuracy 最大。

### 4. `stage_ceiling diagnostic table`

明确列出：

- `stage_ceiling_gap <= 0.01`
- `0.01 < stage_ceiling_gap <= 0.03`
- `stage_ceiling_gap > 0.03`

原因：

- 这会直接告诉我们哪些 family 还有 faithful 范围内的回收空间，哪些已经基本到头。

## 9.5 Phase D：论文主线收束

目标：
不再追求“所有方法都说一遍”，而是形成自洽的三层叙事。

建议的论文结构：

### 主线一：same-context 下的预算保真

优先使用：

- `dala_lite_same_context_v1`
- `trigger_early_exit_v1`
- `trigger_voc_v2`
- `sparc_v1_smoke`

这条线回答的问题是：
**在信息并未缺失的前提下，通信是否还能在有限预算中保持性能。**

### 主线二：split-context 下的通信必要性

优先使用：

- `hotpotqa_split_evidence_v1`
- `hotpotqa_split500_main`
- `dala_lite_split_context_v1`

这条线回答的问题是：
**当信息本身被切分时，通信不再只是预算优化，而是能力恢复的必要步骤。**

### 主线三：控制研究与负结果

放入：

- `multi_agent` 两条 debate 控制
- `free_mad_lite`
- `cue_v1`
- `sid_lite`
- `sparc` ablations

这条线回答的问题是：

- 哪些 gain 来自 debate 本身
- 哪些来自 judge 或 auditing
- 哪些方法在 faithful 约束下并没有成为强正结果

这会让论文更有说服力，因为它不是只展示赢家。

## 9.6 Phase E：哪些 family 还值得继续投入，哪些应该冻结

### 建议继续投入的 family

前提：只允许非结构优化和分析增强。

- `dala_lite_same_context_v1`
- `dala_lite_split_context_v1`
- `trigger_early_exit_v1`
- `trigger_voc_v2`
- `sparc_v1_smoke`

理由：

- 它们已经是正结果。
- 仍有一定 `stage_ceiling_gap`，说明还有少量 faithful 空间可回收。
- 它们和论文主问题贴得更近。

### 建议冻结为控制/配角的 family

- `multi_agent/debate_vs_vote_controlled`
- `multi_agent/vanilla_mad_clean_smoke`
- `free_mad_lite_v1`
- `comm_necessary/full_packet_exchange`

理由：

- 这些结果已经足够清楚。
- 再投入的边际收益主要是“更漂亮”，不是“更改变结论”。

### 建议冻结为诊断或弱信号的 family

- `cue_v1`
- `sid_lite_v1`
- `sparc/content_ablation_v1`

理由：

- 当前不是工程坏了，而是 empirical strength 不够。
- 即使继续 faithful 微调，预期收益也有限。

## 10. 最终判断

如果问题是：
“当前 `pilot100` 结果是否符合预期？”

那么最自洽的答案是：

- **工程上：符合预期。**
- **faithful gate 上：符合预期。**
- **论文主线选择上：部分符合预期。**

更具体地说：

- 项目已经不再缺“能讲的正结果”。
- 当前真正缺的是“把哪些结果该作为主张、哪些只该作为控制”讲清楚。
- 因此，下一步最重要的不是继续全量重跑，而是：
  **冻结当前结果，补足分析层分级，然后用这些结果收束论文叙事。**

## 11. 建议执行顺序

建议按下面顺序推进：

1. 冻结当前 `pilot100` 与 `smoke20` authoritative state。
2. 在 `faithful_matrix` 中加入 `headline_tier` 与 `acceptance_version`。
3. 补 bootstrap CI、dataset-wise win/loss、budget frontier。
4. 形成正式论文主表与附表草稿。
5. 仅对已经是正结果的 family 做最后一轮非结构 polish。
6. 不再对 `cue_v1`、`sid_lite`、`content_ablation_v1` 做新的 full rerun。

这条路线最符合当前证据，也最符合 faithful 原则。
