# MADJudge

基于 Beta-Binomial 混合模型和 KS 检验的自适应多智能体辩论框架，用于 LLM-as-a-Judge 场景。

论文：*Multi-Agent Debate for LLM Judges with Adaptive Stability Detection* (arXiv:2510.12697)

## 核心机制

### Beta-Binomial 混合模型

使用时变 Beta-Binomial 混合模型跟踪 judges 共识动态：

- **正确分布** Beta(α_correct, β_correct)：建模正确 agent 的投票行为
- **随机分布** Beta(α_random, β_random)：建模随机猜测的 agent
- **混合权重** π_t：随时间变化的正确 agent 比例

通过 EM 算法（L-BFGS-B 优化）估计参数，计算后验概率 P(agent i correct | votes)。

### KS 检验自适应停止

使用 Kolmogorov-Smirnov 检验检测共识稳定性：

- D_t = sup|F_t(θ) - F_{t-1}(θ)|
- 当 D_t < threshold（默认 0.05）连续 N 轮（默认 2 轮）时停止辩论
- 跨样本聚合观测值，确保统计检验的有效性

### 多数投票聚合

最终答案使用多数投票（Majority Vote / SoM）聚合，与论文一致。

## 配置

- `configs/families/madjudge/experiments/madjudge_main.toml` — 主实验配置
- `configs/families/madjudge/protocols/default.toml` — 协议参数（7 agents, 10 rounds, temp 1.0）
- `configs/families/madjudge/rosters/homogeneous_7agent.toml` — 7 agent 同构阵容

## CLI

```bash
# 运行实验
uv run research_cli family madjudge run --experiment configs/families/madjudge/experiments/madjudge_main.toml --phase count20

# 查看实验配置
uv run research_cli family madjudge inspect-experiment --experiment configs/families/madjudge/experiments/madjudge_main.toml
```

## 算法流程

1. **初始轮次**：7 个 agent 独立生成答案和推理
2. **辩论轮次**：agent 看到其他 agent 的答案和推理，更新自己的判断
3. **稳定性检测**：每轮结束后，跨样本聚合投票，计算 Beta-Binomial 混合模型参数，执行 KS 检验
4. **自适应停止**：当 KS 统计量连续 2 轮低于阈值（0.05）时停止辩论
5. **最终聚合**：使用多数投票生成最终答案
