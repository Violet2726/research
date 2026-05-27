# CONSENSAGENT

基于触发机制的多智能体辩论框架，通过检测和缓解谄媚行为（Sycophancy）提升多智能体共识的效率与有效性。

论文：*Towards Efficient and Effective Consensus in Multi-Agent LLM Interactions Through Sycophancy Mitigation*

## 四阶段流程

1. **Phase 1 — 初始响应生成**：各 agent 独立生成答案、推理链和置信度分数
2. **Phase 2 — 多轮辩论**：agent 交换答案与推理，触发机制（t0 停滞 / t1 答案互换 / t2 复制型谄媚）检测并提前终止低效辩论
3. **Phase 3 — 提示优化**（论文核心，暂未复现）：基于辩论历史用微调 GPT-4o 优化 prompt
4. **Phase 4 — 团队答案生成**：置信度 × log(1+n_r) × (1+S_r) 加权聚合

## 触发机制

| 触发 | 条件 | 论文激活率 |
|------|------|-----------|
| t0 停滞 | 多数 agent 连续 N 轮保持相同答案 | 3–7% |
| t1 答案互换 | 多数 agent 在轮次间互换答案 | 15–40% |
| t2 复制型谄媚 | 多数 agent 转向多数答案且一致性 > 80% | — |

## 配置

- `configs/families/consensagent/experiments/consensagent_main.toml` — 主实验配置
- `configs/families/consensagent/protocols/default.toml` — 协议参数
- `configs/families/consensagent/rosters/homogeneous_3agent.toml` — 3 agent 同构阵容

## CLI

```bash
# 运行实验
uv run research_cli family consensagent run --experiment configs/families/consensagent/experiments/consensagent_main.toml --phase count20

# 查看实验配置
uv run research_cli family consensagent inspect-experiment --experiment configs/families/consensagent/experiments/consensagent_main.toml
```
