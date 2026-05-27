# configs/madjudge

MADJudge 自适应多智能体辩论框架配置目录。

## 目录组成

- `experiments/`：正式实验入口配置
- `protocols/`：辩论协议参数（agent 数量、轮次、温度、稳定性阈值）
- `rosters/`：agent 阵容配置

## 当前正式实验

- `madjudge_main`：7 数据集主实验

## 核心参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| agent_count | 7 | 辩论 agent 数量 |
| max_debate_rounds | 10 | 最大辩论轮次 |
| temperature | 1.0 | 采样温度 |
| ks_threshold | 0.05 | KS 检验阈值 |
| consecutive_stable_required | 2 | 连续稳定轮次要求 |

## 维护约定

- MADJudge 实现论文 "Multi-Agent Debate for LLM Judges with Adaptive Stability Detection"
- 使用 Beta-Binomial 混合模型和 KS 检验进行自适应停止
- 跨样本聚合观测值确保统计检验有效性
