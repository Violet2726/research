# econ

`econ` 用于运行 ECON 低通信协调论文复现实验。

## 入口

- CLI：`research_cli family econ`
- 配置：`configs/families/econ/`
- 默认运行目录：`local/runs/econ/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/econ/`

## 常用命令

```powershell
uv run research_cli family econ inspect-experiment --experiment configs/families/econ/experiments/econ_same_context_main.toml
uv run research_cli family econ run --experiment configs/families/econ/experiments/econ_same_context_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family econ render-report --run-dir local/runs/econ/econ_same_context_main/count20/<run_id>
```

## 当前口径

- `econ_same_context_main` 是当前项目下一篇正式论文复现主线。
- 这条线直接服务 faithful 主研究问题，因此会进入 `faithful_matrix`，但首轮 evidence tier 固定为 `supporting`。
- 当前 canonical benchmark 固定为 `GSM8K / StrategyQA / HotpotQA`。
- 当前 canonical 方法固定为 `single_agent_cot / vote_mv3 / econ_full_comm_r1 / econ_bne_main`。

## 论文逻辑对齐

- `single_agent_cot` 提供单智能体强 CoT 基线。
- `vote_mv3` 提供三智能体无通信多数投票基线。
- `econ_full_comm_r1` 提供一轮显式共享理由的高通信参考线。
- `econ_bne_main` 复现“先独立求解，再按 belief 选择协调动作”的低通信协调主逻辑。
- 第一版不实现多轮自由辩论，也不引入额外 router、额外 verifier 或额外 majority vote。

## 当前实现边界

- 当前重点是复现 `belief-driven coordination + low communication + equilibrium action selection`。
- `equilibrium` 采用有限动作集上的项目内近似，而不是完整论文训练式求解器。
- 受控动作空间固定为 `keep_local / adopt_vote / query_best_peer / query_two_peers`。
- 如果 `count300` 只体现出微弱增益或明显成本失衡，这条线应冻结为 supporting reproduction，而不是继续扩展通信轮次。

