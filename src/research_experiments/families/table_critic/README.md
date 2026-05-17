# table_critic

`table_critic` 用于运行 Table-Critic 论文复现实验。

## 入口

- CLI：`research_cli family table_critic`
- 配置：`configs/families/table_critic/`
- 默认运行目录：`local/runs/table_critic/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/table_critic/`

## 常用命令

```powershell
uv run research_cli family table_critic inspect-experiment --experiment configs/families/table_critic/experiments/table_critic_main.toml
uv run research_cli family table_critic run --experiment configs/families/table_critic/experiments/table_critic_main.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family table_critic render-report --run-dir local/runs/table_critic/table_critic_main/count20/<run_id>
```

## 当前口径

- `table_critic_main` 是当前项目的正式表推理复现主线；它进入 `reproduction_matrix`，但不并入 `faithful_matrix`。
- 当前 canonical benchmark 固定为 `WikiTQ / TabFact`。
- 当前 canonical 方法固定为 `end_to_end_qa / few_shot_qa / chain_of_table / critic_cot / table_critic_paper`。
- `Binder` 与 `Dater` 不纳入 v1 canonical 复现，避免把程序执行与异质执行栈引入成新的主要变量。

## 论文逻辑对齐

- `chain_of_table` 负责提供论文式初答。
- `table_critic_paper` 在此基础上执行 `Judge -> Critic -> Refiner -> Curator`，并维护 self-evolving template tree。
- `critic_cot` 只保留单轮 generic critic-refiner，用作弱批判基线。
- 停止条件固定为：judge 通过、答案/关键推理稳定，或达到最大 refinement 轮数。

## 本机复现边界

- 当前只使用仓内主模型，不追求和论文绝对分数完全一致。
- 当前重点是复现 `critic-refiner + template tree` 的流程逻辑，而不是一次性搬运论文全部外围 baseline。
- 若 `count300` 只体现出微弱增益且 token 成本明显失衡，则应冻结为 supporting reproduction，而不是继续扩展任务面。
