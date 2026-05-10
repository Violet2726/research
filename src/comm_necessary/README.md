# comm_necessary

`comm_necessary` 用于运行 HotpotQA split-context 通信必要性实验。

## 入口

- CLI：`comm_necessary_cli`
- 配置：`configs/comm_necessary/`
- 默认运行目录：`local/runs/comm_necessary/<experiment>/<phase>/<run_id>/`
- 默认报告目录：`local/reports/comm_necessary/`

## 常用命令

```powershell
uv run comm_necessary_cli inspect-experiment --experiment configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml
uv run comm_necessary_cli run --experiment configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run comm_necessary_cli render-report --run-dir local/runs/comm_necessary/hotpotqa_split_context_communication_necessity/smoke20/<run_id>
```
