# comm_necessary

`comm_necessary` 用于运行 HotpotQA split-context 通信必要性实验。

## 入口

- CLI：`comm_necessary_cli`
- 配置：`configs/comm_necessary/`
- 运行目录：`runs/comm_necessary/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/comm_necessary/`

## 常用命令

```powershell
uv run comm_necessary_cli inspect-experiment --experiment configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml
uv run comm_necessary_cli run --experiment configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run comm_necessary_cli report-run --run-dir runs/comm_necessary/hotpotqa_split_context_communication_necessity/smoke20/<run_id>
```

## 维护约定

- full-context 与 split-context 视图都通过共享数据读取和本家族视图层生成。
- supporting facts、答案聚合和通信包恢复逻辑保持在这一家族内聚。
- 结构化输出失败时先走共享恢复，再做 HotpotQA 专属兜底。
