# sparc

`sparc` 用于运行 SPARC 主实验、内容消融和局部审计消融。

## 入口

- CLI：`sparc_cli`
- 配置：`configs/sparc/`
- 运行目录：`runs/sparc/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/sparc/`

## 常用命令

```powershell
uv run sparc_cli inspect-experiment --experiment configs/sparc/experiments/end_to_end_main.toml
uv run sparc_cli run --experiment configs/sparc/experiments/content_ablation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run sparc_cli render-report --run-dir runs/sparc/content_ablation/smoke20/<run_id>
```

## 维护约定

- 路径、experiment 名和报告入口都统一使用规范化后的 `snake_case`。
- 聚合、局部审计和消息内容消融都通过 experiment / protocol 配置切换。
- 共享的解析与 provider 恢复逻辑走 `experiment_core`。
