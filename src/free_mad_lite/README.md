# free_mad_lite

`free_mad_lite` 用于运行 Free-MAD-lite 机制验证实验。

## 入口

- CLI：`free_mad_lite_cli`
- 配置：`configs/free_mad_lite/`
- 运行目录：`runs/free_mad_lite/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/free_mad_lite/`

## 常用命令

```powershell
uv run free_mad_lite_cli inspect-experiment --experiment configs/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml
uv run free_mad_lite_cli run --experiment configs/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run free_mad_lite_cli render-report --run-dir runs/free_mad_lite/free_mad_lite_mechanism_validation/smoke20/<run_id>
```

## 维护约定

- Anti-conformity、trajectory judging 和 fallback 规则只在这一家族内部维护。
- 公共 provider、限流和结构化输出恢复逻辑走共享层。
- 运行与报告路径遵循统一 workspace 规范。
