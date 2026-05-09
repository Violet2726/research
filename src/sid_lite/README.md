# sid_lite

`sid_lite` 用于运行 SID-lite 机制验证实验，重点关注 early-exit、压缩通信和 belief update 的协同效果。

## 入口

- CLI：`sid_lite_cli`
- 配置：`configs/sid_lite/`
- 运行目录：`runs/sid_lite/<experiment>/<phase>/<run_id>/`
- 报告目录：`reports/sid_lite/`

## 常用命令

```powershell
uv run sid_lite_cli inspect-experiment --experiment configs/sid_lite/experiments/sid_lite_mechanism_validation.toml
uv run sid_lite_cli run --experiment configs/sid_lite/experiments/sid_lite_mechanism_validation.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run sid_lite_cli report-run --run-dir runs/sid_lite/sid_lite_mechanism_validation/smoke20/<run_id>
```

## 维护约定

- SID-lite 的方法列表和协议参数全部通过配置声明。
- Solver / belief update 的结构化恢复逻辑走共享层。
- 报告默认围绕 equal-budget 比较与退出诊断展开。
