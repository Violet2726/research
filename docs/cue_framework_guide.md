# CUE Framework Guide

## 定位

CUE 是一个黑盒、免训练的多智能体通信效用框架：

- 先独立求解
- 再从文本可观测信号估计通信 utility
- 只对高 utility 样本做定向通信
- 必要时用局部审计器做最小验证

## 关键对象

- `AgentPacket`
- `ConflictObject`
- `BeliefUpdate`
- `AuditVerdict`

## 运行命令

查看实验配置：

```bash
uv run cue_cli inspect-experiment --experiment configs/cue/experiments/cue_black_box_utility_main.toml
```

执行 `smoke20`：

```bash
uv run cue_cli run --experiment configs/cue/experiments/cue_black_box_utility_main.toml --phase smoke20
```

校验运行：

```bash
uv run cue_cli validate-run --run-dir runs/cue/<experiment>/<phase>/<run_id>
```

生成报告：

```bash
uv run cue_cli render-report --run-dir runs/cue/<experiment>/<phase>/<run_id>
```
