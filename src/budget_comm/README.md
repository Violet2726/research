# budget_comm 实验说明

## 1. 模块定位

`budget_comm` 用于运行 DALA-lite 风格的预算约束通信实验：

- same-context smoke20
- split-context smoke20
- 固定 5 个方法的严格配对比较
- 预算冻结、消息分层、knapsack 选择与中文报告

该包依赖共享层 `experiment_core`，不依赖其他实验包。

## 2. 目录结构

- `config.py`
  负责 experiment、protocol、auction policy 与 context view 配置加载。
- `dataset_views.py`
  负责 same-context / split-context 视图生成。
- `logic.py`
  负责 packet 投影、value density、tier 分配、knapsack 与 gate 诊断。
- `prompting.py`
  负责 Stage A solver 与 Stage B belief update 提示词。
- `runner.py`
  主执行链路，负责预算校准、共享 Stage A、多方法 Stage B 和产物落盘。
- `reporting.py`
  负责运行摘要与中文正式报告。
- `validation.py`
  负责预算、tier、paired design 与 split-context 约束校验。
- `cli.py`
  命令行入口，对外暴露 `budget-cli`。

## 3. 常用命令

查看实验配置：

```bash
uv run budget-cli inspect-experiment --experiment configs/budget_comm/experiments/dala_lite_same_context_v1.toml
```

运行 same-context smoke20：

```bash
uv run budget-cli run \
  --experiment configs/budget_comm/experiments/dala_lite_same_context_v1.toml \
  --phase smoke20 \
  --backbone dashscope/qwen-turbo-1101
```

运行 split-context smoke20：

```bash
uv run budget-cli run \
  --experiment configs/budget_comm/experiments/dala_lite_split_context_v1.toml \
  --phase smoke20 \
  --backbone dashscope/qwen-turbo-1101
```

校验运行结果：

```bash
uv run budget-cli validate-run --run-dir runs/budget_comm/<experiment>/<phase>/<run_id>
```

重新生成报告：

```bash
uv run budget-cli report-run --run-dir runs/budget_comm/<experiment>/<phase>/<run_id>
```
