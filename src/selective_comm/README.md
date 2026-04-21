# 选择性通信实验说明

## 1. 模块定位

`selective_comm` 用于运行 trigger / early-exit 实验。

这条实验线的核心思想是：

- 先共享执行一份 `Stage A` 初始求解。
- 仅在策略判断“值得通信”时，继续执行共享的 `Stage B` debate。
- 用多条 trigger 策略在同一批共享前缀上做比较，评估准确率、通信开销与 oracle 对齐程度。

该包依赖共享层 `experiment_core`，不依赖 `single_agent_baselines` 或 `multi_agent_baselines`。

## 2. 目录结构

- `config.py`
  负责 experiment、protocol、policy 与 control catalog 的加载。
- `prompting.py`
  负责 Stage A 与 Stage B 的提示词构造。
- `runner.py`
  主执行逻辑，负责共享前缀执行、trigger 决策、控制方法运行和多类产物写出。
- `reporting.py`
  负责 trigger 报告、失败案例、共享前缀节省说明和推荐默认策略。
- `validation.py`
  负责共享哈希、一致性、early-exit token 等校验。
- `cli.py`
  命令行入口，对外暴露 `selective-cli`。

## 3. 配置入口

选择性通信实验配置位于：

- `configs/selective_comm/experiments/`
- `configs/selective_comm/protocols/`
- `configs/selective_comm/policies/`

共享 benchmark / provider / model catalog 位于：

- `configs/shared/benchmarks/`
- `configs/shared/benchmarks/splits/`
- `configs/shared/providers/`
- `configs/shared/model_catalog.toml`

当前主实验文件：

- `configs/selective_comm/experiments/trigger_early_exit_v1.toml`

## 4. 常用命令

查看实验配置：

```bash
uv run selective-cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml
```

带 backbone 解析查看实验配置：

```bash
uv run selective-cli inspect-experiment \
  --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml \
  --backbone dashscope/qwen-turbo-1101
```

执行实验：

```bash
uv run selective-cli run \
  --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml \
  --phase smoke20 \
  --backbone dashscope/qwen-turbo-1101
```

查看运行摘要：

```bash
uv run selective-cli summarize-run --run-dir local/runs/selective_comm/<experiment>/<phase>/<run_id>
```

校验运行结果：

```bash
uv run selective-cli validate-run --run-dir local/runs/selective_comm/<experiment>/<phase>/<run_id>
```

重新生成 trigger 报告：

```bash
uv run selective-cli report-trigger --run-dir local/runs/selective_comm/<experiment>/<phase>/<run_id>
```

## 5. 输出产物

默认运行目录：

- `local/runs/selective_comm/<experiment>/<phase>/<run_id>/`

默认报告目录：

- `local/reports/selective_comm/`

一次完整运行通常包含：

- `manifest.json`
  记录 backbone、protocol、policies、controls 与 benchmark 列表。
- `stage_a_turns.jsonl`
  共享初始求解阶段的逐 turn 记录。
- `stage_b_turns.jsonl`
  共享 debate 阶段的逐 turn 记录。
- `control_turns.jsonl`
  独立控制方法的逐 turn 记录。
- `trigger_decisions.jsonl`
  每题每策略的触发决策记录。
- `policy_predictions.jsonl`
  所有策略与控制方法的题级预测。
- `policy_metrics.json`
  方法级 summary 指标。
- `policy_diagnostics.json`
  trigger 诊断、共享前缀节省和推荐默认策略。
- `oracle_trigger_eval.json`
  oracle 视角下的 beneficial communication 评估。
- `run_validation.json`
  校验结果。
- `progress.json`
  进度快照。
- `trigger_report.md`
  中文正式报告。

报告会同步发布到：

- `local/reports/selective_comm/`

## 6. 实验约定

- `Stage A` 与 `Stage B` 是共享前缀产物，同一题上的多条策略共用，不重复发网络请求。
- `mv_3` 在这里是共享 `Stage A` 投票得到的内部无通信基线。
- `always_communicate` 表示总是执行 `Stage B`，常用作 oracle 近似参考。
- trigger 决策支持 disagreement、confidence 和 hybrid 等类型，并支持 confidence 非法时 fail-open。
- 报告层会同时给出 accuracy、communication tokens、oracle precision / recall 以及失败案例。

## 7. 适用场景

适合用于：

- 研究“何时值得通信”而不是“如何通信”。
- 评估 early-exit 是否能在较小准确率损失下显著节省 token。
- 对 trigger 策略做共享前缀下的公平比较。

不适合用于：

- 纯单模型 baseline 对比。
- 标准多智能体 debate 协议本身的结构研究。

