# 多智能体实验说明

## 1. 模块定位

`multi_agent_baselines` 用于运行 Vanilla MAD 风格的多智能体实验，并与等预算的单模型控制方法做配对比较。

这条实验线重点解决：

- 多 agent 初始独立求解与后续 debate 的完整执行。
- 与 `mv_*` / `sc_*` 等控制方法做同预算对照。
- 输出 Debate vs Vote 所需的题级记录、诊断指标和正式报告。

该包依赖共享层 `experiment_core`，不依赖 `single_agent_baselines` 或 `selective_comm`。

## 2. 目录结构

- `config.py`
  负责加载 experiment、protocol、roster 与 matched controls。
- `prompting.py`
  负责初始求解和 debate 轮的提示词构造。
- `runner.py`
  主执行逻辑，负责 setup 展开、turn 执行、投票聚合和运行产物落盘。
- `reporting.py`
  负责 Debate vs Vote 的配对分析、统计检验和 Markdown 报告。
- `validation.py`
  负责运行完整性与关键产物检查。
- `cli.py`
  命令行入口，对外暴露 `mad-cli`。

## 3. 配置入口

多智能体实验配置位于：

- `configs/multi_agent/experiments/`
- `configs/multi_agent/protocols/`
- `configs/multi_agent/rosters/`

共享 benchmark / provider / model catalog 位于：

- `configs/shared/benchmarks/`
- `configs/shared/benchmarks/splits/`
- `configs/shared/providers/`
- `configs/shared/model_catalog.toml`

当前常用实验文件包括：

- `configs/multi_agent/experiments/debate_vs_vote_controlled.toml`
- `configs/multi_agent/experiments/vanilla_mad_clean_smoke.toml`
- `configs/multi_agent/experiments/vanilla_mad_minimal.toml`

## 4. 常用命令

查看实验配置：

```bash
uv run mad-cli inspect-experiment --experiment configs/multi_agent/experiments/debate_vs_vote_controlled.toml
```

带 backbone 解析查看实验配置：

```bash
uv run mad-cli inspect-experiment \
  --experiment configs/multi_agent/experiments/debate_vs_vote_controlled.toml \
  --backbone dashscope/qwen-turbo-1101
```

执行实验：

```bash
uv run mad-cli run \
  --experiment configs/multi_agent/experiments/debate_vs_vote_controlled.toml \
  --phase smoke20 \
  --backbone dashscope/qwen-turbo-1101
```

查看运行摘要：

```bash
uv run mad-cli summarize-run --run-dir runs/multi_agent/<run_id>
```

校验运行结果：

```bash
uv run mad-cli validate-run --run-dir runs/multi_agent/<run_id>
```

生成 Debate vs Vote 报告：

```bash
uv run mad-cli report-debate-vs-vote --run-dir runs/multi_agent/<run_id>
```

## 5. 输出产物

默认运行目录：

- `runs/multi_agent/<run_id>/`

默认报告目录：

- `reports/multi_agent/`

一次完整运行通常包含：

- `manifest.json`
  记录 backbone、setup、protocol、roster 和 matched controls。
- `agent_turns.jsonl`
  记录每个 agent 每一轮的输入输出、usage 与解析状态。
- `debate_messages.jsonl`
  记录显式 peer message 的可见内容。
- `final_predictions.jsonl`
  记录题级最终投票结果与中间统计字段。
- `metrics.json`
  方法级 summary 指标。
- `cost_breakdown.json`
  初始轮、debate 轮和控制方法的 token 成本拆分。
- `debate_diagnostics.json`
  disagreement、flip、consensus 等诊断指标。
- `run_summary.json`
  运行摘要。
- `run_validation.json`
  校验结果。
- `progress.json`
  进度快照。

在执行 `report-debate-vs-vote` 后，还会生成：

- `paired_debate_vs_vote.json`
- `debate_vs_vote_report.md`

并发布到：

- `reports/multi_agent/`

## 6. 实验约定

- 每个样本内部严格保持 MAD 回合顺序，样本之间可以并发执行。
- `setup` 用于声明一组具体多智能体协议及其配套控制方法。
- 报告层默认把多智能体实验看作“配对设计”，重点比较 initial vote 与 debate vote 的差异。
- 当 phase 不是 `pilot100` 时，不输出 McNemar / bootstrap 统计检验，只保留描述性结果。

## 7. 适用场景

适合用于：

- 验证 debate 是否相对初始投票带来稳定收益。
- 分析多轮通信的 token / latency 开销。
- 生成多智能体 baseline 章节所需的题级和报告级产物。

不适合用于：

- 单模型方法公平性实验。
- 仅研究“是否触发通信”的 trigger / early-exit 机制。

