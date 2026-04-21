# 单智能体实验说明

## 1. 模块定位

`single_agent_baselines` 用于运行单模型、多方法的基线实验。

这条实验线的核心目标是：

- 在统一 benchmark 上比较 `cot`、`sc_*`、`mv_*` 等单智能体方法。
- 保持统一的 prompt 格式、缓存策略、解析兜底和评分逻辑。
- 输出可直接用于后续汇总、论文表格导出和运行校验的结构化产物。

该包依赖共享层 `experiment_core`，但不依赖其他实验包。

## 2. 目录结构

- `config.py`
  负责解析单智能体实验配置，以及 phase 级标签约束。
- `prompting.py`
  负责构造单次请求的 system / user messages。
- `runner.py`
  主执行入口，负责请求调度、缓存命中、结果解析、指标聚合和落盘。
- `reporting.py`
  负责运行摘要、预算公平性检查和论文表格导出。
- `validation.py`
  负责运行后的一致性与完整性检查。
- `cli.py`
  命令行入口，对外暴露 `single-agent-cli`。

## 3. 配置入口

单智能体实验配置位于：

- `configs/single_agent/experiments/`
- `configs/single_agent/methods/`

共享 benchmark / provider / model catalog 位于：

- `configs/shared/benchmarks/`
- `configs/shared/benchmarks/splits/`
- `configs/shared/providers/`
- `configs/shared/model_catalog.toml`

当前常用实验文件包括：

- `configs/single_agent/experiments/main-baselines.toml`
- `configs/single_agent/experiments/qwen2.5-math-pilot100.toml`
- `configs/single_agent/experiments/robustness.toml`

## 4. 常用命令

查看实验配置：

```bash
uv run single-agent-cli inspect-experiment --experiment configs/single_agent/experiments/main-baselines.toml
```

查看某个模型在各 phase 下的解析结果：

```bash
uv run single-agent-cli inspect-experiment \
  --experiment configs/single_agent/experiments/main-baselines.toml \
  --model dashscope/qwen-turbo-1101
```

生成冻结 split：

```bash
uv run single-agent-cli generate-splits
```

执行实验：

```bash
uv run single-agent-cli run \
  --experiment configs/single_agent/experiments/main-baselines.toml \
  --phase smoke20 \
  --model dashscope/qwen-turbo-1101
```

查看运行摘要：

```bash
uv run single-agent-cli summarize-run --run-dir local/runs/single_agent/<run_id>
```

导出论文表格：

```bash
uv run single-agent-cli export-paper-tables --run-dir local/runs/single_agent/<run_id>
```

检查预算公平性：

```bash
uv run single-agent-cli check-budget-fairness --run-dir local/runs/single_agent/<run_id>
```

校验运行结果：

```bash
uv run single-agent-cli validate-run --run-dir local/runs/single_agent/<run_id>
```

## 5. 输出产物

默认运行目录：

- `local/runs/single_agent/<run_id>/`

默认报告目录：

- `local/reports/single_agent/`

一次完整运行通常包含：

- `manifest.json`
  记录本次运行最终使用的配置。
- `raw_responses.jsonl`
  记录每次底层调用的原始响应、解析状态和 usage 信息。
- `predictions.jsonl`
  记录题级预测结果。
- `metrics.json`
  记录方法级 summary 指标。
- `run_summary.json`
  记录简短运行摘要。
- `budget_fairness.json`
  记录 SC / MV 在同预算下的 token 公平性检查结果。
- `paper_tables.md`
  导出的论文风格表格。
- `run_validation.json`
  运行后校验结果。
- `progress.json`
  运行过程中的进度快照。

此外，runner 会更新：

- `local/reports/single_agent/leaderboard.csv`

## 6. 实验约定

- 模型输入采用统一 JSON 输出协议，目标字段为 `reasoning` 与 `final_answer`。
- 当模型未返回合法 JSON 时，会先尝试通用解析修复，再按数据集规则做兜底抽取。
- `cot` 默认为单次确定性求解；`sc_*` 与 `mv_*` 通过多次调用构造等预算比较。
- phase 级别的标签过滤和 benchmark 过滤会在真实请求发出前做 fail-fast 校验。

## 7. 适用场景

适合用于：

- 新模型接入后的快速 baseline 验证。
- 不同单智能体推理方法的等预算比较。
- 论文或实验报告中单模型基线部分的数据生成。

不适合用于：

- 多智能体 debate 协议研究。
- trigger / early-exit 选择性通信研究。

