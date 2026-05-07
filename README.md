# Research Experiments

统一的研究实验仓库，面向单智能体、多智能体、选择性通信、预算约束通信和局部审计等推理实验。

## 当前实验线

- `single_agent`：单智能体 CoT / Self-Consistency 基线
- `multi_agent`：标准多智能体 debate 与 vote 对照
- `selective_comm`：trigger / early-exit 选择性通信
- `sparc`：内容裁剪、局部审计与聚合消融
- `budget_comm`：预算约束通信与 auction / knapsack 风格分配
- `sid_lite`：SID-lite 机制验证
- `free_mad_lite`：Free-MAD-lite 机制验证
- `comm_necessary`：HotpotQA split-context 通信必要性实验
- `cue`：Communication Utility Estimation 黑盒选择性通信框架

## 目录概览

```text
src/
  experiment_core/   唯一共享核心层
  single_agent/      单智能体实验
  multi_agent/       多智能体实验
  selective_comm/    选择性通信实验
  sparc/             SPARC 相关实验
  budget_comm/       预算约束通信实验
  sid_lite/          SID-lite 实验
  free_mad_lite/     Free-MAD-lite 实验
  comm_necessary/    通信必要性实验
  cue/               CUE 实验

configs/
  shared/            benchmark / provider / model catalog
  <family>/          各实验线自己的 experiments / protocols / policies

datasets/            本地基准数据
docs/                仓库级设计说明
files/               研究笔记与参考资料
runs/                默认运行产物
reports/             默认发布报告
tests/               回归测试
```

更详细的结构说明见 [docs/project_structure.md](docs/project_structure.md)。

## 仓库约定

- 共享能力只放在 `src/experiment_core/`。
- 不同实验包之间不直接互相导入。
- 公共配置字段统一使用 `primary_model_ref`。
- 默认运行目录统一为 `runs/<family>/<experiment>/<phase>/<run_id>/`。
- 默认报告目录统一为 `reports/<family>/`，跨家族汇总放在 `reports/summary/`。
- 文本文件统一使用 UTF-8。

## 安装

```powershell
uv sync
Copy-Item .env.example .env.local
```

将需要的 API Key 写入 `.env.local`。只提交 `.env.example`，不要提交真实密钥。

如果 Windows PowerShell 出现中文乱码，先执行：

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
```

## 常用命令

查看单智能体实验配置：

```powershell
uv run single_agent_cli inspect-experiment --experiment configs/single_agent/experiments/main_baselines.toml
```

运行单智能体 `smoke20`：

```powershell
uv run single_agent_cli run --experiment configs/single_agent/experiments/main_baselines.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
```

查看多智能体实验配置：

```powershell
uv run multi_agent_cli inspect-experiment --experiment configs/multi_agent/experiments/multi_agent_main.toml
```

运行选择性通信实验：

```powershell
uv run selective_comm_cli run --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
```

运行 CUE：

```powershell
uv run cue_cli run --experiment configs/cue/experiments/cue_v1.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
```

查看全量 `smoke20` 矩阵：

```powershell
uv run faithful_matrix_cli inspect-matrix
```

按统一模型与限流运行全量 `smoke20`：

```powershell
uv run faithful_matrix_cli run --model xiaomimimo/mimo-v2.5 --phase smoke20
```

清理失效产物：

```powershell
uv run cleanup_artifacts_cli --dry-run
```

## 运行时目录覆盖

默认工作目录由 `src/experiment_core/workspace.py` 统一管理，可通过环境变量覆盖：

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_FILES_ROOT`

示例：

```powershell
$env:RESEARCH_RUNS_ROOT = "artifacts/runs"
$env:RESEARCH_REPORTS_ROOT = "artifacts/reports"
$env:RESEARCH_CACHE_ROOT = "artifacts/cache"
uv run selective_comm_cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml
```

## 文档入口

- [src/README.md](src/README.md)
- [src/experiment_core/README.md](src/experiment_core/README.md)
- [configs/README.md](configs/README.md)
- [docs/README.md](docs/README.md)
