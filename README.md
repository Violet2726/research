# Research Experiments

统一的研究实验仓库，面向单智能体、多智能体、选择性通信、预算约束通信、局部审计与论文复现实验。

## 当前实验线

- `single_agent`：单智能体 CoT / Self-Consistency 基线
- `multi_agent`：标准多智能体 debate 与 vote 对照
- `selective_comm`：trigger / early-exit 选择性通信
- `sparc`：内容裁剪、局部审计与联合消融
- `budget_comm`：预算约束通信与分配策略实验
- `sid_lite`：SID-lite 机制验证
- `free_mad_lite`：Free-MAD-lite 机制验证
- `comm_necessary`：HotpotQA split-context 通信必要性实验
- `cue`：Communication Utility Estimation 黑盒选择性通信框架

## 目录概览

```text
src/
  experiment_core/   唯一共享核心层
  <family>/          各实验线实现

configs/
  shared/            benchmark / provider / model catalog
  <family>/          各实验线自己的 experiments / protocols / policies

datasets/            本地基准数据
docs/                仓库级设计说明
files/               研究笔记与参考资料
runs/                默认运行产物
reports/             默认发布报告
tests/               自动化测试
```

更详细的目录分层说明见 [docs/project_structure.md](/d:/user/research/docs/project_structure.md)。

## 仓库约定

- 共享能力只放在 `src/experiment_core/`。
- 不同实验包之间不直接互相导入。
- 公开配置字段统一使用 `primary_model_ref`。
- 默认运行目录统一为 `runs/<family>/<experiment>/<phase>/<run_id>/`。
- 默认报告目录统一为 `reports/<family>/`，跨家族汇总放在 `reports/summary/`。
- 每个 run 的正式图资产统一固化在 `runs/<family>/<experiment>/<phase>/<run_id>/figures/`，并通过 `figure_manifest.json` 编目。
- 项目文本文件统一使用 UTF-8。
- 注解、docstring 和仓库级说明默认使用中文；写法约定见 [docs/code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)。

## 安装

```powershell
uv sync --group dev
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
uv run single_agent_cli inspect-experiment --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml
```

运行单智能体 `smoke20`：

```powershell
uv run single_agent_cli run --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
```

重渲染单智能体正式报告与图表：

```powershell
uv run single_agent_cli render-report --run-dir runs/single_agent/main_baselines/smoke20/<run_id>
```

查看多智能体实验配置：

```powershell
uv run multi_agent_cli inspect-experiment --experiment configs/multi_agent/experiments/same_context_controlled_debate.toml
```

运行选择性通信实验：

```powershell
uv run selective_comm_cli run --experiment configs/selective_comm/experiments/trigger_early_exit_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
```

运行 CUE：

```powershell
uv run cue_cli run --experiment configs/cue/experiments/cue_black_box_utility_main.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
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

默认工作目录由 `src/experiment_core/foundation/workspace.py` 统一管理，可通过环境变量覆盖：

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_FILES_ROOT`

示例：

```powershell
$env:RESEARCH_RUNS_ROOT = "artifacts/runs"
$env:RESEARCH_REPORTS_ROOT = "artifacts/reports"
$env:RESEARCH_CACHE_ROOT = "artifacts/cache"
uv run selective_comm_cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_main.toml
```

## 产物策略

- 本仓库当前按“源码 + 代表性实验档案”维护，`runs/`、`reports/` 和部分 `cache/` 产物可以作为可复查证据进入版本库。
- 图资产以 `runs/.../figures/` 为唯一规范落点；`reports/summary/` 与 `reports/<family>/` 中的 Markdown 只引用这些 canonical 图文件，不再复制图表到 `reports/figures/`。
- 一次性试验、临时调参与个人分析建议通过 `RESEARCH_*_ROOT` 输出到独立目录，或写入 `local/`，不要把临时产物直接混入主线。
- 批量跑矩阵阶段时，Linux / macOS 使用 [run_all_phases.sh](/d:/user/research/run_all_phases.sh)，Windows PowerShell 使用 [run_all_phases.ps1](/d:/user/research/run_all_phases.ps1)。

## 文档入口

- [src/README.md](/d:/user/research/src/README.md)
- [src/experiment_core/README.md](/d:/user/research/src/experiment_core/README.md)
- [configs/README.md](/d:/user/research/configs/README.md)
- [docs/README.md](/d:/user/research/docs/README.md)
- [docs/project_structure.md](/d:/user/research/docs/project_structure.md)
- [docs/run_report_pipeline.md](/d:/user/research/docs/run_report_pipeline.md)
- [docs/code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)
- [runs/README.md](/d:/user/research/runs/README.md)
- [CONTRIBUTING.md](/d:/user/research/CONTRIBUTING.md)

## Hugging Face 归档

后续 `runs/` 默认采用“可浏览外壳 + 压缩内核”的 Hugging Face dataset repo 归档方式：

- 在线保留：`report.md`、`metrics.json`、`figure_manifest.json`、`figures/*.svg`、`figures/*.csv`
- 重型 `*.jsonl` trace、预测目录与细碎中间产物压入 `traces.tar.zst` / `predictions.tar.zst` / `artifacts.tar.zst`
- `cache/` 不再依赖 Git LFS，改为 latest-only 快照语义，优先使用独立私有 repo 或本地阶段性备份

常用命令：

```powershell
uv run archive_runs_cli pack-run --run-root runs/<family>/<experiment>/<phase>/<run_id>
```

```powershell
uv run archive_runs_cli publish-run --run-root runs/<family>/<experiment>/<phase>/<run_id> --repo <owner>/research-runs
```

```powershell
uv run cache_archive_cli push-latest --cache-root cache --repo <owner>/research-cache
```
