# Research Experiments

统一的研究实验仓库，覆盖单智能体、多智能体、选择性通信、预算约束通信、局部审计与论文复现实验。

## 当前实验线

- `single_agent`：单智能体 `CoT / Self-Consistency` 基线
- `multi_agent`：标准多智能体 `debate vs vote`
- `selective_comm`：trigger / early-exit 选择性通信
- `sparc`：内容压缩、局部审计与联合消融
- `budget_comm`：预算约束通信与分配策略
- `sid_lite`：SID-lite 机制验证
- `free_mad_lite`：Free-MAD-lite 机制验证
- `comm_necessary`：HotpotQA split-context 通信必要性
- `cue`：Communication Utility Estimation 黑盒选择性通信框架

## 目录概览

```text
src/
  experiment_core/   唯一共享核心层
  <family>/          各实验线实现

configs/
  shared/            benchmark / provider / model catalog
  <family>/          各实验线 experiments / protocols / policies

datasets/            数据集恢复说明与合规说明
docs/                仓库级设计说明
files/               研究资料
local/               默认本地工作区（runs / reports / cache / datasets）
runs/                仅保留说明文件与历史占位
reports/             仅保留说明文件与历史占位
cache/               仅保留说明文件与历史占位
tests/               自动化测试
```

更详细的目录说明见 [docs/project_structure.md](/d:/user/research/docs/project_structure.md)。

## 仓库约定

- 共享能力只放在 `src/experiment_core/`
- 不同实验包之间不直接互相导入
- 公开配置字段统一使用 `primary_model_ref`
- 默认工作区统一放在 `local/`
  - `local/runs/<family>/<experiment>/<phase>/<run_id>/`
  - `local/reports/<family>/`
  - `local/cache/providers/<provider>/<request_model>/<dataset>/requests.sqlite`
  - `local/datasets/<dataset>/...`
- 每个 run 的正式图资产统一固化在 `local/runs/.../figures/`，并通过 `figure_manifest.json` 编目
- 正式远程归档统一使用 Hugging Face dataset repo，不再把 `runs/` 与 `cache/` 作为 Git 产物提交
- 项目文本文件统一使用 UTF-8

## 安装

```powershell
uv sync --group dev
Copy-Item .env.example .env.local
```

把需要的 API Key 与 Hugging Face 配置写入 `.env.local`。只提交 `.env.example`，不要提交真实密钥。

如果 Windows PowerShell 出现中文乱码，先执行：

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
```

## 常用命令

```powershell
uv run single_agent_cli inspect-experiment --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml
uv run single_agent_cli run --experiment configs/single_agent/experiments/same_context_core_benchmarks.toml --phase smoke20 --model xiaomimimo/mimo-v2.5
uv run single_agent_cli render-report --run-dir local/runs/single_agent/same_context_core_benchmarks/smoke20/<run_id>
```

```powershell
uv run faithful_matrix_cli inspect-matrix
uv run faithful_matrix_cli run --model xiaomimimo/mimo-v2.5 --phase smoke20
```

```powershell
.\run_all_phases.ps1
```

## 工作区与环境变量

默认工作区根目录由 [workspace.py](/d:/user/research/src/experiment_core/foundation/workspace.py) 统一管理：

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_DATASETS_ROOT`
- `RESEARCH_FILES_ROOT`

默认值分别是：

- `local/runs`
- `local/reports`
- `local/cache`
- `local/datasets`
- `files`

如果要覆盖：

```powershell
$env:RESEARCH_RUNS_ROOT = "D:/artifacts/runs"
$env:RESEARCH_REPORTS_ROOT = "D:/artifacts/reports"
$env:RESEARCH_CACHE_ROOT = "D:/artifacts/cache"
$env:RESEARCH_DATASETS_ROOT = "D:/artifacts/datasets"
```

## 数据集资产

顶层 `datasets/` 现在只保留说明文档，不再承载原始 benchmark 大文件。

正式数据集资产统一放在 `local/datasets/`，并通过下面的命令一键恢复：

```powershell
uv run dataset_assets_cli prepare-used
uv run dataset_assets_cli prepare-all-sources
```

## Hugging Face 归档

项目现在采用单一正式归档语义：

- `runs`：公开 Hugging Face dataset repo，保存可浏览报告与压缩归档包
- `cache`：独立 latest-only Hugging Face dataset repo，保存最新快照

推荐在 `.env.local` 中配置：

```text
RESEARCH_RUNS_HF_REPO=Violet1307/research-runs
RESEARCH_CACHE_HF_REPO=Violet1307/research-cache
RESEARCH_AUTO_PUBLISH_RUNS=1
RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1
HF_TOKEN=hf_xxx
```

常用归档命令：

```powershell
uv run archive_runs_cli publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
uv run archive_runs_cli fetch-run --run-id <run_id>
```

```powershell
uv run cache_archive_cli push-latest --cache-root local/cache
uv run cache_archive_cli pull-latest --target local/cache
```

```powershell
uv run hf_sync_cli status
uv run hf_sync_cli push-workspace
uv run hf_sync_cli pull-workspace
```

说明：

- family 级 run 在 `finalize_run_outputs()` 后会按环境开关自动发布到 `RESEARCH_RUNS_HF_REPO`
- `run_all_phases.ps1` / `run_all_phases.sh` 在启用 `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1` 时，会在四阶段结束后自动推送 cache 最新快照
- 本仓库不再以 Git 形式承载 `runs/`、`reports/`、`cache/` 的正式产物

## 文档入口

- [src/README.md](/d:/user/research/src/README.md)
- [src/experiment_core/README.md](/d:/user/research/src/experiment_core/README.md)
- [docs/README.md](/d:/user/research/docs/README.md)
- [docs/project_structure.md](/d:/user/research/docs/project_structure.md)
- [docs/run_report_pipeline.md](/d:/user/research/docs/run_report_pipeline.md)
- [docs/huggingface_archive_workflow.md](/d:/user/research/docs/huggingface_archive_workflow.md)
- [docs/huggingface_operations.md](/d:/user/research/docs/huggingface_operations.md)
- [CONTRIBUTING.md](/d:/user/research/CONTRIBUTING.md)
