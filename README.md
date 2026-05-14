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
  research_experiments/
    cli_support/      CLI 输出与终端编码支持
    core/             共享核心能力
    families/         各实验家族
    matrix/           faithful matrix 编排
    reporting/        共享报告与论文包能力
    tools/            归档、缓存、数据集等工具
    workspace/        工作区布局、归档、同步与数据集资产

configs/
  core/
    shared/           benchmark / provider / model catalog
    matrix/           faithful matrix 规格
  families/
    <family>/         各实验线 experiments / protocols / policies

datasets/             数据集恢复说明与合规说明
docs/                 仓库级设计说明
files/                研究资料
local/                默认本地工作区（runs / reports / cache / datasets）
tests/                自动化测试
```

更详细的目录说明见 [docs/project_structure.md](/d:/user/research/docs/project_structure.md)。

## 仓库约定

- 共享能力只放在 `src/research_experiments/core/`
- family 之间不直接互相导入
- 公开配置字段统一使用 `primary_model_ref`
- faithful matrix 规格统一放在 `configs/core/matrix/faithful_matrix.toml`
- 默认工作区统一放在 `local/`
  - `local/runs/<family>/<experiment>/<phase>/<run_id>/`
  - `local/reports/<family>/`
  - `local/cache/providers/<provider>/<request_model>/<dataset_path_key>/requests.sqlite`
  - `local/datasets/<dataset_path>/...`
- 每个 run 的正式图资产统一固化在 `local/runs/.../figures/`，并通过 `figure_manifest.json` 编目
- 正式远程归档统一使用 Hugging Face dataset repo
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
uv run research_cli family single_agent inspect-experiment --experiment configs/families/single_agent/experiments/same_context_core_benchmarks.toml
uv run research_cli family single_agent run --experiment configs/families/single_agent/experiments/same_context_core_benchmarks.toml --phase count20 --model xiaomimimo/mimo-v2.5
uv run research_cli family single_agent render-report --run-dir local/runs/single_agent/same_context_core_benchmarks/count20/<run_id>
```

```powershell
uv run research_cli matrix inspect-matrix
uv run research_cli matrix run --model xiaomimimo/mimo-v2.5 --phase count20
```

```powershell
uv run research_cli tools dataset-assets prepare-used
uv run research_cli tools archive-runs publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
```

```powershell
.\run_all_phases.ps1
```

## 工作区与环境变量

默认工作区根目录由 [layout.py](/d:/user/research/src/research_experiments/workspace/layout.py) 统一管理：

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

顶层 `datasets/` 只保留恢复说明文档，不再承载原始 benchmark 大文件。

正式数据集资产统一放在 `local/datasets/`，并通过下面的命令一键恢复：

```powershell
uv run research_cli tools dataset-assets prepare-used
uv run research_cli tools dataset-assets prepare-all-sources
```

层级约束：

- `configs/core/shared/benchmarks/` 下的 benchmark 配置路径必须镜像 `local/datasets/` 的相对路径层级，并使用“去掉数据文件扩展名后的路径”作为配置路径。
- `local/cache/providers/<provider>/<request_model>/...` 下的数据集 cache 分库必须沿用同一套层级键。
- 示例：`local/datasets/dog-freebase/cwq.json` 对应 `configs/core/shared/benchmarks/dog-freebase/cwq.toml` 与 `local/cache/providers/xiaomimimo/mimo-v2-5/dog-freebase/cwq/requests.sqlite`。

## Hugging Face 归档

项目采用单一正式归档语义：

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
uv run research_cli tools archive-runs publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
uv run research_cli tools archive-runs fetch-run --run-id <run_id>
```

```powershell
uv run research_cli tools cache-archive push-latest --cache-root local/cache
uv run research_cli tools cache-archive pull-latest --target local/cache
```

```powershell
uv run research_cli tools hf-sync status
uv run research_cli tools hf-sync push-workspace
uv run research_cli tools hf-sync pull-workspace
```

说明：

- family 级 run 在 `finalize_run_outputs()` 后会按环境开关自动发布到 `RESEARCH_RUNS_HF_REPO`
- `run_all_phases.ps1` / `run_all_phases.sh` 在启用 `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1` 时，会在四阶段结束后自动推送 cache 最新快照
- Git 主仓不承载 `local/runs/`、`local/reports/`、`local/cache/` 下的正式产物

## 文档入口

- [src/README.md](/d:/user/research/src/README.md)
- [src/research_experiments/core/README.md](/d:/user/research/src/research_experiments/core/README.md)
- [docs/README.md](/d:/user/research/docs/README.md)
- [docs/project_structure.md](/d:/user/research/docs/project_structure.md)
- [docs/run_report_pipeline.md](/d:/user/research/docs/run_report_pipeline.md)
- [docs/huggingface_archive_workflow.md](/d:/user/research/docs/huggingface_archive_workflow.md)
- [docs/huggingface_operations.md](/d:/user/research/docs/huggingface_operations.md)
- [CONTRIBUTING.md](/d:/user/research/CONTRIBUTING.md)
