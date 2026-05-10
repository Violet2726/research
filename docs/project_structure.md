# 项目结构说明

## 1. 总体分层

仓库分为四层：

1. `experiment_core`
   共享基础设施层，负责 provider、缓存、数据集加载、评测、结构化输出、归档与运行时工具。
2. `src/<experiment_kind>/`
   具体实验实现层。每个包只依赖 `experiment_core`，不互相依赖。
3. `configs/`
   配置层。共享配置放在 `configs/shared/`，实验专属配置放在各自子目录。
4. `datasets/`、`files/`、`local/`
   数据与工作区层。`datasets/` 只保留恢复说明，真实原始数据放在 `local/datasets/`。

## 2. 关键目录职责

### `src/experiment_core/`

- `foundation/`：配置、缓存、数据集、provider、限流、运行时、归档
- `controls/`：跨实验复用的控制逻辑
- `matrix/`：faithful matrix 编排、分析与验收
- `reporting/`：科研报告、图资产、论文包与统计输出
- `tools/`：缓存检查、归档、清理等 CLI 工具

### `src/<experiment_kind>/`

每条实验线通常包含：

- `config.py`
- `prompting.py`
- `logic.py`
- `runner.py`
- `reporting.py`
- `validation.py`
- `cli.py`

## 3. 默认工作区

默认路径由 [workspace.py](/d:/user/research/src/experiment_core/foundation/workspace.py) 统一管理：

- `local/runs/<family>/<experiment>/<phase>/<run_id>/`
- `local/reports/<family>/`
- `local/cache/providers/<provider>/<request_model>/<dataset>/requests.sqlite`
- `local/datasets/<dataset>/...`
- `files/`

### 环境变量覆盖

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_DATASETS_ROOT`
- `RESEARCH_FILES_ROOT`

示例：

```powershell
$env:RESEARCH_RUNS_ROOT = "D:/artifacts/runs"
$env:RESEARCH_REPORTS_ROOT = "D:/artifacts/reports"
$env:RESEARCH_CACHE_ROOT = "D:/artifacts/cache"
$env:RESEARCH_DATASETS_ROOT = "D:/artifacts/datasets"
$env:RESEARCH_FILES_ROOT = "D:/artifacts/files"
```

## 4. Hugging Face 远程归档

本仓库不再把 `runs/` 与 `cache/` 当作 Git 主线产物，而是使用：

- `RESEARCH_RUNS_HF_REPO`
- `RESEARCH_CACHE_HF_REPO`
- `RESEARCH_AUTO_PUBLISH_RUNS`
- `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT`

语义分别是：

- `runs`：正式 run 的公开 dataset repo 归档
- `cache`：latest-only 私有或受控 dataset repo 快照
- 自动开关：决定是否在 run 完成或批量脚本结束后自动同步

## 5. 顶层 `runs/`、`reports/`、`cache/`

顶层这三个目录现在只保留说明文件与历史占位，不再作为默认工作区：

- 实际运行输出默认进入 `local/`
- 正式远程归档进入 Hugging Face
- Git 主仓不再承担这些目录下的正式产物版本管理

## 6. UTF-8 与文本规范

- 文本文件统一采用 UTF-8
- Python 文本 I/O 一律显式写 `encoding="utf-8"`
- 中文注解、docstring 与文档写法见 [code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)
