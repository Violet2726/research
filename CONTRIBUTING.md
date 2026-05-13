# Contributing

## Setup

```powershell
uv sync --group dev
Copy-Item .env.example .env.local
```

默认工作区在 `local/`。如需覆盖，请使用：

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_FILES_ROOT`

如需启用正式远程归档，请同时配置：

- `RESEARCH_RUNS_HF_REPO`
- `RESEARCH_CACHE_HF_REPO`
- `RESEARCH_AUTO_PUBLISH_RUNS`
- `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT`

## Before You Commit

```powershell
uv run pytest
uv run python -m compileall src tests
```

如果你修改了：

- CLI 入口：同步检查 `pyproject.toml`、`README.md` 和相关 `docs/`
- 实验配置名：同步检查根 README、`src/*/README.md` 和 `configs/*`
- 共享结构：同步检查 `src/research_experiments/core/README.md`、`src/README.md` 与 `docs/project_structure.md`
- family 共享脚手架：同步检查 `src/research_experiments/families/shared/`
- 中文注解或仓库说明：遵循 [docs/code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)

## Annotation Policy

- 共享层和仓库级文档默认使用中文注解
- docstring 重点说明职责、输入输出和关键约束
- 注释只写非显然信息
- 新增文本文件统一使用 UTF-8

## Artifact Policy

- Git 主仓只承载代码、配置和文档
- `local/runs/`、`local/reports/`、`local/cache/` 是默认本地工作区，不进入主线
- 正式 `runs` 归档进入 `RESEARCH_RUNS_HF_REPO`
- 正式 `cache` 快照进入 `RESEARCH_CACHE_HF_REPO`
- 不再维护顶层 `runs/`、`reports/`、`cache/` 目录占位
