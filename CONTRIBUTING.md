# Contributing

## Setup

```powershell
uv sync --group dev
Copy-Item .env.example .env.local
```

如果只做本地实验，请把临时输出写到 `local/`，或通过 `RESEARCH_RUNS_ROOT`、`RESEARCH_REPORTS_ROOT`、`RESEARCH_CACHE_ROOT`、`RESEARCH_FILES_ROOT` 覆盖默认产物目录。

## Before You Commit

```powershell
uv run pytest
```

如果你修改了：

- CLI 入口：同步检查 `pyproject.toml`、`README.md` 和相关 `docs/`
- 实验配置名：同步检查根 README、`configs/*/README.md`、`src/*/README.md`
- 共享层结构：同步检查 `src/experiment_core/README.md` 与 `docs/project_structure.md`
- 中文注解或仓库说明：遵循 [docs/code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)

## Annotation Policy

- 共享层和仓库级文档默认使用中文注解。
- docstring 以“说明职责、输入输出、关键约束”为主，不堆叠实现细节。
- 注释只写非显然信息，不写变量赋值这类低信息内容。
- 新增文本文件统一使用 UTF-8。

## Artifact Policy

- 本仓库当前按“源码 + 代表性实验档案”维护，允许保留部分 `runs/`、`reports/` 和 `cache/` 产物作为可复查证据。
- 一次性调试、失败中间产物和个人分析笔记不要直接混入主线。
- 若需保留新的正式产物，请确保它们能支撑复现实验或论文结论，而不是仅仅重复已有内容。
