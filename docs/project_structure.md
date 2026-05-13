# 项目结构说明

## 1. 总体分层

仓库分为四层：

1. `src/research_experiments/core/`
   共享基础设施层，负责 provider、缓存、数据集加载、评测、结构化输出、归档与运行时工具。
2. `src/research_experiments/families/<family>/`
   具体实验实现层。每个 family 只依赖共享核心，不互相依赖。
   `src/research_experiments/families/shared/` 只承接 family 侧共享脚手架，不算实验家族。
3. `configs/core/` 与 `configs/families/`
   配置层。共享配置放在 `configs/core/shared/`，faithful matrix 规格放在 `configs/core/matrix/`，实验专属配置放在各自 family 目录。
4. `datasets/`、`files/`、`local/`
   数据与工作区层。`datasets/` 只保留恢复说明，真实原始数据放在 `local/datasets/`。

## 2. 关键目录职责

### `src/research_experiments/core/`

- `config/`：provider、model、benchmark 共享配置
- `data/`：数据集加载、split 与评分归一化
- `execution/`：provider、缓存、限流、runner 原语、运行时
- `prompts/`：题型指令与提示词契约
- `controls/`：跨实验复用的控制逻辑
- `structured_outputs/`：共享结构化输出校验与恢复

### `src/research_experiments/families/`

- `shared/`：family CLI、配置加载、报告/校验共用脚手架
- `<family>/config.py`：实验配置解析
- `<family>/prompts.py`：提示词模板
- `<family>/algorithms.py` / `dataset_views.py`：family 机制逻辑
- `<family>/run/execute.py`：顶层运行编排
- `<family>/run/io.py`：运行目录与落盘路径
- `<family>/run/sample.py`：样本级执行链路与私有辅助逻辑
- `<family>/run/report.py`：汇总与报告
- `<family>/run/validate.py`：运行校验
- `<family>/spec.py`：family CLI 规格

### 其他共享目录

- `src/research_experiments/matrix/`：faithful matrix 编排、分析与验收
- `src/research_experiments/reporting/`：科研报告、图资产、论文包与统计输出
- `src/research_experiments/workspace/`：工作区布局、归档、HF 同步、数据集资产
- `src/research_experiments/cli_support/`：命令行输出与 UTF-8 编码支持
- `src/research_experiments/tools/`：缓存检查、归档、数据集与清理工具

## 3. 默认工作区

默认路径由 [layout.py](/d:/user/research/src/research_experiments/workspace/layout.py) 统一管理：

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

本仓库不把运行产物当作 Git 主线产物，而是使用：

- `RESEARCH_RUNS_HF_REPO`
- `RESEARCH_CACHE_HF_REPO`
- `RESEARCH_AUTO_PUBLISH_RUNS`
- `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT`

语义分别是：

- `runs`：正式 run 的公开 dataset repo 归档
- `cache`：latest-only 私有或受控 dataset repo 快照
- 自动开关：决定是否在 run 完成或批量脚本结束后自动同步

## 5. UTF-8 与文本规范

- 文本文件统一采用 UTF-8
- Python 文本 I/O 一律显式写 `encoding="utf-8"`
- 中文注解、docstring 与文档写法见 [code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)
