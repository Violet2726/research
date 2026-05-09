# 项目结构说明

## 1. 总体分层

仓库分为四层：

1. `experiment_core`
   共享基础设施层，负责 provider、缓存、数据集加载、评测、结构化输出、路径配置与运行时工具。
2. `src/<experiment_kind>/`
   具体实验实现层。每个包只依赖 `experiment_core`，不互相依赖。
3. `configs/`
   配置层。共享配置放在 `configs/shared/`，实验专属配置放在各自子目录。
4. `datasets/`、`files/`、`runs/`、`reports/`、`local/`、`cache/`
   数据与工作目录层，分别承载原始数据、研究资料、运行产物、正式报告、本地临时文件与请求缓存。

## 2. 关键目录职责

### `src/experiment_core/`

- `foundation/`：共享配置、provider、缓存、限流、运行时、结构化输出
- `controls/`：跨实验复用的控制逻辑
- `matrix/`：faithful 矩阵编排、分析与验收
- `reporting/`：正式报告、图资产、论文包与统计输出
- `tools/`：缓存检查、失效产物清理等运维工具

### `src/<experiment_kind>/`

每条实验线通常包含：

- `config.py`：实验配置解析
- `prompting.py`：提示词构造
- `logic.py`：实验机制或策略逻辑
- `runner.py`：主执行流程
- `reporting.py`：摘要、图资产与正式报告
- `validation.py`：运行结果校验
- `cli.py`：命令行入口

## 3. 配置目录约定

### `configs/shared/`

- `benchmarks/`：基准定义
- `benchmarks/splits/`：冻结 split 清单
- `providers/`：provider 默认配置
- `model_catalog.toml`：模型目录与覆盖项

### `configs/<experiment_kind>/`

按实验自身需要细分，例如：

- `experiments/`
- `protocols/`
- `policies/`
- `controls/`
- `views/`
- `rosters/`

## 4. 默认输出目录

默认输出统一由 `src/experiment_core/foundation/workspace.py` 管理，不再在各实验包里散落硬编码。

### 默认值

- `runs/<experiment_kind>/`
- `reports/<experiment_kind>/`
- `cache/providers/<provider>/<request_model>/<dataset>/requests.sqlite`
- `files/`

### 环境变量覆盖

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_FILES_ROOT`

示例：

```powershell
$env:RESEARCH_RUNS_ROOT = "artifacts/runs"
$env:RESEARCH_REPORTS_ROOT = "artifacts/reports"
$env:RESEARCH_CACHE_ROOT = "artifacts/cache"
$env:RESEARCH_FILES_ROOT = "artifacts/files"
```

## 5. `runs/` 与 `local/` 的关系

- `runs/`：默认运行产物目录，适合保留实验结果、`figures/` 图资产与归档
- `reports/`：默认正式报告目录，适合统一发布各实验线报告
- `local/`：本地临时目录，适合放不打算入库的草稿与调试文件

这样做的目的，是把“运行产物”“正式报告”和“本地临时文件”分开，同时让根目录下的产物入口保持一致。

## 6. UTF-8 与文本规范

- 文本文件统一采用 UTF-8
- `.editorconfig` 负责约束编码、换行和缩进
- Python 文本 I/O 一律显式写 `encoding="utf-8"`
- 中文注解、docstring 与文档写法见 [code_annotation_guidelines.md](/d:/user/research/docs/code_annotation_guidelines.md)

## 7. 开发边界

- 共享能力只能下沉到 `experiment_core`
- 实验包之间禁止交叉导入
- 新增输出目录或新类型产物时，优先先补 `workspace.py` 和文档，再接入具体 runner / reporting
