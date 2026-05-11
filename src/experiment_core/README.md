# experiment_core

`experiment_core/` 是全仓唯一共享核心层，负责为所有实验家族提供通用运行能力。

## 分层结构

- `foundation/`
  基础设施层，包含配置、数据集、缓存、provider、限流、运行时、归档与工作区管理。
- `controls/`
  跨实验复用的控制逻辑，如选择性通信信号与无通信对照。
- `structured_outputs/`
  按语义 schema 组织的结构化输出校验与恢复层，避免共享核心继续吸收 family 语义。
- `matrix/`
  faithful matrix 编排、分析与验收。
- `reporting/`
  科研报告、图资产、论文包与统计输出。
- `tools/`
  缓存检查、归档、清理等 CLI 工具。

## 关键职责

- `foundation/workspace.py`
  统一管理 `local/runs`、`local/reports`、`local/cache`、`files/` 与 HF 归档环境变量。
- `foundation/runtime.py`
  统一管理进度、`run_id` 与 run 收尾流程。
- `foundation/run_archives.py`
  统一打包 run 重型文件，并提供 HF 发布与回取能力。
- `foundation/cache_snapshots.py`
  统一管理 cache latest-only 快照的压缩、恢复与 HF 同步。
- `reporting/report_pipeline.py`
  统一输出本地报告、图资产与附录报告。
- `matrix/faithful_matrix.py`
  统一 faithful matrix 的编排、恢复、分析与矩阵级摘要输出。

## 维护约定

- 新共享能力只进入 `experiment_core`
- 新增运行产物时，优先补 `workspace.py`、归档合同与文档
- `runs/` 与 `cache/` 的正式远程归档统一走 Hugging Face dataset repo，不再发明实验家族私有同步逻辑
