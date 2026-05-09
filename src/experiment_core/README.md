# experiment_core

`experiment_core/` 是全仓唯一共享核心层，负责为所有实验家族提供通用运行能力。

## 分层结构

- `foundation/`
  - 基础设施层，放配置解析、数据集加载、缓存、运行时、provider、结构化输出、通用评测等能力。
- `controls/`
  - 跨实验复用的控制逻辑层，放无通信对照执行器、选择性通信信号工具等。
- `matrix/`
  - faithful 实验矩阵层，放矩阵入口、实验规格、分析与验收逻辑。
- `reporting/`
  - 论文与报告产物层，放统计汇总、run 级图资产渲染、figure manifest 与发布文档生成。
  - `report_pipeline.py` 负责统一总装 local report、published report 与附录报告。
  - `report_views.py` 提供 summary / diagnostic 行的显式视图，减少字典字段漂移风险。
- `tools/`
  - 运维工具层，放缓存检查、失效产物清理等 CLI 工具。

## 目录责任

- `foundation/config.py`：共享 benchmark / provider / model catalog 解析
- `foundation/datasets.py`：数据集读取与 frozen split 支持
- `foundation/providers/`：OpenAI-compatible provider 封装与重试逻辑
- `foundation/structured_output.py`：结构化输出校验与恢复
- `foundation/evaluation.py`：统一评分逻辑
- `foundation/runtime.py`：运行进度、时间戳与 `run_id`
- `foundation/workspace.py`：默认 `runs/`、`reports/`、`cache/`、`files/` 路径
- `foundation/cache.py`：请求缓存、分库路由与统计能力
- `foundation/rate_limits.py`：请求并发与 RPM / TPM 节流
- `matrix/faithful_matrix.py`：统一的 faithful 实验矩阵入口，支持 `smoke20` 与 `pilot100`
- `reporting/run_figures.py`：run 级 `figures/`、`figure_manifest.json` 与科研图 SVG/CSV 渲染
- `reporting/report_pipeline.py`：正式 `report.md`、发布报告和附录报告的统一输出管线
- `reporting/report_views.py`：报告层使用的显式 summary / diagnostic 行视图
- `reporting/paper_package.py`：论文打包、matrix 级 figure 产物与摘要文档生成
- `tools/artifact_cleanup.py`：失效运行与失效报告清理工具
- `tools/cache_inspector.py`：缓存分库统计与目标定位工具

## 维护约定

- 这里只放跨实验复用能力，不放某个实验家族私有策略。
- 新的共享能力优先按职责落到对应子层，不再堆回 `experiment_core/` 根目录。
- 高层矩阵编排、论文产物和运维工具不再和基础设施模块并列混放。
- 文本 I/O 统一显式使用 `encoding="utf-8"`。
