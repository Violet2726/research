# research_experiments.core

`research_experiments/core/` 是全仓唯一共享核心层，负责为所有实验家族提供通用运行能力。

## 分层结构

- `config/`
  provider、model、benchmark 的共享配置加载与解析。
- `data/`
  数据集加载、冻结 split 与答案评分归一化。
- `execution/`
  provider 调用、缓存、限流、runner 原语与运行时收尾。
- `prompts/`
  跨实验共享的题型指令与提示词契约。
- `controls/`
  跨实验复用的控制逻辑，如选择性通信信号与无通信对照。
- `structured_outputs/`
  按语义 schema 组织的结构化输出校验与恢复层，避免共享核心继续吸收 family 语义。

## 关键职责

- `config/catalog.py`
  解析 benchmark、provider 与 model catalog。
- `data/datasets.py`
  读取上游数据源并生成冻结 split。
- `execution/providers/`
  统一 provider 客户端、请求载荷与响应归一化。
- `execution/runtime.py`
  统一管理进度、`run_id` 与 run 收尾流程。
- `reporting/report_pipeline.py`
  统一输出本地报告、图资产与附录报告。
- `matrix/faithful_matrix.py`
  统一 faithful matrix 的编排、恢复、分析与矩阵级摘要输出。

## 维护约定

- 新共享能力只进入 `research_experiments/core`
- 工作区、归档、同步与数据集资产统一进入 `research_experiments/workspace`
- `runs/` 与 `cache/` 的正式远程归档统一走 Hugging Face dataset repo，不再发明实验家族私有同步逻辑
