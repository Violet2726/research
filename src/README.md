# src

`src/` 存放全部 Python 实现。

## 结构

- `research_experiments/core/`
  唯一共享核心层。
- `research_experiments/families/`
  各实验家族实现，如 `single_agent`、`multi_agent`、`selective_comm`、`sparc`。
- `research_experiments/matrix/`
  faithful matrix 编排与分析。
- `research_experiments/reporting/`
  共享报告、图表与论文包能力。
- `research_experiments/tools/`
  工作区、缓存、归档与数据集工具。

## 约定

- family 之间不直接互相导入
- 共享能力统一下沉到 `research_experiments/core`
- 默认工作区路径与 Hugging Face 归档设置统一由 `research_experiments.core.foundation.workspace` 管理
