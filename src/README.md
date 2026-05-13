# src

`src/` 存放全部 Python 实现。

## 结构

- `research_experiments/core/`
  唯一共享核心层，按 `config/`、`data/`、`execution/`、`prompts/` 拆分共享能力。
- `research_experiments/families/`
  各实验家族实现，以及仅服务 family 层的 `shared/` 共享脚手架。
- `research_experiments/workspace/`
  工作区布局、运行归档、HF 同步与数据集资产工具。
- `research_experiments/cli_support/`
  CLI 输出、UTF-8 终端编码等命令行支撑。
- `research_experiments/matrix/`
  faithful matrix 编排与分析。
- `research_experiments/reporting/`
  共享报告、图表与论文包能力。
- `research_experiments/tools/`
  工作区、缓存、归档与数据集工具。

## 约定

- family 之间不直接互相导入
- 共享能力统一下沉到 `research_experiments/core`
- family 级共享脚手架统一放在 `research_experiments/families/shared`
- 默认工作区路径与 Hugging Face 归档设置统一由 `research_experiments.workspace.layout` 管理

