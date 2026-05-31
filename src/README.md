# src

`src/` 存放全部 Python 实现。

## 结构

- `research_experiments/cli/`
  根 CLI、family CLI 接入层与工具命令包装器。
- `research_experiments/core/`
  唯一共享核心层，按 `contracts/`、`config/`、`data/`、`execution/`、`prompts/`、`controls/` 拆分通用基础设施。
- `research_experiments/family_runtime/`
  跨 family 共享的运行骨架、artifact 索引、比较器、提示模板与校验辅助。
- `research_experiments/families/`
  各实验家族实现；只保留具体实验代码与 `registration.py`。
- `research_experiments/workspace/`
  工作区布局、运行归档、HF 同步、清理与数据集资产服务；数据集资产公开入口收敛在 `workspace/datasets/__init__.py` 与 `workspace/datasets/service.py`。
- `research_experiments/cli_support/`
  CLI 输出、UTF-8 终端编码等命令行支撑。
- `research_experiments/matrix/`
  矩阵编排、恢复、分析与矩阵级状态模型。
- `research_experiments/reporting/`
  共享报告、图表与论文包能力。

## 约定

- family 之间不直接互相导入
- 共享能力统一下沉到 `research_experiments/core`
- 跨 family 的共享运行骨架统一下沉到 `research_experiments/family_runtime`
- family 注册信息统一由各自目录下的 `registration.py` 声明并自动发现
- 默认工作区路径与 Hugging Face 归档设置统一由 `research_experiments.workspace.layout` 管理
- 实验配置未显式声明运行时限流字段时，统一使用项目标准默认值 `90 / 95 / 9000000`

