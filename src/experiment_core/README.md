# `experiment_core` 目录说明

`experiment_core/` 是全仓库唯一的共享核心层，负责为所有实验线提供通用能力。

## 主要文件

- `config.py`：解析 benchmark、provider、model catalog
- `datasets.py`：加载数据与冻结 split
- `providers/`：OpenAI-compatible provider 访问封装
- `evaluation.py`：统一评分逻辑
- `structured_output.py`：结构化输出校验
- `runtime.py`：运行进度与 `run_id` 工具
- `workspace.py`：默认输出目录与环境变量覆盖

## 维护约定

- 这里只放跨实验复用的能力
- 不在这里放具体实验策略
- 文本 I/O 统一显式使用 `encoding="utf-8"`
