# `comm_necessary` 目录说明

`comm_necessary/` 用于运行 HotpotQA split-context 通信必要性实验，重点研究不同通信强度对答案与 supporting facts 的影响。

## 主要文件

- `config.py`：实验与协议配置解析
- `dataset_views.py`：构造 split-context 与 full-context 视图
- `logic.py`：通信包构造、聚合与 HotpotQA 评分
- `runner.py`：主执行流程
- `reporting.py`：摘要与正式报告
- `validation.py`：运行结果校验
- `cli.py`：命令行入口

## 对应配置

- `configs/comm_necessary/`
