# `free_mad_lite` 目录说明

`free_mad_lite/` 用于运行 Free-MAD-lite 机制验证实验，重点验证单轮 anti-conformity debate 与 trajectory judging。

## 主要文件

- `config.py`：实验与协议配置解析
- `logic.py`：方法定义、trajectory 决策与回退逻辑
- `prompting.py`：初始求解、反从众辩论与 judge 提示词
- `runner.py`：主执行流程
- `reporting.py`：摘要与正式报告
- `validation.py`：运行结果校验
- `cli.py`：命令行入口

## 对应配置

- `configs/free_mad_lite/`
