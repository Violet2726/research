# `sid_lite` 目录说明

`sid_lite/` 用于运行 SID-lite 机制验证实验，关注 early-exit、压缩通信和 belief update 的协同效果。

## 主要文件

- `config.py`：实验与协议配置解析
- `logic.py`：SID-lite 规则与聚合逻辑
- `prompting.py`：Stage A / belief update 提示词
- `runner.py`：主执行流程
- `reporting.py`：摘要与正式报告
- `validation.py`：运行结果校验
- `cli.py`：命令行入口

## 对应配置

- `configs/sid_lite/`
