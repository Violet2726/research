# SPARC 实验说明

## 1. 模块定位

`sparc` 用于运行 SPARC v1 smoke 相关实验：

- communication content ablation
- aggregation / local auditing ablation
- SPARC v1 端到端 smoke

该包依赖共享层 `experiment_core`，不依赖其他实验包。

## 2. 目录结构

- `config.py`
  负责加载实验、协议和 backbone 配置。
- `logic.py`
  负责消息裁剪、退化规则、审计候选选择等纯逻辑。
- `prompting.py`
  负责 solver、belief update、judge 和 auditor 的提示词。
- `runner.py`
  主执行链路，负责共享 Stage A / Stage B、审计和结果落盘。
- `reporting.py`
  负责渲染中文报告和论文汇总表。
- `validation.py`
  负责关键运行约束校验。
- `cli.py`
  命令行入口，对外暴露 `sparc-cli`。

## 3. 输出目录

- 默认运行目录：`local/runs/sparc/`
- 默认报告目录：`local/reports/sparc/`
