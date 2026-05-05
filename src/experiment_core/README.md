# experiment_core

`experiment_core/` 是全仓唯一共享核心层，负责为所有实验家族提供通用运行能力。

## 主要内容

- `config.py`：共享 benchmark / provider / model catalog 解析
- `datasets.py`：数据集读取与 frozen split 支持
- `providers/`：OpenAI-compatible provider 封装与重试逻辑
- `structured_output.py`：结构化输出校验与恢复
- `evaluation.py`：统一评分逻辑
- `runtime.py`：运行进度、时间戳和 `run_id`
- `workspace.py`：默认 `runs/`、`reports/`、`cache/`、`files/` 路径
- `rate_limits.py`：请求并发与 RPM / TPM 节流
- `smoke20_matrix.py`：统一的 `smoke20` 全量矩阵入口
- `artifact_cleanup.py`：失效运行与失效报告清理工具

## 维护约定

- 这里只放跨实验复用能力，不放某个实验独有策略。
- 新的结构化输出恢复逻辑优先放到共享层，而不是在单个 runner 里做临时补丁。
- 文本 I/O 统一显式使用 `encoding="utf-8"`。
