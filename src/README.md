# src

`src/` 存放全部 Python 实现。

## 结构

- `experiment_core/`
  唯一共享核心层。
- `<family>/`
  各实验家族实现，如 `single_agent`、`multi_agent`、`selective_comm`、`sparc`。

## 约定

- experiment family 之间不直接互相导入
- 共享能力统一下沉到 `experiment_core`
- 默认工作区路径与 Hugging Face 归档设置统一由 `experiment_core.foundation.workspace` 管理
