# src

`src/` 是仓库的源码根目录，所有可安装 Python 包都放在这里。

## 目录组成

- `experiment_core/`：唯一共享核心层
- `single_agent/`：单智能体实验
- `multi_agent/`：多智能体实验
- `selective_comm/`：选择性通信实验
- `sparc/`：SPARC 相关实验
- `budget_comm/`：预算约束通信实验
- `sid_lite/`：SID-lite 实验
- `free_mad_lite/`：Free-MAD-lite 实验
- `comm_necessary/`：通信必要性实验
- `cue/`：CUE 实验

## 维护约定

- 共享能力优先下沉到 `experiment_core/`。
- 实验家族之间不直接导入彼此代码。
- 对外 CLI 名称与包名保持一致，统一使用 `snake_case`。
- 运行产物路径、报告路径和缓存路径都走共享 workspace 入口，不在家族包里硬编码。

更详细的共享层说明见 [experiment_core/README.md](/d:/user/research/src/experiment_core/README.md)。
