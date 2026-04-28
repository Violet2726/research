# `src` 目录说明

`src/` 是仓库的源码根目录，所有可安装 Python 包都放在这里。

## 当前内容

- `experiment_core/`：共享基础设施层
- `single_agent_baselines/`：单智能体实验
- `multi_agent_baselines/`：多智能体实验
- `selective_comm/`：选择性通信实验
- `sparc/`：SPARC 实验
- `budget_comm/`：预算约束通信实验
- `sid_lite/`：SID-lite 实验
- `free_mad_lite/`：Free-MAD-lite 实验
- `comm_necessary/`：通信必要性实验

## 维护约定

- 共享能力优先下沉到 `experiment_core/`
- 不同实验包之间不要互相导入
- `research_experiments.egg-info/` 是构建产物，不作为人工维护目录
