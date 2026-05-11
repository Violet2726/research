# configs/single_agent

单智能体实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `methods/`：`cot`、`sc_*`、`mv_*` 等方法卡片

## 维护约定

- `experiments/` 负责组合 benchmark、method 集合和 phase 约束。
- `methods/` 只描述单个方法本身，不在这里复制 benchmark 信息。
