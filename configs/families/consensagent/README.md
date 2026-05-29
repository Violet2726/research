# configs/consensagent

基于触发机制的反谄媚多智能体辩论配置目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：辩论协议配置
- `rosters/`：智能体阵容配置

## 维护约定

- `experiments/` 负责组合 benchmark、协议集合和 phase 约束。
- `protocols/` 只描述辩论协议本身，不在这里复制 benchmark 信息。
- `rosters/` 定义参与辩论的智能体角色和数量。
