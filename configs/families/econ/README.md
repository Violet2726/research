# configs/econ

ECON 低通信协调论文复现配置目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：协调协议配置

## 维护约定

- `experiments/` 负责组合 benchmark、协议集合和 phase 约束。
- `protocols/` 只描述协调协议本身，不在这里复制 benchmark 信息。
