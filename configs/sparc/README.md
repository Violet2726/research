# configs/sparc

SPARC 主实验与消融实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：消息内容、聚合和局部审计协议

## 维护约定

- experiment 名直接体现消融目的，例如 `content_ablation_v1`。
- 聚合方法和局部审计策略优先在 protocol 层切换，不在 runner 里硬编码分叉。
