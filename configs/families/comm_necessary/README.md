# configs/comm_necessary

HotpotQA split-context 通信必要性实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：不同通信包粒度与聚合协议

## 维护约定

- full-context、split-context 和 evidence packet 差异优先体现在 protocol 或 view 层。
- supporting facts 相关字段保持稳定，避免同义字段漂移。
