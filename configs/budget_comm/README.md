# budget_comm 配置说明

## 目录

- `experiments/`
  顶层实验声明。
- `protocols/`
  共享 Stage A / Stage B 协议配置。
- `policies/`
  DALA-lite v1 的 value density 与 packet cap 配置。
- `views/`
  same-context / split-context 视图轨道配置。

## 当前实验

- `dala_lite_same_context_v1.toml`
  GSM8K + StrategyQA + HotpotQA 的 same-context smoke20。
- `dala_lite_split_context_v1.toml`
  StrategyQA + HotpotQA 的 split-context smoke20。
