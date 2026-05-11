# configs/selective_comm

选择性通信实验配置根目录。

## 目录组成

- `experiments/`：正式实验入口配置
- `protocols/`：共享 `Stage A / Stage B` 协议
- `policies/`：可组合的 trigger 策略
- `controls/`：无通信或固定通信对照

## 当前正式实验

- `trigger_early_exit_main`
- `voc_trigger_main`

## 维护约定

- 新策略优先作为 `policies/` 下的可组合配置加入。
- 不再为单个模型复制一套 experiment 命名，只保留语义唯一版本。
- 变体型策略若不再被正式 experiment 引用，应及时删除，不保留历史残留配置。
