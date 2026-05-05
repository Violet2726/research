# configs/selective_comm

选择性通信实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：共享 `Stage A` / `Stage B` 协议
- `policies/`：trigger 策略
- `controls/`：无通信或固定通信对照
- `reports/`：报告侧规则配置

## 维护约定

- 新策略优先作为 `policies/` 下的可组合配置加入。
- 不再为单个模型复制一套 experiment 名，只保留语义唯一版本。
