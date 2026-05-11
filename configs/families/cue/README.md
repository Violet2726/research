# configs/cue

CUE 实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `policies/`：utility 公式、阈值和消息上限
- `protocols/`：solver、通信和审计协议

## 维护约定

- 正式实验入口统一使用 `cue_black_box_utility_main.toml` 这类语义化命名，不再把历史框架版本直接写成 experiment 名。
- `cue_v1.toml` 保留在 `policies/` 中作为效用策略配置，而不是正式 experiment 入口。
- utility 组件、阈值和消息类型切换应保持配置化。
