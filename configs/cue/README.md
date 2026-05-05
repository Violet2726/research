# configs/cue

CUE 实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `policies/`：utility 公式、阈值和消息上限
- `protocols/`：solver、通信和审计协议

## 维护约定

- `cue_v1` 一类 experiment 负责装配框架版本，不在 experiment 名里带模型后缀。
- utility 组件、阈值和消息类型切换应保持配置化。
