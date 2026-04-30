# CUE 实验说明

## 1. 模块定位

`cue` 用于运行 CUE（Communication Utility Estimation）实验：

- 先独立求解
- 再基于可观测文本信号估计通信效用
- 只对高 utility 冲突做定向通信
- 必要时用局部审计器做最小验证

该包依赖共享层 `experiment_core`，不依赖其他实验包。

## 2. 输出目录

- 默认运行目录：`runs/cue/`
- 默认报告目录：`reports/cue/`

