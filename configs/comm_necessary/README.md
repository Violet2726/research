# `configs/comm_necessary` 目录说明

这个目录存放 HotpotQA 通信必要性实验的配置。

## 当前子目录

- `experiments/`：实验入口配置
- `protocols/`：通信协议与 token cap 配置

## 维护约定

- 实验入口负责组合 benchmark、协议与阶段
- 协议文件负责约束不同通信方式的消息规模
