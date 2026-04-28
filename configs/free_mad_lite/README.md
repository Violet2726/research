# `configs/free_mad_lite` 目录说明

这个目录存放 Free-MAD-lite 实验的配置。

## 当前子目录

- `experiments/`：实验入口配置
- `protocols/`：agent 数量与辩论轮次配置

## 维护约定

- 实验入口负责组合 benchmark、协议、方法列表与阶段
- 协议文件尽量稳定，便于多版本实验复用
