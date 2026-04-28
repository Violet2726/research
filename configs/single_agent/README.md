# `configs/single_agent` 目录说明

这个目录存放单智能体实验使用的配置。

## 当前子目录

- `experiments/`：实验入口配置
- `methods/`：CoT、SC、MV 等方法目录

## 维护约定

- `experiments/` 负责组合 benchmark、method catalog 与阶段设置
- `methods/` 负责定义单个方法的预算与采样参数
