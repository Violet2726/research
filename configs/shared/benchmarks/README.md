# `configs/shared/benchmarks` 目录说明

这里存放 benchmark 的元配置文件。

## 当前内容

- 各数据集的基础定义文件，如 `gsm8k.toml`、`hotpotqa.toml`
- `splits/` 子目录中的冻结样本划分清单

## 维护约定

- benchmark 定义描述数据来源、字段名、默认样本规模与随机种子
- split 一旦用于实验，应尽量冻结并复用
