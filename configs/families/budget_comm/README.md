# configs/budget_comm

预算约束通信实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置
- `protocols/`：solver / belief update 协议
- `policies/`：auction / tier / gate 相关策略
- `views/`：same-context / split-context 数据视图

## 维护约定

- experiment 层声明视图与预算方案，策略细节下沉到 policy / protocol。
- same-context 与 split-context 保持同名结构，方便 paired comparison。
