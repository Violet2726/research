# 选择性通信实验配置目录

这里专门放 `trigger / early-exit / communication content / auditing` 这条实验线的配置，
避免和 `configs/experiments/`、`configs/multi_agent/` 混在一起。

目录职责：

- `experiments/`
  具体实验规格，绑定 benchmark、phase、协议、策略目录与控制方法目录。
- `protocols/`
  共享前缀协议定义，例如 `3 agent + 1 round` 的固定 debate 结构。
- `policies/`
  trigger 策略定义，例如 `always_communicate`、`hybrid_trigger`。
- `controls/`
  无通信或等预算控制方法定义，例如 `mv_3`、`mv_6`、`sc_6`。
- `reports/`
  报告模板与输出约定说明。

使用约定：

- 当前 v1 只回答“什么时候该通信，什么时候该 early exit”。
- `Stage A` 与 `Stage B` 必须共享前缀，不能为每个策略重复跑完整网络请求。
- 新增消息内容消融或局部审计时，只扩新 protocol / experiment，不修改当前 trigger 目录语义。
