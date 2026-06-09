# configs/single_agent

单智能体实验配置根目录。

## 目录组成

- `experiments/`：实验入口配置，负责组合 benchmark、method 集合和 phase 约束。
- `methods/`：方法卡片，只描述方法本身，不复制 benchmark 信息。

## 当前主线

- `experiments/canonical_simple_baselines.toml` 是 `xiaomimimo/mimo-v2.5` 当前正式 simple baseline 入口。
- 主线方法固定为 `cot_1@temp=0.7 / mv_3@temp=0.7 / sc_5@temp=0.7`。
- `count100` 使用 3 reruns，`competition_math` 固定解析到 `count100_total_seed0`。
- `count20` 只用于实现校验和轻量 sanity check，不再承担 prompt/temperature 筛选职责。
- 旧的 `baseline_ceiling_v1_*` screening 配置与 `baseline_ceiling_candidates.toml` 已退休，不应再新增实验依赖它们。
