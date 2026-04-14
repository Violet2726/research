# API Baseline Experiments

这个仓库现在包含一套可复现的 API 基线实验框架，用于运行以下无通信方法：

- `Single-agent CoT`
- `Self-Consistency`
- `Majority Vote`

当前主实验配置遵循以下原则：

- 主 backbone 固定为 `qwen2.5-7b-instruct`
- 主数据集固定为 `GSM8K`、`StrategyQA`、`HotpotQA`
- `CoT` 单独作为 `1-call lower bound`
- `SC(B)` 与 `MV(B)` 仅在相同 `B` 下做 equal-budget 对比
- 全部运行都记录原始响应、解析结果、usage、延迟、缓存命中情况与聚合指标

## 环境准备

1. 安装依赖

```powershell
uv sync
```

2. 准备本地环境变量

```powershell
Copy-Item .env.example .env.local
```

然后把 `.env.local` 中的 `DASHSCOPE_API_KEY`、`ZHIPU_API_KEY` 填好。

## 生成固定 split

```powershell
uv run baseline-cli generate-splits
```

这会在 `configs/benchmarks/splits/` 下生成固定清单。

## 查看实验配置

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml
```

## 运行 smoke / pilot / main

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase pilot100
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase main
```

## 运行稳健性检查

```powershell
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase pilot100
```

## 产物目录

- `cache/requests.sqlite`：幂等请求缓存
- `runs/<run_id>/manifest.json`：完整实验卡
- `runs/<run_id>/raw_responses.jsonl`：单次 API 调用日志
- `runs/<run_id>/predictions.jsonl`：聚合后的题目级预测
- `runs/<run_id>/metrics.json`：指标与预算摘要
- `reports/leaderboard.csv`：汇总排行榜

## 设计注意事项

- `StrategyQA` 官方本地 dev 集只有 `229` 条，因此主配置里使用 `dev_full_229`，而不是伪造 `dev300`。
- `HotpotQA` 首轮主指标仅使用 answer EM，不纳入 supporting facts。
- 若某个 `SC(B)` 与 `MV(B)` 在同一 `B` 下平均总 token 偏离超过 `10%`，应先调整 `max_output_tokens` 或 prompt，再重新跑主表。
