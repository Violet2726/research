# API 基线实验操作手册

本文档对应当前仓库内的专业版无通信基线框架，覆盖以下方法：

- `Single-agent CoT`
- `Self-Consistency`
- `Majority Vote`

## 1. 实验目标

这套框架的目标不是“先跑起来再说”，而是从一开始就把以下要素固定住：

- 公平预算口径
- 固定数据子集
- 原始响应留档
- 可追溯缓存
- 题目级预测与聚合指标分离
- 后续可扩展到 debate / audit / trigger

## 2. 当前主实验设置

主实验 backbone：

- `qwen2.5-7b-instruct`

补充稳健性模型：

- `qwen2.5-math-7b-instruct`
- `deepseek-r1-distill-qwen-7b`
- `glm-4.6v-flash`

主实验数据集：

- `GSM8K`
- `StrategyQA`
- `HotpotQA`

方法矩阵：

- `CoT(1)`
- `SC(3)`
- `SC(5)`
- `SC(7)`
- `MV(3)`
- `MV(5)`
- `MV(7)`

## 3. 路径说明

- 模型配置：`configs/models/`
- 数据集配置：`configs/benchmarks/`
- 固定 split：`configs/benchmarks/splits/`
- 实验矩阵：`configs/experiments/`
- 运行产物：`runs/<run_id>/`
- 请求缓存：`cache/requests.sqlite`
- 汇总表：`reports/leaderboard.csv`

## 4. 环境准备

1. 安装依赖

```powershell
uv sync
```

2. 创建本地环境变量文件

```powershell
Copy-Item .env.example .env.local
```

3. 在 `.env.local` 中填写：

- `DASHSCOPE_API_KEY`
- `ZHIPU_API_KEY`

注意：

- 代码不会把 API Key 写入配置文件或日志。
- `.env.local` 已加入忽略规则，不会进入 git。

## 5. 固定 split 生成

首次使用或更新 benchmark 配置后，先生成固定 split：

```powershell
uv run baseline-cli generate-splits
```

当前固定规则：

- `GSM8K`：`smoke20` / `pilot100` / `dev300`
- `HotpotQA`：`smoke20` / `pilot100` / `dev300`
- `StrategyQA`：`smoke20` / `pilot100` / `dev_full_229`

说明：

- `StrategyQA` 本地官方 dev 只有 `229` 条，所以主配置使用 `dev_full_229`，这是刻意保持评测口径干净，而不是缺失功能。

## 6. 运行顺序

### 6.1 查看实验配置

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml
```

### 6.2 先跑 smoke

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20
```

这一步主要检查：

- endpoint 是否可用
- API 认证是否正确
- JSON 输出是否稳定
- 缓存是否工作
- usage 是否能正常落盘

### 6.3 再跑 pilot

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase pilot100
```

建议先检查：

- `SC(5)` 和 `MV(5)` 的平均总 token 是否接近
- `parse_fail` 是否过多
- 各数据集答案归一化是否符合预期

### 6.4 最后跑 main

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase main
```

### 6.5 跑稳健性实验

```powershell
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase smoke20
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase pilot100
```

## 7. 输出文件说明

每次运行会生成 `runs/<run_id>/`，目录中包含：

- `manifest.json`
  - 记录实验名、phase、模型配置、benchmark 配置、prompt 版本等
- `raw_responses.jsonl`
  - 每次 API 调用一行
  - 包含 `sample_id`、`method`、`replicate_id`、`agent_id`、`usage`、`latency`、`raw_response`
- `predictions.jsonl`
  - 每道题聚合后的一行结果
  - 包含最终投票答案、gold、score、题目级 token 总量与 latency
- `metrics.json`
  - 数据集级聚合指标

此外：

- `cache/requests.sqlite`
  - 按请求指纹缓存响应，避免重复烧 API
- `reports/leaderboard.csv`
  - 当前运行的汇总表

## 8. 公平性检查清单

在报告主表前，至少检查以下几点：

1. `CoT` 是否单独成表，而不是和 `B>1` 方法混排。
2. `SC(B)` 与 `MV(B)` 是否使用相同模型、相同 prompt、相同采样参数。
3. `SC(B)` 与 `MV(B)` 是否使用相同的题目清单。
4. 相同 `B` 下，`total_tokens_mean` 是否没有明显偏离。
5. `StrategyQA` 是否使用 `yes/no` 归一。
6. `GSM8K` 是否只比最终数值。
7. `HotpotQA` 当前是否只看 answer EM。

## 9. 常见问题

### 9.1 缺少 API Key

如果运行时报缺少 `DASHSCOPE_API_KEY` 或 `ZHIPU_API_KEY`：

- 检查 `.env.local` 是否存在
- 检查变量名是否写对
- 检查是否在仓库根目录执行命令

### 9.2 JSON 解析失败

如果 `raw_responses.jsonl` 里 `parse_status=parse_fail` 较多：

- 先降低输出长度
- 再检查 provider 是否稳定支持 `response_format=json_object`
- 必要时按模型单独调 prompt，但要保证同模型下所有方法共用同一 solver 风格

### 9.3 预算不对齐

如果 `SC(5)` 和 `MV(5)` 的平均总 token 相差过大：

- 先不要出主表
- 优先缩短 reasoning
- 必要时收紧 `max_output_tokens`

### 9.4 StrategyQA 为什么不是 dev300

因为本地官方 dev 只有 `229` 条。  
框架选择保留真实评测口径，而不是伪造一个所谓的 `dev300`。

## 10. 当前建议的首轮执行顺序

```powershell
uv run baseline-cli generate-splits
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase pilot100
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase main
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase smoke20
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase pilot100
```

如果想先控制费用，建议先只跑：

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20
```

确认日志、缓存、usage、解析都稳定之后，再进 `pilot100`。
