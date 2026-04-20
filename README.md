# Research Experiments

统一共享核心层的研究实验仓库，当前包含 3 条独立实验线：

- `single_agent_baselines`：单智能体 CoT / Self-Consistency / Majority Vote
- `multi_agent_baselines`：多智能体 debate 基线
- `selective_comm`：触发式通信 / early-exit 实验

## 项目结构

- `src/experiment_core/`
  共享运行时能力：provider/model/benchmark 解析、方法目录、数据集加载与 split、缓存、provider 客户端、解析、评测、限流、运行进度
- `src/single_agent_baselines/`
  单智能体实验实现
- `src/multi_agent_baselines/`
  多智能体实验实现
- `src/selective_comm/`
  选择性通信实验实现
- `configs/shared/`
  共享 benchmarks、splits、providers、model catalog
- `configs/single_agent/`
  单智能体 experiments、methods
- `configs/multi_agent/`
  多智能体 experiments、controls、protocols、rosters
- `configs/selective_comm/`
  选择性通信 experiments、controls、policies、protocols

## 安装

```powershell
uv sync
```

```powershell
Copy-Item .env.example .env.local
```

在 `.env.local` 中填入需要的 API Key，例如：

- `DASHSCOPE_API_KEY`
- `ZHIPU_API_KEY`

## 常用命令

生成共享 split：

```powershell
uv run single-agent-cli generate-splits
```

查看模型目录：

```powershell
uv run single-agent-cli list-models
```

查看单智能体实验配置：

```powershell
uv run single-agent-cli inspect-experiment --experiment configs/single_agent/experiments/main-baselines.toml
```

运行单智能体实验：

```powershell
uv run single-agent-cli run --experiment configs/single_agent/experiments/main-baselines.toml --phase smoke20 --model dashscope/qwen2.5-7b-instruct
```

查看多智能体实验配置：

```powershell
uv run mad-cli inspect-experiment --experiment configs/multi_agent/experiments/vanilla_mad_minimal.toml --backbone dashscope/qwen-turbo-1101
```

运行多智能体实验：

```powershell
uv run mad-cli run --experiment configs/multi_agent/experiments/vanilla_mad_minimal.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
```

查看选择性通信实验配置：

```powershell
uv run selective-cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml
```

运行选择性通信实验：

```powershell
uv run selective-cli run --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml --phase smoke20
```

## 默认输出

- 单智能体：
  `local/runs/single_agent/`
  `cache/single_agent_requests.sqlite`
  `local/reports/single_agent/`
- 多智能体：
  `local/runs/multi_agent/`
  `cache/multi_agent_requests.sqlite`
  `local/reports/multi_agent/`
- 选择性通信：
  `local/runs/selective_comm/`
  `cache/selective_comm_requests.sqlite`
  `local/reports/selective_comm/`

## 开发约束

- `experiment_core` 是唯一共享层
- 实验包之间禁止互相导入
- 共享配置统一放在 `configs/shared/`
- 单智能体配置统一放在 `configs/single_agent/`
