# Research Experiments

统一共享核心层的研究实验仓库，当前包含 5 条独立实验线：

- `single_agent_baselines`：单智能体 CoT / Self-Consistency / Majority Vote
- `multi_agent_baselines`：多智能体 debate 基线
- `selective_comm`：触发式通信 / early-exit 实验
- `sparc`：内容消融、局部审计与 SPARC v1 smoke
- `budget_comm`：DALA-lite 风格的预算约束通信实验
- `sid_lite`：SID-lite 风格的置信度早退与压缩消息实验
- `free_mad_lite`：Free-MAD-lite 风格的单轮 anti-conformity 与轨迹裁决实验
- `comm_necessary`：HotpotQA split-context 通信必要性实验

## 项目结构

- `src/experiment_core/`
  共享运行时能力：provider/model/benchmark 解析、方法目录、数据集加载与 split、缓存、provider 客户端、解析、评测、限流、运行进度
- `src/single_agent_baselines/`
  单智能体实验实现
- `src/multi_agent_baselines/`
  多智能体实验实现
- `src/selective_comm/`
  选择性通信实验实现
- `src/sparc/`
  SPARC v1 smoke 实验实现
- `src/budget_comm/`
  DALA-lite budget-aware communication 实验实现
- `src/sid_lite/`
  SID-lite smoke20 机制验证实现
- `src/free_mad_lite/`
  Free-MAD-lite smoke20 机制验证实现
- `src/comm_necessary/`
  HotpotQA evidence exchange 通信必要性实验实现
- `configs/shared/`
  共享 benchmarks、splits、providers、model catalog
- `configs/single_agent/`
  单智能体 experiments、methods
- `configs/multi_agent/`
  多智能体 experiments、controls、protocols、rosters
- `configs/selective_comm/`
  选择性通信 experiments、controls、policies、protocols
- `configs/sparc/`
  SPARC experiments、protocols
- `configs/budget_comm/`
  DALA-lite experiments、policies、protocols、views
- `configs/sid_lite/`
  SID-lite experiments、protocols
- `configs/free_mad_lite/`
  Free-MAD-lite experiments、protocols
- `configs/comm_necessary/`
  Communication-necessary experiments、protocols

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
uv run single-agent-cli run --experiment configs/single_agent/experiments/main-baselines.toml --phase smoke20 --model dashscope/qwen-turbo-1101
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

查看 SPARC 实验配置：

```powershell
uv run sparc-cli inspect-experiment --experiment configs/sparc/experiments/content_ablation_v1.toml
```

运行 SPARC 实验：

```powershell
uv run sparc-cli run --experiment configs/sparc/experiments/sparc_v1_smoke.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
```

查看 budget_comm 实验配置：

```powershell
uv run budget-cli inspect-experiment --experiment configs/budget_comm/experiments/dala_lite_same_context_v1.toml
```

运行 DALA-lite same-context 实验：

```powershell
uv run budget-cli run --experiment configs/budget_comm/experiments/dala_lite_same_context_v1.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
```

查看并运行 SID-lite smoke20：

```powershell
uv run sid-lite-cli inspect-experiment --experiment configs/sid_lite/experiments/sid_lite_v1.toml
uv run sid-lite-cli run --experiment configs/sid_lite/experiments/sid_lite_v1.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
```

查看并运行 Free-MAD-lite smoke20：

```powershell
uv run free-mad-lite-cli inspect-experiment --experiment configs/free_mad_lite/experiments/free_mad_lite_v1.toml
uv run free-mad-lite-cli run --experiment configs/free_mad_lite/experiments/free_mad_lite_v1.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
```

查看并运行 HotpotQA 通信必要性 smoke20：

```powershell
uv run comm-necessary-cli inspect-experiment --experiment configs/comm_necessary/experiments/hotpotqa_split_evidence_v1.toml --backbone dashscope/qwen-turbo-1101
uv run comm-necessary-cli run --experiment configs/comm_necessary/experiments/hotpotqa_split_evidence_v1.toml --phase smoke20 --backbone dashscope/qwen-turbo-1101
uv run comm-necessary-cli validate-run --run-dir <run_dir>
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
- SPARC：
  `local/runs/sparc/`
  `cache/sparc_requests.sqlite`
  `local/reports/sparc/`
- DALA-lite：
  `local/runs/budget_comm/`
  `cache/budget_comm_requests.sqlite`
  `local/reports/budget_comm/`
- SID-lite：
  `local/runs/sid_lite/`
  `cache/sid_lite_requests.sqlite`
  `local/reports/sid_lite/`
- Free-MAD-lite：
  `local/runs/free_mad_lite/`
  `cache/free_mad_lite_requests.sqlite`
  `local/reports/free_mad_lite/`
- HotpotQA 通信必要性：
  `local/runs/comm_necessary/`
  `cache/comm_necessary_requests.sqlite`
  `local/reports/comm_necessary/`

## 开发约束

- `experiment_core` 是唯一共享层
- 实验包之间禁止互相导入
- 共享配置统一放在 `configs/shared/`
- 单智能体配置统一放在 `configs/single_agent/`
## Local Ollama Smoke Run

Use the local provider `local_ollama/qwen3:4b` to run the single-agent smoke experiment on a laptop GPU with the v4 minimal JSON contract.

1. Install Ollama for Windows from the official installer and confirm `ollama --version`.
2. Pull the base model:

```powershell
ollama pull qwen3:4b
```

3. Ensure `.env.local` contains `OLLAMA_API_KEY=ollama`.
4. Inspect the resolved config:

```powershell
uv run single-agent-cli inspect-experiment --experiment configs/single_agent/experiments/local_ollama_smoke.toml --model local_ollama/qwen3:4b
```

5. Run the smoke split:

```powershell
uv run single-agent-cli run --experiment configs/single_agent/experiments/local_ollama_smoke.toml --phase smoke20 --model local_ollama/qwen3:4b --cache-path cache/single_agent_local_ollama.sqlite
```

6. Validate the run:

```powershell
uv run single-agent-cli validate-run --run-dir <run_dir>
```
