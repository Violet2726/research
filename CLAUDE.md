# Research Experiments — Claude Code 指令

## 项目概述

统一的研究实验仓库，覆盖单智能体、多智能体辩论、选择性通信、预算约束通信等实验线。核心目标：复现和推进 LLM-as-a-Judge、Multi-Agent Debate 等方向的论文。

## 技术栈

- Python 3.12 + uv（包管理和运行）
- pytest（测试）
- TOML（配置）
- HuggingFace（数据集和归档）

## 常用命令

```powershell
# 安装依赖
uv sync --group dev

# 运行测试
uv run python -m pytest tests/ -x -q

# 运行实验
uv run research_cli family <family> run --experiment <config.toml> --phase <phase>

# 查看实验配置
uv run research_cli family <family> inspect-experiment --experiment <config.toml>

# 渲染报告
uv run research_cli family <family> render-report --run-dir <run_dir>

# 数据集资产
uv run research_cli tools dataset-assets prepare-used
```

## 架构约定

- 共享能力只放 `src/research_experiments/core/`
- family 之间不直接互相导入
- 配置字段统一用 `primary_model_ref`
- 工作区统一在 `local/`（runs/reports/cache/datasets）
- 文本文件统一 UTF-8，中文注释和文档

## 目录结构

```
src/research_experiments/
  core/           # 共享基础设施（provider/缓存/数据集/评测/限流）
  families/       # 各实验家族（consensagent/dmad/madjudge/...）
  matrix/         # faithful matrix 编排
  reporting/      # 报告与论文包
  workspace/      # 工作区布局与归档

configs/
  core/shared/    # benchmark/provider/model 共享配置
  families/       # 各实验线配置

tests/            # 自动化测试
local/            # 本地工作区（不提交）
files/            # 研究资料
```

## 实验家族

当前有 14 个实验家族：
- `single_agent`：CoT/Self-Consistency 基线
- `multi_agent`：标准 debate vs vote
- `consensagent`：基于触发机制的反谄媚多智能体辩论
- `dmad`：策略异质化多智能体辩论
- `madjudge`：自适应多智能体辩论（Beta-Binomial + KS 检验）
- `imad`：自适应停止复现
- `selective_comm`：trigger / early-exit 选择性通信
- `budget_comm`：预算约束通信与分配策略
- `sid_lite`：SID-lite 机制验证
- `free_mad_lite`：Free-MAD-lite 机制验证
- `comm_necessary`：HotpotQA split-context 通信必要性
- `colmad`：协作监督协议复现
- `econ`：低通信协调论文复现
- `macnet`：拓扑协作论文复现

## 编码规范

- Python 文本 I/O 显式写 `encoding="utf-8"`
- 中文注释、docstring 和文档
- 测试用 `uv run python -m pytest tests/ -x -q`
- 不提交 `local/`、`.env.local`、`__pycache__/`

## 环境变量

- `XIAOMI_MIMO_API_KEY`：MIMO 模型 API
- `HF_TOKEN`：HuggingFace 归档
- `RESEARCH_RUNS_ROOT` / `RESEARCH_CACHE_ROOT` 等：工作区覆盖

## 提交规范

- 中文 commit message
- 不提交密钥和本地产物
- 测试必须通过后再提交

## 代码质量

- Lint: `uv run ruff check src tests`
- Format: `uv run ruff format src tests`
- 详细贡献规范见 [CONTRIBUTING.md](CONTRIBUTING.md)
