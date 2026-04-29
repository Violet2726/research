# Research Experiments

统一共享核心层的研究实验仓库，当前包含以下实验线：

- `single_agent_baselines`：单智能体 CoT / Self-Consistency / Majority Vote 基线
- `multi_agent_baselines`：标准多智能体 debate 基线
- `selective_comm`：触发式通信与 early-exit 实验
- `sparc`：内容压缩、局部审计与 SPARC 主实验
- `budget_comm`：DALA-lite 风格的预算约束通信实验
- `sid_lite`：SID-lite 机制验证
- `free_mad_lite`：Free-MAD-lite 机制验证
- `comm_necessary`：HotpotQA split-context 通信必要性实验

## 项目结构

```text
src/
  experiment_core/         共享运行时、provider、缓存、评测、结构化输出与路径配置
  single_agent_baselines/  单智能体实验
  multi_agent_baselines/   多智能体实验
  selective_comm/          选择性通信实验
  sparc/                   SPARC 实验
  budget_comm/             预算约束通信实验
  sid_lite/                SID-lite 实验
  free_mad_lite/           Free-MAD-lite 实验
  comm_necessary/          通信必要性实验

configs/
  shared/                  benchmark、split、provider、model catalog
  */                       各实验线自己的 experiment / protocol / policy 配置

datasets/                  本地基准数据
files/                     研究笔记与汇总文档
local/                     默认报告输出目录，已被 .gitignore 忽略
cache/                     默认请求缓存目录，已被 .gitignore 忽略
runs/                      默认运行输出目录，可同时保留归档结果
tests/                     测试
docs/                      项目结构与约定说明
```

更详细的目录职责见 [docs/project_structure.md](/d:/user/research/docs/project_structure.md:1)。

## 安装

```powershell
uv sync
Copy-Item .env.example .env.local
```

在 `.env.local` 中填入需要的 API Key。只提交 `.env.example`，不要提交真实密钥。

## UTF-8 约定

- 仓库新增了 `.editorconfig`，统一要求文本文件使用 `UTF-8`、`LF`、末尾换行。
- Python 读写文本时统一显式使用 `encoding="utf-8"`。
- Windows PowerShell 如果出现中文显示乱码，建议先执行：

```powershell
[Console]::InputEncoding = [System.Text.UTF8Encoding]::new($false)
[Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false)
$env:PYTHONUTF8 = "1"
```

## 输出目录配置

默认输出现在统一由 `src/experiment_core/workspace.py` 管理：

- 运行目录：`runs/<experiment_kind>/`
- 报告目录：`local/reports/<experiment_kind>/`
- 请求缓存：`cache/<experiment_kind>_requests.sqlite`
- 通用资料目录：`files/`

可以用以下环境变量全局覆盖：

- `RESEARCH_LOCAL_ROOT`
- `RESEARCH_RUNS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_FILES_ROOT`

例如：

```powershell
$env:RESEARCH_LOCAL_ROOT = "artifacts"
$env:RESEARCH_RUNS_ROOT = "artifacts/runs"
$env:RESEARCH_CACHE_ROOT = "artifacts/cache"
uv run selective-cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml
```

说明：

- `local/` 与 `cache/` 已被 `.gitignore` 忽略，适合本地报告与缓存。
- 根目录下的 `runs/` 现在也是默认运行目录；如需重定向请覆盖 `RESEARCH_RUNS_ROOT`。

## 常用命令

生成冻结 split：

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
uv run single-agent-cli run --experiment configs/single_agent/experiments/main-baselines.toml --phase smoke20 --model deepseek/deepseek-v4-flash
```

查看多智能体实验配置：

```powershell
uv run mad-cli inspect-experiment --experiment configs/multi_agent/experiments/vanilla_mad_minimal.toml --backbone deepseek/deepseek-v4-flash
```

运行多智能体实验：

```powershell
uv run mad-cli run --experiment configs/multi_agent/experiments/vanilla_mad_minimal.toml --phase smoke20 --backbone deepseek/deepseek-v4-flash
```

查看选择性通信实验配置：

```powershell
uv run selective-cli inspect-experiment --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml
```

运行选择性通信实验：

```powershell
uv run selective-cli run --experiment configs/selective_comm/experiments/trigger_early_exit_v1.toml --phase smoke20
```

其他实验线的命令入口：

- `sparc-cli`
- `budget-cli`
- `sid-lite-cli`
- `free-mad-lite-cli`
- `comm-necessary-cli`

各实验线的专属说明位于：

- [src/single_agent_baselines/README.md](/d:/user/research/src/single_agent_baselines/README.md:1)
- [src/multi_agent_baselines/README.md](/d:/user/research/src/multi_agent_baselines/README.md:1)
- [src/selective_comm/README.md](/d:/user/research/src/selective_comm/README.md:1)
- [src/sparc/README.md](/d:/user/research/src/sparc/README.md:1)
- [src/budget_comm/README.md](/d:/user/research/src/budget_comm/README.md:1)

## 开发约束

- `experiment_core` 是唯一共享层。
- 不同实验包之间禁止互相导入。
- 共享 benchmark / provider / model catalog 统一放在 `configs/shared/`。
- 默认输出应写入 `runs/`、`local/`、`cache/`、`files/` 这些统一工作目录，不要继续散落硬编码路径。
