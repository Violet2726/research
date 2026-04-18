# API 基线实验项目

这是一个面向研究复现与方法对比的 API 基线实验框架，当前覆盖：

- 单次 Chain-of-Thought（`cot`）
- Self-Consistency（`sc`）
- Majority Vote（`mv`）

项目重点是把“配置解析、模型调用、结果聚合、报告导出”串成一条清晰且可复现的实验链，方便在探索阶段快速替换模型，同时保持实验记录可追溯。

## 配置链

当前项目采用统一配置链：

`CLI --model provider/model -> 实验规格 -> 方法目录 -> 模型目录 -> 供应商配置 -> runner`

这条链的含义是：

- 运行时必须显式传入 `--model provider/model`
- `configs/experiments/` 只描述实验规格，不绑定默认模型
- `configs/methods/` 维护可复用的方法定义
- `configs/model_catalog.toml` 维护已知常用模型的标签与少量覆盖项
- `configs/providers/` 只维护供应商级连接参数与默认请求参数

## 目录职责

- `src/api_baselines/`
  代码主体，包括配置解析、数据集加载、提示词构造、模型请求、评测、报告与校验
- `configs/providers/`
  供应商配置，例如 `dashscope`、`zhipu`
- `configs/model_catalog.toml`
  模型元信息目录，不是唯一可运行模型全集
- `configs/methods/`
  共享方法目录
- `configs/experiments/`
  实验规格
- `configs/benchmarks/`
  benchmark 配置与冻结 split
- `configs/rosters/`
  未来多智能体角色配置的预留目录，当前 runner 不读取
- `runs/`
  每次实验的输出目录
- `cache/`
  请求缓存

## 安装与准备

安装依赖：

```powershell
uv sync
```

准备本地环境变量：

```powershell
Copy-Item .env.example .env.local
```

然后在 `.env.local` 中填入需要的 API Key，例如：

- `DASHSCOPE_API_KEY`
- `ZHIPU_API_KEY`

## 常用命令

生成冻结后的 split：

```powershell
uv run baseline-cli generate-splits
```

查看已登记的常用模型：

```powershell
uv run baseline-cli list-models
```

查看实验规格：

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml
```

查看某个模型在该实验中的最终解析结果：

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml --model dashscope/qwen2.5-7b-instruct
```

运行一个实验 phase：

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20 --model dashscope/qwen2.5-7b-instruct
```

运行数学专项 pilot：

```powershell
uv run baseline-cli run --experiment configs/experiments/qwen2.5-math-pilot100.toml --phase pilot100 --model dashscope/qwen2.5-math-7b-instruct
```

导出论文表格：

```powershell
uv run baseline-cli export-paper-tables --run-dir runs/<run_id>
```

校验一次运行结果：

```powershell
uv run baseline-cli validate-run --run-dir runs/<run_id>
```

## 模型使用规则

- `--model provider/model` 是运行时唯一标准模型引用格式
- 未登记在 `configs/model_catalog.toml` 的新模型也可以直接运行
- 未登记模型默认没有 tags，也没有模型级覆盖项
- 只有当某个模型需要额外标签或特殊覆盖参数时，才建议把它补进模型目录

## 运行产物

每次运行都会写入 `runs/<run_id>/`，常见产物包括：

- `manifest.json`
  这次运行最终解析后的配置与模型信息
- `raw_responses.jsonl`
  原始请求日志
- `predictions.jsonl`
  题级预测结果
- `metrics.json`
  方法级汇总指标
- `run_summary.json`
  运行摘要
- `budget_fairness.json`
  预算公平性检查结果
- `paper_tables.md`
  论文草稿用表格
- `run_validation.json`
  运行后校验结果
- `progress.json`
  长运行中的实时进度快照

全局产物包括：

- `cache/requests.sqlite`
- `reports/leaderboard.csv`

## 约定与扩展

- 新实验优先新增 experiment 配置，而不是把模型写死在代码里
- 新方法优先补到 `configs/methods/`，避免 experiment 文件重复定义
- 新模型默认直接用 `--model provider/model` 运行；需要标签或覆盖参数时再进入模型目录
- 多智能体阶段将单独进入 `configs/rosters/`，不会污染当前单模型 baseline 语义
