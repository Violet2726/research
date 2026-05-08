# API 基线实验操作手册

## 1. 项目定位

本项目用于复现和比较 API 模型上的基础推理方法，当前重点覆盖：

- 单次 CoT
- Self-Consistency
- Majority Vote

项目强调三件事：

- 配置链清晰
- 运行结果可复现
- 模型可替换，但实验记录不混乱

## 2. 当前配置结构

当前统一配置链如下：

`命令行 --model -> experiment -> methods -> model catalog -> provider -> runner`

具体目录职责：

- `configs/providers/`
  供应商配置，只放连接参数和默认请求参数
- `configs/model_catalog.toml`
  已知常用模型的元信息目录，包含标签与少量覆盖项
- `configs/methods/`
  共享方法目录
- `configs/experiments/`
  与具体模型解耦的实验规格
- `configs/benchmarks/`
  benchmark 配置
- `configs/benchmarks/splits/`
  冻结后的 split 清单
- `configs/rosters/`
  未来多智能体角色配置预留层，当前 runner 不读取

## 3. 运行原则

- 运行实验时必须显式传入 `--model provider/model`
- provider 文件只放公共连接参数，不再承担模型目录职责
- model catalog 只维护“已知常用模型”，不是唯一可运行模型名单
- 未登记模型也允许直接运行
- 未登记模型默认没有 tags，也没有 model-specific override
- 带标签约束的实验会在发请求前 fail-fast，避免浪费 API 配额

## 4. 常用命令

安装依赖：

```powershell
uv sync
```

准备环境变量：

```powershell
Copy-Item .env.example .env.local
```

查看已登记模型：

```powershell
uv run baseline-cli list-models
```

生成冻结 split：

```powershell
uv run baseline-cli generate-splits
```

查看实验规格：

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml
```

查看某个模型在实验中的最终解析结果：

```powershell
uv run baseline-cli inspect-experiment --experiment configs/experiments/main-baselines.toml --model dashscope/qwen-turbo-1101
```

运行主基线 smoke：

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20 --model dashscope/qwen-turbo-1101
```

运行数学专项 pilot：

```powershell
uv run baseline-cli run --experiment configs/experiments/main_table_same_context.toml --phase pilot100 --model dashscope/qwen-turbo-1101
```

运行稳健性实验：

```powershell
uv run baseline-cli run --experiment configs/experiments/robustness.toml --phase pilot100 --model dashscope/qwen-turbo-1101
```

## 5. 如何新增模型

如果新模型只需要继承 provider 默认参数，可以直接运行：

```powershell
uv run baseline-cli run --experiment configs/experiments/main-baselines.toml --phase smoke20 --model dashscope/qwen-turbo-1101
```

如果希望这个模型：

- 出现在 `list-models`
- 带有标签
- 带有模型级覆盖参数

则把它补入 `configs/model_catalog.toml`。

## 6. 如何新增实验

新增 experiment 时，建议按下面顺序进行：

1. 选择一个 method catalog
2. 指定 benchmark 配置
3. 配置 phases
4. 只在必要时声明：
   - `required_model_tags`
   - `benchmark_required_tags`

不要在 experiment 里写死具体模型名。

## 7. 运行产物说明

每次运行都会生成 `local/runs/<run_id>/`，通常包含：

- `manifest.json`
  本次运行的最终配置与模型快照
- `raw_responses.jsonl`
  原始 API 调用日志
- `predictions.jsonl`
  题级预测
- `metrics.json`
  汇总指标
- `run_summary.json`
  精简摘要
- `budget_fairness.json`
  SC 与 MV 的预算公平性检查
- `paper_tables.md`
  导出的论文表格
- `run_validation.json`
  严格校验结果
- `progress.json`
  实时进度

全局文件包括：

- `cache/providers/<provider>/<request_model>/requests.sqlite`
- `reports/leaderboard.csv`

## 8. 保持项目整洁的约定

- 旧配置入口废弃后要及时删除，不保留平行旧逻辑
- provider 只负责连接，不再混入模型目录职责
- experiment 保持模型无关，不回退到旧的写死模型方式
- 通用方法统一放进 `configs/methods/`
- 多智能体角色配置未来单独进入 `configs/rosters/`
