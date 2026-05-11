# datasets

这个目录不再承载正式数据文件。

项目的数据集资产现在统一放在本地工作区 `local/datasets/`，并通过 `research_cli tools dataset-assets` 一键恢复。

这样做的目标是：

- 不把上游原始数据作为 Git 大文件长期提交
- 保持 benchmark split 可复现
- 把合规说明、恢复命令和本地资产边界写清楚

## 当前本地资产根目录

- 默认路径：`local/datasets`
- 环境变量覆盖：`RESEARCH_DATASETS_ROOT`

## 一键恢复

只恢复主评测源并重建 split：

```powershell
uv run research_cli tools dataset-assets prepare-used
```

同时恢复训练集与可公开下载的验证补充源：

```powershell
uv run research_cli tools dataset-assets prepare-all-sources
```

强制覆盖已有本地文件：

```powershell
uv run research_cli tools dataset-assets prepare-all-sources --force
```

## 主评测源文件

### GPQA Diamond (`gpqa_diamond`)

- 配置路径：`gpqa/dataset.zip`
- 本地资产：`local/datasets/gpqa/dataset.zip`
- 上游来源：GitHub raw，`https://github.com/idavidrein/gpqa/raw/main/dataset.zip`
- 上游 split：`diamond`
- 样本数：`198`
- 文件大小：`2.24 MiB`
- 冻结 split：`count100/gpqa_diamond-seed42.json`, `count20/gpqa_diamond-seed42.json`, `full/gpqa_diamond-seed42.json`
- 说明：官方 zip 同时内嵌 gpqa_main、gpqa_experts 与 gpqa_extended；请遵循上游许可与使用说明。

### GSM8K (`gsm8k`)

- 配置路径：`gsm8k/test.jsonl`
- 本地资产：`local/datasets/gsm8k/test.jsonl`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`
- 上游 split：`test`
- 样本数：`1319`
- 文件大小：`732.17 KiB`
- 冻结 split：`count100/gsm8k-seed42.json`, `count20/gsm8k-seed42.json`, `count300/gsm8k-seed42.json`, `count500/gsm8k-seed42.json`, `full/gsm8k-seed42.json`

### GSM-Symbolic (`gsm_symbolic`)

- 配置路径：`gsm-symbolic/GSM_symbolic.jsonl`
- 本地资产：`local/datasets/gsm-symbolic/GSM_symbolic.jsonl`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/apple/GSM-Symbolic/blob/main/main/test.jsonl`
- 上游 split：`test`
- 样本数：`5000`
- 文件大小：`5.94 MiB`
- 冻结 split：`count100/gsm_symbolic-seed42.json`, `count20/gsm_symbolic-seed42.json`, `count300/gsm_symbolic-seed42.json`, `count500/gsm_symbolic-seed42.json`, `full/gsm_symbolic-seed42.json`
- 说明：公开版本只提供生成后的 test 集。

### HotpotQA (`hotpotqa`)

- 配置路径：`hotpotqa/validation_distractor.parquet`
- 本地资产：`local/datasets/hotpotqa/validation_distractor.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`validation_distractor`
- 样本数：`7405`
- 文件大小：`26.18 MiB`
- 冻结 split：`count100/hotpotqa-seed42.json`, `count20/hotpotqa-seed42.json`, `count300/hotpotqa-seed42.json`, `count500/hotpotqa-seed42.json`, `full/hotpotqa-seed42.json`

### MATH500 (`math500`)

- 配置路径：`math500/test.jsonl`
- 本地资产：`local/datasets/math500/test.jsonl`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/math-ai/Math-500`
- 上游 split：`test`
- 样本数：`500`
- 文件大小：`436.10 KiB`
- 冻结 split：`count100/math500-seed42.json`, `count20/math500-seed42.json`, `count300/math500-seed42.json`, `full/math500-seed42.json`
- 说明：官方公开数据仅提供 test 集。

### MMLU-Pro (`mmlu_pro`)

- 配置路径：`mmlu-pro/test.parquet`
- 本地资产：`local/datasets/mmlu-pro/test.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`
- 上游 split：`test`
- 样本数：`12032`
- 文件大小：`3.95 MiB`
- 冻结 split：`count100/mmlu_pro-seed42.json`, `count20/mmlu_pro-seed42.json`, `count300/mmlu_pro-seed42.json`, `count500/mmlu_pro-seed42.json`, `full/mmlu_pro-seed42.json`

### StrategyQA (`strategyqa`)

- 配置路径：`strategyqa/dev.json`
- 本地资产：`local/datasets/strategyqa/dev.json`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/dev.json`
- 上游 split：`dev`
- 样本数：`229`
- 文件大小：`414.25 KiB`
- 冻结 split：`count100/strategyqa-seed42.json`, `count20/strategyqa-seed42.json`, `full/strategyqa-seed42.json`

## 训练集与补充上游 split

### GSM8K / train

- 本地资产：`local/datasets/gsm8k/train.jsonl`
- 用途：`train`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl`
- 上游 split：`train`
- 文件大小：`3.97 MiB`
- 样本数：`7473`

### HotpotQA / train_shard_0

- 本地资产：`local/datasets/hotpotqa/distractor/train-00000-of-00002.parquet`
- 用途：`train`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`distractor_train`
- 文件大小：`157.95 MiB`
- 样本数：`45224`

### HotpotQA / train_shard_1

- 本地资产：`local/datasets/hotpotqa/distractor/train-00001-of-00002.parquet`
- 用途：`train`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`distractor_train`
- 文件大小：`158.46 MiB`
- 样本数：`45223`

### MMLU-Pro / validation

- 本地资产：`local/datasets/mmlu-pro/validation.parquet`
- 用途：`validation`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`
- 上游 split：`validation`
- 文件大小：`41.85 KiB`
- 样本数：`70`
- 说明：官方未公开 train split；这里只保留 validation 作为唯一非测试补充源。

### StrategyQA / train

- 本地资产：`local/datasets/strategyqa/train.json`
- 用途：`train`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json`
- 上游 split：`train`
- 文件大小：`3.68 MiB`
- 样本数：`2061`

## 未公开或不建议镜像的补充源

- `gpqa_diamond`：官方未提供独立 train split；补充题型已内嵌在 dataset.zip 中，不建议额外镜像分发。
- `gsm_symbolic`：官方公开版本只提供生成后的 test 集。
- `math500`：官方公开版本只提供 test 集。

## 合规与治理说明

- 本仓库只保留下载逻辑、冻结 split 和说明文档，不默认镜像上游原始数据。
- 具体使用时请遵循各数据集上游仓库、自带 license 文件以及发布页面的约束。
- `gpqa_diamond` 这种带官方压缩包与密码的资产，尤其应以原始发布方说明为准，不建议再公开二次分发。
- 本地盘点清单位于 `local/datasets/manifest.json`，用于脚本读取，不进入 Git 主线。
