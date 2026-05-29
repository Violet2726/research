# datasets

这个目录不再承载正式数据文件。

项目的数据集资产现在统一放在本地工作区 `local/datasets/`，并通过 `research_cli tools dataset-assets` 一键恢复公开可下载的数据文件与冻结 split。

这组命令只负责数据资产、split 与盘点文档，不负责外部后端服务、数据库、模型访问凭证或 family 运行时依赖的安装配置。

这样做的目标是：

- 不把上游原始数据作为 Git 大文件长期提交
- 保持 benchmark split 可复现
- 把合规说明、恢复命令和本地资产边界写清楚

## 层级约束

- `configs/core/shared/benchmarks/` 下的 benchmark 配置必须镜像 `local/datasets/` 的相对路径层级，并使用“去掉数据文件扩展名后的路径”作为配置路径。
- `local/cache/providers/<provider>/<model>/...` 下的数据集缓存分片必须使用同一套层级键，避免把方法名或实验线名写成 dataset shard 名。
- 示例：`local/datasets/gsm8k/test.jsonl` 对应 `configs/core/shared/benchmarks/gsm8k/test.toml` 与 `local/cache/providers/<provider>/<model>/gsm8k/test/requests.sqlite`。

## 当前本地资产根目录

- 默认路径：`local/datasets`
- 环境变量覆盖：`RESEARCH_DATASETS_ROOT`

## 一键恢复

恢复主评测源、公开可下载的运行必需补充资产，并重建 split：

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

说明：

- `prepare-used` 会下载当前项目实际用到的主评测数据资产，以及公开可下载的运行必需补充资产，重建 frozen split，并刷新 `datasets/README.md` 与 `local/datasets/manifest.json`。
- `prepare-all-sources` 会额外下载公开可得的训练集、验证集与注释补充源，但仍只处理数据资产本身。
- 这些命令不会自动安装或启动外部后端，例如 Freebase/Virtuoso、SPARQL 服务或其他 family 专属运行时依赖。

## 主评测源文件

### CommonGen-Hard (`commongen_hard`)

- benchmark 配置：`configs/core/shared/benchmarks/commongen-hard/commongen_hard_nohuman.toml`
- 数据相对路径：`commongen-hard/commongen_hard_nohuman.json`
- 本地资产：`local/datasets/commongen-hard/commongen_hard_nohuman.json`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/allenai/commongen_hard`
- 上游 split：`test`
- 样本数：`400`
- 文件大小：`372.47 KiB`
- 冻结 split：`count100/commongen-hard/commongen_hard_nohuman-seed42.json`, `count20/commongen-hard/commongen_hard_nohuman-seed42.json`, `count300/commongen-hard/commongen_hard_nohuman-seed42.json`, `full/commongen-hard/commongen_hard_nohuman-seed42.json`
- 说明：公开 no-human 版本；本地主指标采用稳定的 concept coverage 代理。

### MATH (`competition_math`)

- benchmark 配置：`configs/core/shared/benchmarks/competition_math/MATH.toml`
- 数据相对路径：`competition_math/MATH.zip`
- 本地资产：`local/datasets/competition_math/MATH.zip`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hendrycks/competition_math`
- 上游 split：`test`
- 样本数：`5000`
- 文件大小：`2.72 MiB`
- 冻结 split：`count100/competition_math/MATH-seed0.json`, `count20/competition_math/MATH-seed0.json`, `full/competition_math/MATH-seed0.json`
- 说明：与 Hendrycks MATH 论文主结果口径对齐的官方压缩包；运行时按 `MATH/test/<subject>/*.json` 读取。

### GPQA Diamond (`gpqa_diamond`)

- benchmark 配置：`configs/core/shared/benchmarks/gpqa/dataset.toml`
- 数据相对路径：`gpqa/dataset.zip`
- 本地资产：`local/datasets/gpqa/dataset.zip`
- 上游来源：GitHub raw，`https://github.com/idavidrein/gpqa/raw/main/dataset.zip`
- 上游 split：`diamond`
- 样本数：`198`
- 文件大小：`2.24 MiB`
- 冻结 split：`count100/gpqa/dataset-seed42.json`, `count20/gpqa/dataset-seed42.json`, `full/gpqa/dataset-seed42.json`
- 说明：官方 zip 同时内嵌 gpqa_main、gpqa_experts 与 gpqa_extended；请遵循上游许可与使用说明。

### GSM8K (`gsm8k`)

- benchmark 配置：`configs/core/shared/benchmarks/gsm8k/test.toml`
- 数据相对路径：`gsm8k/test.jsonl`
- 本地资产：`local/datasets/gsm8k/test.jsonl`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl`
- 上游 split：`test`
- 样本数：`1319`
- 文件大小：`732.17 KiB`
- 冻结 split：`count100/gsm8k/test-seed42.json`, `count20/gsm8k/test-seed42.json`, `count300/gsm8k/test-seed42.json`, `count500/gsm8k/test-seed42.json`, `full/gsm8k/test-seed42.json`

### HotpotQA (`hotpotqa`)

- benchmark 配置：`configs/core/shared/benchmarks/hotpotqa/validation_distractor.toml`
- 数据相对路径：`hotpotqa/validation_distractor.parquet`
- 本地资产：`local/datasets/hotpotqa/validation_distractor.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`validation_distractor`
- 样本数：`7405`
- 文件大小：`26.18 MiB`
- 冻结 split：`count100/hotpotqa/validation_distractor-seed42.json`, `count20/hotpotqa/validation_distractor-seed42.json`, `count300/hotpotqa/validation_distractor-seed42.json`, `count500/hotpotqa/validation_distractor-seed42.json`, `full/hotpotqa/validation_distractor-seed42.json`

### HumanEval (`humaneval`)

- benchmark 配置：`configs/core/shared/benchmarks/humaneval/test.toml`
- 数据相对路径：`humaneval/test.parquet`
- 本地资产：`local/datasets/humaneval/test.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/openai/openai_humaneval`
- 上游 split：`test`
- 样本数：`164`
- 文件大小：`81.95 KiB`
- 冻结 split：`count100/humaneval/test-seed42.json`, `count20/humaneval/test-seed42.json`, `count300/humaneval/test-seed42.json`, `full/humaneval/test-seed42.json`
- 说明：保留 prompt、entry point 与 tests，供 MacNet 本地 pass@1 评测使用。

### MATH500 (`math500`)

- benchmark 配置：`configs/core/shared/benchmarks/math500/test.toml`
- 数据相对路径：`math500/test.jsonl`
- 本地资产：`local/datasets/math500/test.jsonl`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/math-ai/Math-500`
- 上游 split：`test`
- 样本数：`500`
- 文件大小：`436.10 KiB`
- 冻结 split：`count100/math500/test-seed42.json`, `count20/math500/test-seed42.json`, `count300/math500/test-seed42.json`, `full/math500/test-seed42.json`
- 说明：官方公开数据仅提供 test 集。

### MMLU (`mmlu`)

- benchmark 配置：`configs/core/shared/benchmarks/mmlu/test.toml`
- 数据相对路径：`mmlu/test.parquet`
- 本地资产：`local/datasets/mmlu/test.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/cais/mmlu`
- 上游 split：`test`
- 样本数：`14042`
- 文件大小：`3.34 MiB`
- 冻结 split：`count100/mmlu/test-seed42.json`, `count20/mmlu/test-seed42.json`, `count300/mmlu/test-seed42.json`, `count500/mmlu/test-seed42.json`, `full/mmlu/test-seed42.json`
- 说明：使用原始 MMLU all/test 聚合 parquet，避免用 MMLU-Pro 代替 canonical 多选主线。

### MMLU Abstract Algebra (`mmlu_abstract_algebra`)

- benchmark 配置：`configs/core/shared/benchmarks/mmlu/abstract_algebra.toml`
- 数据相对路径：`mmlu/test.parquet`
- 本地资产：`local/datasets/mmlu/test.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/cais/mmlu`
- 上游 split：`test`
- 样本数：`100`
- 文件大小：`3.34 MiB`
- 冻结 split：`count100/mmlu/abstract_algebra-seed42.json`, `count20/mmlu/abstract_algebra-seed42.json`, `count300/mmlu/abstract_algebra-seed42.json`
- 说明：MMLU abstract_algebra 子集，复用 mmlu/test.parquet，运行时通过 record_filters 过滤。

### MMLU-Pro (`mmlu_pro`)

- benchmark 配置：`configs/core/shared/benchmarks/mmlu-pro/test.toml`
- 数据相对路径：`mmlu-pro/test.parquet`
- 本地资产：`local/datasets/mmlu-pro/test.parquet`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`
- 上游 split：`test`
- 样本数：`12032`
- 文件大小：`3.95 MiB`
- 冻结 split：`count100/mmlu-pro/test-seed42.json`, `count20/mmlu-pro/test-seed42.json`, `count300/mmlu-pro/test-seed42.json`, `count500/mmlu-pro/test-seed42.json`, `full/mmlu-pro/test-seed42.json`

### ReaLMistake Answerability Classification (`realmistake_answerability_classification`)

- benchmark 配置：`configs/core/shared/benchmarks/realmistake/answerability_classification.toml`
- 数据相对路径：`realmistake/answerability_classification`
- 本地资产：`local/datasets/realmistake/answerability_classification`
- 上游来源：ReaLMistake official GitHub，`https://raw.githubusercontent.com/psunlpgroup/ReaLMistake/main/data.zip`
- 上游 split：`answerability_classification`
- 样本数：`300`
- 文件大小：`4.00 KiB`
- 冻结 split：`count100/realmistake/answerability_classification-seed42.json`, `count20/realmistake/answerability_classification-seed42.json`, `count300/realmistake/answerability_classification-seed42.json`, `full/realmistake/answerability_classification-seed42.json`
- 说明：官方公开压缩包，密码为 `open-realmistake`；当前 benchmark 会直接从 zip 中读取 answerability task 的 GPT-4 与 Llama-2 两个 JSONL 分片。

### ReaLMistake Fine-grained Fact Verification (`realmistake_fine_grained_fact_verification`)

- benchmark 配置：`configs/core/shared/benchmarks/realmistake/fine_grained_fact_verification.toml`
- 数据相对路径：`realmistake/fine_grained_fact_verification`
- 本地资产：`local/datasets/realmistake/fine_grained_fact_verification`
- 上游来源：ReaLMistake official GitHub，`https://raw.githubusercontent.com/psunlpgroup/ReaLMistake/main/data.zip`
- 上游 split：`finegrained_fact_verification`
- 样本数：`300`
- 文件大小：`4.00 KiB`
- 冻结 split：`count100/realmistake/fine_grained_fact_verification-seed42.json`, `count20/realmistake/fine_grained_fact_verification-seed42.json`, `count300/realmistake/fine_grained_fact_verification-seed42.json`, `full/realmistake/fine_grained_fact_verification-seed42.json`
- 说明：官方公开压缩包，密码为 `open-realmistake`；当前 benchmark 会直接从 zip 中读取 fact verification task 的 GPT-4 与 Llama-2 两个 JSONL 分片。

### ReaLMistake Math Problem Generation (`realmistake_math_problem_generation`)

- benchmark 配置：`configs/core/shared/benchmarks/realmistake/math_problem_generation.toml`
- 数据相对路径：`realmistake/math_problem_generation`
- 本地资产：`local/datasets/realmistake/math_problem_generation`
- 上游来源：ReaLMistake official GitHub，`https://raw.githubusercontent.com/psunlpgroup/ReaLMistake/main/data.zip`
- 上游 split：`math_word_problem_generation`
- 样本数：`300`
- 文件大小：`4.00 KiB`
- 冻结 split：`count100/realmistake/math_problem_generation-seed42.json`, `count20/realmistake/math_problem_generation-seed42.json`, `count300/realmistake/math_problem_generation-seed42.json`, `full/realmistake/math_problem_generation-seed42.json`
- 说明：官方公开压缩包，密码为 `open-realmistake`；当前 benchmark 会直接从 zip 中读取 math task 的 GPT-4 与 Llama-2 两个 JSONL 分片。

### StrategyQA (`strategyqa`)

- benchmark 配置：`configs/core/shared/benchmarks/strategyqa/dev.toml`
- 数据相对路径：`strategyqa/dev.json`
- 本地资产：`local/datasets/strategyqa/dev.json`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/dev.json`
- 上游 split：`dev`
- 样本数：`229`
- 文件大小：`414.25 KiB`
- 冻结 split：`count100/strategyqa/dev-seed42.json`, `count20/strategyqa/dev-seed42.json`, `full/strategyqa/dev-seed42.json`

### WebQuestions (`webquestions`)

- benchmark 配置：`configs/core/shared/benchmarks/webquestions/test.toml`
- 数据相对路径：`webquestions/test.json`
- 本地资产：`local/datasets/webquestions/test.json`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/main/test.json`
- 上游 split：`test`
- 样本数：`2032`
- 文件大小：`267.79 KiB`
- 冻结 split：`count100/webquestions/test-seed42.json`, `count20/webquestions/test-seed42.json`, `count300/webquestions/test-seed42.json`, `count500/webquestions/test-seed42.json`, `full/webquestions/test-seed42.json`
- 说明：主文件只包含问题与答案。若要恢复更完整的图注释，请再下载 supplementary 里的 Freebase 路径与实体链接文件。

## 训练集与补充上游 split

### MacNet SRDD Profile / srdd_profile_repo

- 本地资产：`local/datasets/macnet/srdd-profile-repo.zip`
- 用途：`profile_bank`
- 上游来源：MacNet official GitHub archive，`https://github.com/OpenBMB/ChatDev/archive/refs/heads/macnet.zip`
- 上游 split：`shared`
- 文件大小：`8.62 MiB`
- 说明：包含官方 SRDD_Profile 目录；MacNet family 会直接从 zip 中读取角色 profile 文本。

### GSM8K / train

- 本地资产：`local/datasets/gsm8k/train.jsonl`
- 用途：`train`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/train.jsonl`
- 上游 split：`train`
- 文件大小：尚未下载

### HotpotQA / train_shard_0

- 本地资产：`local/datasets/hotpotqa/distractor/train-00000-of-00002.parquet`
- 用途：`train`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`distractor_train`
- 文件大小：尚未下载

### HotpotQA / train_shard_1

- 本地资产：`local/datasets/hotpotqa/distractor/train-00001-of-00002.parquet`
- 用途：`train`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/hotpotqa/hotpot_qa`
- 上游 split：`distractor_train`
- 文件大小：尚未下载

### MMLU / validation

- 本地资产：`local/datasets/mmlu/validation.parquet`
- 用途：`validation`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/cais/mmlu`
- 上游 split：`all_validation`
- 文件大小：尚未下载
- 说明：供 MacNet 的 topic 诊断与离线抽样使用。

### MMLU-Pro / validation

- 本地资产：`local/datasets/mmlu-pro/validation.parquet`
- 用途：`validation`
- 上游来源：Hugging Face dataset，`https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro`
- 上游 split：`validation`
- 文件大小：尚未下载
- 说明：官方未公开 train split；这里只保留 validation 作为唯一非测试补充源。

### StrategyQA / train

- 本地资产：`local/datasets/strategyqa/train.json`
- 用途：`train`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/eladsegal/strategyqa/main/data/strategyqa/train.json`
- 上游 split：`train`
- 文件大小：尚未下载

### WebQuestions / question_dump_test

- 本地资产：`local/datasets/webquestions/question_dump_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-dump/test.json`
- 上游 split：`test`
- 文件大小：`3.67 MiB`
- 样本数：`2032`
- 说明：YodaQA 生成的问题概念、clue 与词汇注释，可用于更稳的 topic seed 与角色提示。

### WebQuestions / freebase_key_test

- 本地资产：`local/datasets/webquestions/freebase_key_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase/test.json`
- 上游 split：`test`
- 文件大小：`107.51 KiB`
- 样本数：`2032`
- 说明：官方提供的单实体 Freebase key 注释，可作为 topic seed。

### WebQuestions / freebase_mids_test

- 本地资产：`local/datasets/webquestions/freebase_mids_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-mids/test.json`
- 上游 split：`test`
- 文件大小：`412.31 KiB`
- 样本数：`2032`
- 说明：问题概念到 Freebase MID 的链接结果。

### WebQuestions / relation_paths_test

- 本地资产：`local/datasets/webquestions/relation_paths_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-rp/test.json`
- 上游 split：`test`
- 文件大小：`254.90 KiB`
- 样本数：`2032`
- 说明：官方发布的关系路径注释，是 v1 WebQuestions 图视角的主要证据源。

### WebQuestions / branched_relation_paths_test

- 本地资产：`local/datasets/webquestions/branched_relation_paths_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-freebase-brp/test.json`
- 上游 split：`test`
- 文件大小：`281.97 KiB`
- 样本数：`2032`
- 说明：Branched relation paths 是 relation-path 视角的更丰富补充；若存在，loader 会优先使用它。

### WebQuestions / entities_test

- 本地资产：`local/datasets/webquestions/entities_test.json`
- 用途：`annotation`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/d-entities/test.json`
- 上游 split：`test`
- 文件大小：`227.19 KiB`
- 样本数：`2032`
- 说明：问题实体识别结果，用于 neighborhood 视角构造。

## 未公开或不建议镜像的补充源

- `gpqa_diamond`：官方未提供独立 train split；补充题型已内嵌在 dataset.zip 中，不建议额外镜像分发。
- `math500`：官方公开版本只提供 test 集。

## 合规与治理说明

- 本仓库只保留下载逻辑、冻结 split 和说明文档，不默认镜像上游原始数据。
- 具体使用时请遵循各数据集上游仓库、自带 license 文件以及发布页面的约束。
- `gpqa_diamond` 这种带官方压缩包与密码的资产，尤其应以原始发布方说明为准，不建议再公开二次分发。
- 本地盘点清单位于 `local/datasets/manifest.json`，用于脚本读取，不进入 Git 主线。
