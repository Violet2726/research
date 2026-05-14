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

### DoG CWQ (`dog_cwq`)

- 配置路径：`dog-freebase/cwq.json`
- 本地资产：`local/datasets/dog-freebase/cwq.json`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/cwq.json`
- 上游 split：`test`
- 样本数：`3531`
- 文件大小：`3.66 MiB`
- 冻结 split：`count100/dog_cwq-seed42.json`, `count20/dog_cwq-seed42.json`, `count300/dog_cwq-seed42.json`, `full/dog_cwq-seed42.json`
- 说明：DoG 官方仓提供的 CWQ 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。

### DoG GrailQA (`dog_grailqa`)

- 配置路径：`dog-freebase/grailqa.json`
- 本地资产：`local/datasets/dog-freebase/grailqa.json`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/grailqa.json`
- 上游 split：`test`
- 样本数：`1000`
- 文件大小：`3.71 MiB`
- 冻结 split：`count100/dog_grailqa-seed42.json`, `count20/dog_grailqa-seed42.json`, `count300/dog_grailqa-seed42.json`, `full/dog_grailqa-seed42.json`
- 说明：DoG 官方仓提供的 GrailQA 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。

### DoG MetaQA 1-hop (`dog_metaqa_1hop`)

- 配置路径：`dog-metaqa/1-hop/qa_test.txt`
- 本地资产：`local/datasets/dog-metaqa/1-hop/qa_test.txt`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/1-hop/qa_test.txt`
- 上游 split：`test`
- 样本数：`9947`
- 文件大小：`659.40 KiB`
- 冻结 split：`count100/dog_metaqa_1hop-seed42.json`, `count20/dog_metaqa_1hop-seed42.json`, `count300/dog_metaqa_1hop-seed42.json`, `full/dog_metaqa_1hop-seed42.json`
- 说明：DoG 官方仓提供的 MetaQA 1-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。

### DoG MetaQA 2-hop (`dog_metaqa_2hop`)

- 配置路径：`dog-metaqa/2-hop/qa_test.txt`
- 本地资产：`local/datasets/dog-metaqa/2-hop/qa_test.txt`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/2-hop/qa_test.txt`
- 上游 split：`test`
- 样本数：`14872`
- 文件大小：`2.05 MiB`
- 冻结 split：`count100/dog_metaqa_2hop-seed42.json`, `count20/dog_metaqa_2hop-seed42.json`, `count300/dog_metaqa_2hop-seed42.json`, `full/dog_metaqa_2hop-seed42.json`
- 说明：DoG 官方仓提供的 MetaQA 2-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。

### DoG MetaQA 3-hop (`dog_metaqa_3hop`)

- 配置路径：`dog-metaqa/3-hop/qa_test.txt`
- 本地资产：`local/datasets/dog-metaqa/3-hop/qa_test.txt`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/3-hop/qa_test.txt`
- 上游 split：`test`
- 样本数：`14274`
- 文件大小：`3.17 MiB`
- 冻结 split：`count100/dog_metaqa_3hop-seed42.json`, `count20/dog_metaqa_3hop-seed42.json`, `count300/dog_metaqa_3hop-seed42.json`, `full/dog_metaqa_3hop-seed42.json`
- 说明：DoG 官方仓提供的 MetaQA 3-hop 测试集；运行时还需要共享的 `dog-metaqa/kb.txt` 图后端。

### DoG WebQSP (`dog_webqsp`)

- 配置路径：`dog-freebase/WebQSP.json`
- 本地资产：`local/datasets/dog-freebase/WebQSP.json`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/WebQSP.json`
- 上游 split：`test`
- 样本数：`1639`
- 文件大小：`6.21 MiB`
- 冻结 split：`count100/dog_webqsp-seed42.json`, `count20/dog_webqsp-seed42.json`, `count300/dog_webqsp-seed42.json`, `full/dog_webqsp-seed42.json`
- 说明：DoG 官方仓提供的 WebQSP 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。

### DoG WebQuestions (`dog_webquestions`)

- 配置路径：`dog-freebase/WebQuestions.json`
- 本地资产：`local/datasets/dog-freebase/WebQuestions.json`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/freebase/dataset/WebQuestions.json`
- 上游 split：`test`
- 样本数：`2032`
- 文件大小：`794.23 KiB`
- 冻结 split：`count100/dog_webquestions-seed42.json`, `count20/dog_webquestions-seed42.json`, `count300/dog_webquestions-seed42.json`, `full/dog_webquestions-seed42.json`
- 说明：DoG 官方仓提供的 WebQuestions 论文复现 JSON；真正运行仍需本地 Freebase/Virtuoso。

### GPQA Diamond (`gpqa_diamond`)

- 配置路径：`gpqa/dataset.zip`
- 本地资产：`local/datasets/gpqa/dataset.zip`
- 上游来源：GitHub raw，`https://github.com/idavidrein/gpqa/raw/main/dataset.zip`
- 上游 split：`diamond`
- 样本数：`198`
- 文件大小：`2.24 MiB`
- 冻结 split：`count100/gpqa_diamond-seed42.json`, `count20/gpqa_diamond-seed42.json`, `full/gpqa_diamond-seed42.json`
- 说明：官方 zip 同时内嵌 gpqa_main、gpqa_experts 与 gpqa_extended；请遵循上游许可与使用说明。

### GrailQA (`grailqa`)

- 配置路径：`grailqa/validation.parquet`
- 本地资产：`local/datasets/grailqa/validation.parquet`
- 上游来源：Hugging Face dataset mirror，`https://huggingface.co/datasets/Hieuman/grail_qa`
- 上游 split：`validation`
- 样本数：`6763`
- 文件大小：`2.76 MiB`
- 冻结 split：`count100/grailqa-seed42.json`, `count20/grailqa-seed42.json`, `count300/grailqa-seed42.json`, `count500/grailqa-seed42.json`, `full/grailqa-seed42.json`
- 说明：上游官方主页提供下载入口；这里使用 Hugging Face parquet 镜像，便于单文件恢复与本地 split 重建。

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

### WebQuestions (`webquestions`)

- 配置路径：`webquestions/test.json`
- 本地资产：`local/datasets/webquestions/test.json`
- 上游来源：GitHub raw，`https://raw.githubusercontent.com/brmson/dataset-factoid-webquestions/master/main/test.json`
- 上游 split：`test`
- 样本数：`2032`
- 文件大小：`267.79 KiB`
- 冻结 split：`count100/webquestions-seed42.json`, `count20/webquestions-seed42.json`, `count300/webquestions-seed42.json`, `count500/webquestions-seed42.json`, `full/webquestions-seed42.json`
- 说明：主文件只包含问题与答案。若要恢复更完整的图注释，请再下载 supplementary 里的 Freebase 路径与实体链接文件。

## 训练集与补充上游 split

### DoG MetaQA / kb

- 本地资产：`local/datasets/dog-metaqa/kb.txt`
- 用途：`backend`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt`
- 上游 split：`shared`
- 文件大小：`4.98 MiB`
- 说明：MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。

### DoG MetaQA / kb

- 本地资产：`local/datasets/dog-metaqa/kb.txt`
- 用途：`backend`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt`
- 上游 split：`shared`
- 文件大小：`4.98 MiB`
- 说明：MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。

### DoG MetaQA / kb

- 本地资产：`local/datasets/dog-metaqa/kb.txt`
- 用途：`backend`
- 上游来源：DoG official GitHub，`https://raw.githubusercontent.com/mira-ai-lab/DoG/main/KBQA_TASK/metaqa/dataset/kb.txt`
- 上游 split：`shared`
- 文件大小：`4.98 MiB`
- 说明：MetaQA 论文复现共享知识图谱后端，供 1/2/3-hop 三个 benchmark 共用。

### GrailQA / train

- 本地资产：`local/datasets/grailqa/train.parquet`
- 用途：`train`
- 上游来源：Hugging Face dataset mirror，`https://huggingface.co/datasets/Hieuman/grail_qa`
- 上游 split：`train`
- 文件大小：`22.58 MiB`
- 样本数：`44337`

### GrailQA / test

- 本地资产：`local/datasets/grailqa/test.parquet`
- 用途：`test_public`
- 上游来源：Hugging Face dataset mirror，`https://huggingface.co/datasets/Hieuman/grail_qa`
- 上游 split：`test`
- 文件大小：`646.37 KiB`
- 样本数：`13231`
- 说明：用于额外泛化检查；正式 family v1 默认仍以 validation split 进入 count20/count100/count300。

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

- `dog_cwq`：需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。
- `dog_grailqa`：需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。
- `dog_webqsp`：需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。
- `dog_webquestions`：需要用户自行准备本地 Freebase/Virtuoso 后端；官方仓只公开题目 JSON，不公开可直接运行的 Freebase 快照。
- `gpqa_diamond`：官方未提供独立 train split；补充题型已内嵌在 dataset.zip 中，不建议额外镜像分发。
- `gsm_symbolic`：官方公开版本只提供生成后的 test 集。
- `math500`：官方公开版本只提供 test 集。

## 合规与治理说明

- 本仓库只保留下载逻辑、冻结 split 和说明文档，不默认镜像上游原始数据。
- 具体使用时请遵循各数据集上游仓库、自带 license 文件以及发布页面的约束。
- `gpqa_diamond` 这种带官方压缩包与密码的资产，尤其应以原始发布方说明为准，不建议再公开二次分发。
- 本地盘点清单位于 `local/datasets/manifest.json`，用于脚本读取，不进入 Git 主线。
