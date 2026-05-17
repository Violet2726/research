# Hugging Face 操作总览

本文档是仓库内所有 Hugging Face 归档与恢复操作的统一索引，覆盖：

- 单个 run 目录
- 多个指定 run 目录
- 整个 `local/runs` 工作区
- 整个 `local/cache` 最新快照
- 一个或多个指定 cache 分库目录

适用对象：

- `Violet1307/research-runs`：公开 dataset repo，保存正式 run 归档
- `Violet1307/research-cache`：私有 dataset repo，保存 latest-only cache 快照

## 基本约定

- `runs` 的同步单位是“run 目录”
  - 标准 run：`local/runs/<family>/<experiment>/<phase>/<run_id>/`
  - matrix run：`local/runs/<matrix_kind>/<run_id>/`
- `cache` 的同步单位是“分库目录”
  - `local/cache/providers/<provider>/<request_model>/<dataset_path_key>/`
- `runs` 推送时会自动打包，可浏览外壳文件会单独保留，重型 JSONL / 预测文件会进入：
  - `traces.tar.zst`
  - `predictions.tar.zst`
  - `artifacts.tar.zst`
- `runs` 拉取时会自动完成全部归档解压
- `cache` 推送时会压缩为 `requests.sqlite.zst`，并附带 `metadata.json`、`sha256.txt`
- `cache` push/pull 会按 shard 级快照 hash 做增量判断；未变化的分库不会重复上传、下载或覆盖

## 一、runs：支持的操作

### 1. 推送单个 run 目录

```powershell
uv run research_cli tools archive-runs publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
```

特点：

- 支持某个具体的本地 run 文件夹
- 推送前自动打包，不需要先显式执行打包命令
- 上传内容包含：
  - `report.md`
  - `metrics.json`
  - `figure_manifest.json`
  - `figures/*.svg`
  - `figures/*.csv`
  - `archive_manifest.json`
  - `traces.tar.zst`
  - `predictions.tar.zst`
  - `artifacts.tar.zst`（若存在）

### 2. 拉取单个 run 目录

按 `run_id`：

```powershell
uv run research_cli tools archive-runs fetch-run --run-id <run_id>
```

按远端目录前缀：

```powershell
uv run research_cli tools archive-runs fetch-run --run-prefix <family>/<experiment>/<phase>/<run_id>
```

说明：

- `--run-id` 适合快速按运行标识回拉
- `--run-prefix` 适合精确指定某个远端文件夹
- 拉取后会自动解压该 run 的全部归档包

## 二、cache：支持的操作

### 1. 推送整个 cache 最新快照

```powershell
uv run research_cli tools cache-archive push-latest --cache-root local/cache
```

### 2. 拉取整个 cache 最新快照

```powershell
uv run research_cli tools cache-archive pull-latest --target local/cache
```

### 3. 推送一个或多个指定 cache 分库目录

单个：

```powershell
uv run research_cli tools cache-archive push-latest --cache-root local/cache --cache-shard providers/xiaomimimo/mimo-v2-5/strategyqa/dev
```

多个：

```powershell
uv run research_cli tools cache-archive push-latest --cache-root local/cache `
  --cache-shard providers/xiaomimimo/mimo-v2-5/strategyqa/dev `
  --cache-shard providers/xiaomimimo/mimo-v2-5/hotpotqa/validation-distractor
```

### 4. 拉取一个或多个指定 cache 分库目录

单个：

```powershell
uv run research_cli tools cache-archive pull-latest --target local/cache --cache-shard providers/xiaomimimo/mimo-v2-5/strategyqa/dev
```

多个：

```powershell
uv run research_cli tools cache-archive pull-latest --target local/cache `
  --cache-shard providers/xiaomimimo/mimo-v2-5/strategyqa/dev `
  --cache-shard providers/xiaomimimo/mimo-v2-5/hotpotqa/validation-distractor
```

说明：

- `--cache-shard` 支持重复传入
- 可以指定到某个 dataset 目录，也可以指定到更高层前缀
  - 例如 `providers/xiaomimimo/mimo-v2-5`

## 三、workspace：支持的操作

### 1. 查看本地与远端同步状态

```powershell
uv run research_cli tools hf-sync status
```

### 2. 推送整个工作区

```powershell
uv run research_cli tools hf-sync push-workspace
```

说明：

- 会扫描 `local/runs`
- 只推送验证通过的标准 run
- 会推送已经完整收敛的矩阵目录，例如 `faithful_matrix` 与 `reproduction_matrix`
- 默认也会同步 `local/cache`

### 3. 只推送某个或某些具体 run 文件夹

单个：

```powershell
uv run research_cli tools hf-sync push-workspace --skip-cache --run-dir local/runs/single_agent/same_context_main_table/count20/<run_id>
```

多个：

```powershell
uv run research_cli tools hf-sync push-workspace --skip-cache `
  --run-dir local/runs/single_agent/same_context_main_table/count20/<run_id> `
  --run-dir local/runs/selective_comm/trigger_early_exit_main/count20/<run_id>
```

### 4. 只拉取某个或某些具体 run 文件夹

按 `run_id`：

```powershell
uv run research_cli tools hf-sync pull-workspace --skip-cache --run-id <run_id_a> --run-id <run_id_b>
```

按远端目录前缀：

```powershell
uv run research_cli tools hf-sync pull-workspace --skip-cache `
  --run-prefix single_agent/same_context_main_table/count20/<run_id> `
  --run-prefix selective_comm/trigger_early_exit_main/count20/<run_id>
```

混合指定也支持：

```powershell
uv run research_cli tools hf-sync pull-workspace --skip-cache `
  --run-id <run_id_a> `
  --run-prefix selective_comm/trigger_early_exit_main/count20/<run_id>
```

### 5. 同时限制 runs 与 cache 的同步范围

```powershell
uv run research_cli tools hf-sync push-workspace `
  --run-dir local/runs/comm_necessary/hotpotqa_split_context_communication_necessity/count300/<run_id> `
  --cache-shard providers/xiaomimimo/mimo-v2-5/hotpotqa/validation-distractor
```

```powershell
uv run research_cli tools hf-sync pull-workspace `
  --run-prefix comm_necessary/hotpotqa_split_context_communication_necessity/count300/<run_id> `
  --cache-shard providers/xiaomimimo/mimo-v2-5/hotpotqa/validation-distractor
```

### 6. 其他常用控制项

- `--skip-runs`
  - 只同步 cache
- `--skip-cache`
  - 只同步 runs
- `--force-runs`
  - 即使本地已有匹配的 `hf_publish.json`，也重新发布
- `--no-matrix`
  - 批量扫描时跳过所有矩阵目录，例如 `faithful_matrix` 与 `reproduction_matrix`
- `--keep-existing-runs`
  - 回拉 runs 时不先删除本地同名目录

## 四、当前不支持的操作

当前项目不支持以下粒度：

- 直接推送任意单文件，例如某个 `report.md`
- 直接推送 `figures/` 下某一个单独文件
- 只推送 run 目录中的某几个零散 JSON 文件
- 只拉取某个 run 的部分归档组而不解压全部内容
- 把任意不符合 run/cache 语义的普通目录直接当成 Hugging Face 归档对象

也就是说，当前支持的是：

- run 目录级
- 多个 run 目录级
- cache 分库目录级
- 整个 workspace 级

## 五、推荐环境变量

```text
RESEARCH_RUNS_ROOT=local/runs
RESEARCH_REPORTS_ROOT=local/reports
RESEARCH_CACHE_ROOT=local/cache

RESEARCH_RUNS_HF_REPO=Violet1307/research-runs
RESEARCH_CACHE_HF_REPO=Violet1307/research-cache
RESEARCH_AUTO_PUBLISH_RUNS=1
RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1
HF_TOKEN=hf_xxx
```

## 六、建议的实际使用方式

- 单个 run：优先用 `research_cli tools archive-runs`
- 单个或多个 cache 分库：优先用 `research_cli tools cache-archive`
- 某些指定 run 目录与 cache 分库一起同步：用 `research_cli tools hf-sync`
- 全量迁移或全量回拉：用 `research_cli tools hf-sync push-workspace / pull-workspace`
