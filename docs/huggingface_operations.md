# Hugging Face 操作总览

本文档描述当前仓库的 Hugging Face 同步体系。

## 基本约定

- `runs` 与 `cache` 继续使用两个独立的 Hugging Face dataset repo
- 默认行为统一为：
  - 不重复推送
  - 不重复拉取

## 一、统一命令入口

```powershell
uv run research_cli tools hf push-cache
uv run research_cli tools hf pull-cache
uv run research_cli tools hf push-runs
uv run research_cli tools hf pull-runs
```

## 二、cache：整库同步

### 1. 推送整个 cache

```powershell
uv run research_cli tools hf push-cache
```

### 2. 拉取整个 cache

```powershell
uv run research_cli tools hf pull-cache
```

说明：

- 只支持整个 `local/cache` 的 push/pull，不再保留 shard 级 CLI 选择
- 每个 `requests.sqlite` 同级维护 `requests.sqlite.hfhash.json`
- sidecar 与远端 `cache_manifest.json` 的 `sqlite_sha256` 一致时，默认跳过

## 三、runs：按完整 run 同步

### 1. 推送整个 runs 工作区

```powershell
uv run research_cli tools hf push-runs
```

默认行为：

- 只推送验证通过的标准实验 run
- 只推送完整收敛的矩阵目录
- 已经发布且 bundle 指纹一致的 run 会自动跳过

### 2. 推送指定目录

```powershell
uv run research_cli tools hf push-runs --source local/runs/dmad
```

```powershell
uv run research_cli tools hf push-runs --source local/runs/dmad/dmad_reasoning_main/count20/20260531T133205Z-xiaomimimo-mimo-v2.5
```

说明：

- `--source` 可以是完整 run，也可以是 `runs` 下任意父级目录
- 真正同步单位始终是最小完整 run 目录

### 3. 无需验证直接推送

```powershell
uv run research_cli tools hf push-runs --skip-validation
```

说明：

- `--skip-validation` 会跳过标准实验 run 的 `run_validation.json` 检查
- 也会跳过矩阵目录的完整性检查

### 4. 拉取指定前缀

```powershell
uv run research_cli tools hf pull-runs --prefix dmad
```

```powershell
uv run research_cli tools hf pull-runs --prefix dmad/dmad_reasoning_main/count20
```

说明：

- `--prefix` 是远端 runs repo 内的相对前缀
- 可以是完整 run 前缀，也可以是任意父级前缀
- 本地已存在且 bundle 指纹一致的 run 默认跳过

### 5. 拉取最近一小时 runs

```powershell
uv run research_cli tools hf pull-runs --recent-hours 1
```

也可以叠加前缀：

```powershell
uv run research_cli tools hf pull-runs --prefix dmad --recent-hours 1
```

说明：

- `--recent-hours` 只按远端 `runs_manifest.json` 中的 `published_at` 过滤
- 不依赖 Hugging Face commit 时间

## 四、当前不支持的操作

- cache 的 shard 级 CLI 过滤
- 自动覆盖本地冲突 run
- 自动备份本地冲突 run 后覆盖

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
