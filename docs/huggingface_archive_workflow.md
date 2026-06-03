# Hugging Face 归档工作流

完整操作索引见：[huggingface_operations.md](/d:/user/research/docs/huggingface_operations.md)。

- `runs`：公开 dataset repo，保存实验 run 归档
- `cache`：独立 latest-only dataset repo，保存最新 cache 快照

## 1. 统一命令入口

```powershell
uv run research_cli tools hf push-cache
uv run research_cli tools hf pull-cache
uv run research_cli tools hf push-runs
uv run research_cli tools hf pull-runs
```

## 2. cache：默认不重复同步

- `push-cache` 会扫描整个 `local/cache`
- 每个 `requests.sqlite` 同级会维护 `requests.sqlite.hfhash.json`
- 侧车哈希与远端 `cache_manifest.json` 一致时，默认跳过上传
- `pull-cache` 先下载 `cache_manifest.json`，本地一致的 shard 默认跳过下载

## 3. runs：目录作选择器，最小单位是完整 run

- `push-runs` 不传 `--source` 时扫描整个 `local/runs`
- `--source` 可以是完整 run，也可以是 `runs` 下任意父级目录
- 真正同步单位始终是最小完整 run 目录，例如：
  - `local/runs/dmad/dmad_reasoning_main/count20/20260531T133205Z-xiaomimimo-mimo-v2.5`
- `pull-runs --prefix` 支持远端任意父级前缀

常用示例：

```powershell
uv run research_cli tools hf push-runs
uv run research_cli tools hf push-runs --source local/runs/dmad
uv run research_cli tools hf push-runs --source local/runs/dmad/dmad_reasoning_main/count20/20260531T133205Z-xiaomimimo-mimo-v2.5
uv run research_cli tools hf pull-runs --prefix dmad
uv run research_cli tools hf pull-runs --prefix dmad/dmad_reasoning_main/count20
uv run research_cli tools hf pull-runs --recent-hours 1
```

## 4. 默认去重与发布时间

- `push-runs` 默认只推送验证通过的标准实验 run 和完整收敛的矩阵目录
- 若需要忽略验证与完整性检查，可加 `--skip-validation`
- runs 去重不再依赖 `hf_publish.json`，改为本地 `hf_run.json` + 远端 `runs_manifest.json` 的 bundle 指纹比较
- 最近一小时 runs 的筛选依据是远端 `runs_manifest.json` 中写入的 `published_at`

## 5. 推荐环境变量

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
