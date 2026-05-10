# Hugging Face 归档工作流

项目采用两套不同的远程治理语义：

- `runs`：公开 Hugging Face dataset repo，强调正式科研档案与在线浏览
- `cache`：latest-only 快照，强调可替换加速资产，不和 `runs` 混仓

## 1. runs：可浏览外壳 + 压缩内核

每个正式 run 在本地完成后会生成：

- `archive_manifest.json`
- `traces.tar.zst`
- `predictions.tar.zst`
- `artifacts.tar.zst`（仅在需要时生成）

在线保留的可浏览文件：

- `report.md`
- `metrics.json`
- `figure_manifest.json`
- `figures/*.svg`
- `figures/*.csv`
- 其他小型诊断、附录与摘要文件

重型文件默认压入归档包：

- `raw_responses.jsonl`
- `*_turns.jsonl`
- `message_packets.jsonl`
- `sample_views.jsonl`
- `predictions.jsonl`
- `final_predictions.jsonl`
- `hotpot_predictions/`

常用命令：

```powershell
uv run archive_runs_cli pack-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
uv run archive_runs_cli publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
uv run archive_runs_cli fetch-run --run-id <run_id> --include all
```

说明：

- 若 `RESEARCH_RUNS_HF_REPO` 已配置，`publish-run` 可省略 `--repo`
- 若 `RESEARCH_AUTO_PUBLISH_RUNS=1` 已配置，family 级 run 完成后会自动发布

## 2. cache：latest-only 快照

`local/cache/` 不再依赖 Git LFS。远程同步时保持“只保最新快照”的语义：

- 目录结构保持 `providers/<provider>/<request_model>/<dataset>/`
- 每个分库压缩为 `requests.sqlite.zst`
- 同目录保留 `metadata.json` 与 `sha256.txt`

常用命令：

```powershell
uv run cache_archive_cli push-latest --cache-root local/cache
uv run cache_archive_cli pull-latest --target local/cache
```

说明：

- 若 `RESEARCH_CACHE_HF_REPO` 已配置，命令可省略 `--repo`
- 若 `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1` 已配置，`run_all_phases` 会在三阶段结束后自动推送最新快照

## 3. 推荐环境变量

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

## 4. 维护建议

- `runs` 与 `cache` 使用不同的 HF repo，不要混仓
- `runs` 优先公开；`cache` 优先私有
- `local/reports/` 是本地发布视图，不再作为 Git 正式产物
- 若只是临时实验，优先通过 `RESEARCH_*_ROOT` 输出到隔离目录
