# Hugging Face 归档工作流

项目后续采用两套不同的远程治理语义：

- `runs/`：公开 Hugging Face dataset repo，强调版本化科研档案与在线浏览
- `cache/`：latest-only 快照，强调可替换加速资产，不和 `runs` 混仓

## runs：可浏览外壳 + 压缩内核

每个正式 run 在本地完成后会自动生成：

- `archive_manifest.json`
- `traces.tar.zst`
- `predictions.tar.zst`
- `artifacts.tar.zst`（仅在需要时生成）

默认保留在线可浏览的文件：

- `report.md`
- `metrics.json`
- `figure_manifest.json`
- `figures/*.svg`
- `figures/*.csv`
- 其他小型诊断、附录与摘要文件

默认压缩的重型文件：

- `raw_responses.jsonl`
- `*_turns.jsonl`
- `message_packets.jsonl`
- `sample_views.jsonl`
- `predictions.jsonl`
- `final_predictions.jsonl`
- `hotpot_predictions/`

常用命令：

```powershell
uv run archive_runs_cli pack-run --run-root runs/<family>/<experiment>/<phase>/<run_id>
```

```powershell
uv run archive_runs_cli publish-run --run-root runs/<family>/<experiment>/<phase>/<run_id> --repo <owner>/research-runs
```

```powershell
uv run archive_runs_cli fetch-run --run-id <run_id> --repo <owner>/research-runs --include all
```

## cache：latest-only 快照

`cache/` 不再依赖 Git LFS。远程同步时保持“只保最新快照”的语义：

- 目录结构保持 `providers/<provider>/<request_model>/<dataset>/`
- 每个分库压缩为 `requests.sqlite.zst`
- 同目录保留 `metadata.json` 与 `sha256.txt`

常用命令：

```powershell
uv run cache_archive_cli push-latest --cache-root cache --repo <owner>/research-cache
```

```powershell
uv run cache_archive_cli pull-latest --target cache --repo <owner>/research-cache
```

## 维护建议

- `runs` 与 `cache` 使用不同的 HF repo，不要混仓。
- `runs` 优先公开；`cache` 优先私有或仅本地备份。
- published report 继续只引用 `runs/.../figures/`，不要把大图或 trace 复制回代码仓库。
- 若只是临时实验，优先通过 `RESEARCH_RUNS_ROOT` / `RESEARCH_CACHE_ROOT` 输出到隔离目录，不要直接进入正式归档链。
