# Hugging Face 归档工作流

项目采用两套不同的远程治理语义：

- 完整操作索引见：[huggingface_operations.md](/d:/user/research/docs/huggingface_operations.md)。
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
uv run research_cli tools archive-runs publish-run --run-root local/runs/<family>/<experiment>/<phase>/<run_id>
uv run research_cli tools archive-runs fetch-run --run-id <run_id>
```

说明：

- `publish-run` 会在推送前自动完成打包，不需要单独执行打包命令
- `fetch-run` 会在拉取后自动完成归档解压，不需要单独执行解压命令
- 若 `RESEARCH_RUNS_HF_REPO` 已配置，`publish-run` 可省略 `--repo`
- 若 `RESEARCH_AUTO_PUBLISH_RUNS=1` 已配置，family 级 run 完成后会自动发布

## 2. cache：latest-only 快照

`local/cache/` 不再依赖 Git LFS。远程同步时保持“只保最新快照”的语义：

- 目录结构保持 `providers/<provider>/<request_model>/<dataset>/`
- 每个分库压缩为 `requests.sqlite.zst`
- 同目录保留 `metadata.json` 与 `sha256.txt`

常用命令：

```powershell
uv run research_cli tools cache-archive push-latest --cache-root local/cache
uv run research_cli tools cache-archive pull-latest --target local/cache
```

## 3. 一键工作区同步

当你希望把 `local/runs` 与 `local/cache` 作为一个整体处理时，可直接使用统一总控命令：

```powershell
uv run research_cli tools hf-sync status
uv run research_cli tools hf-sync push-workspace
uv run research_cli tools hf-sync pull-workspace
```

说明：

- `push-workspace` 会批量扫描 `local/runs`，只发布验证通过的标准 run，以及已经完整收敛的 `faithful_matrix` 目录
- 默认会跳过已经写入 `hf_publish.json` 且远端仓库一致的 run；需要重推时可加 `--force-runs`
- `pull-workspace` 会按远端 `archive_manifest.json` 列表批量回拉 runs，并自动解压归档包
- 若需要只同步其中一侧，可加 `--skip-runs` 或 `--skip-cache`

## 4. 额外说明

- 若 `RESEARCH_CACHE_HF_REPO` 已配置，命令可省略 `--repo`
- 若 `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1` 已配置，`run_all_phases` 会在三阶段结束后自动推送最新快照

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

## 6. 维护建议

- `runs` 与 `cache` 使用不同的 HF repo，不要混仓
- `runs` 优先公开；`cache` 优先私有
- `local/reports/` 是本地发布视图，不再作为 Git 正式产物
- 若只是临时实验，优先通过 `RESEARCH_*_ROOT` 输出到隔离目录
