# local

`local/` 是默认本地工作区，不进入 Git 主线。

默认结构：

- `local/runs/`
- `local/reports/`
- `local/cache/`
- `local/datasets/`

如果需要把工作区迁到别处，请通过 `.env.local` 或环境变量覆盖：

- `RESEARCH_RUNS_ROOT`
- `RESEARCH_REPORTS_ROOT`
- `RESEARCH_CACHE_ROOT`
- `RESEARCH_DATASETS_ROOT`
