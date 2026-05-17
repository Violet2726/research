# 实验运行与报告生成总流程

本仓库把“实验执行”和“科研报告生成”视为同一条正式产物链路，而不是彼此独立的脚本。

## 总流程

```text
config / benchmark / model catalog
            |
            v
runner 执行实验
            |
            v
raw_responses.jsonl / predictions.jsonl / metrics.json / diagnostics.json
            |
            v
research_experiments.reporting.run_figures
            |
            +--> local/runs/<family>/<experiment>/<phase>/<run_id>/figures/*.svg
            +--> local/runs/<family>/<experiment>/<phase>/<run_id>/figures/*.csv
            +--> local/runs/<family>/<experiment>/<phase>/<run_id>/figure_manifest.json
            |
            v
research_experiments.reporting.report_pipeline
            |
            +--> local/runs/.../report.md
            +--> local/runs/.../frontier_report.md 等附录报告
            +--> local/reports/<family>/<date>-<experiment>-<phase>-<backbone>-report.md
            |
            v
research_experiments.workspace.run_archives
            |
            +--> local/runs/.../archive_manifest.json
            +--> local/runs/.../traces.tar.zst
            +--> local/runs/.../predictions.tar.zst
            +--> local/runs/.../artifacts.tar.zst
            |
            v
Hugging Face dataset repo
```

## 关键约定

- 正式图资产只落在 `local/runs/.../figures/`
- `local/reports/` 是本地发布视图，不是正式长期归档
- faithful matrix 的发布视图除 `paper_package.md` 外，还会额外发布 `family_landscape.md`
- reproduction matrix 会发布 `reproduction_package.md` 与 `reproduction_landscape.md`
- 正式远程归档以 Hugging Face dataset repo 为准
- `figure_manifest.json` 是 run 级图资产的唯一索引
- `archive_manifest.json` 是 run 级重型文件归档的唯一索引

## 运行后自动动作

- family 级 runner 在 `finalize_run_outputs()` 中统一执行：
  - 打包重型文件
  - 写出 `run_validation.json`
  - 按环境开关自动发布到 `RESEARCH_RUNS_HF_REPO`
- `run_all_phases.ps1` / `run_all_phases.sh`
  - 顺序运行 `count20 -> count100 -> count300 -> count500`
  - 每个阶段结束后要求 matrix 全部成功
  - faithful matrix 成功后默认补齐 `faithful_analysis`、`acceptance_summary`、`paper_statistics`、`paper_package`、`family_landscape`
  - reproduction matrix 成功后默认补齐 `reproduction_analysis`、`reproduction_package`、`reproduction_landscape`
  - 若启用 `RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT=1`，则在四阶段结束后推送 `local/cache` 最新快照

## 维护建议

- 新增实验家族时，优先复用 `report_pipeline.py`
- 新增图表时，优先扩展 `run_figures.py`
- 临时实验优先通过 `RESEARCH_*_ROOT` 输出到隔离目录，不直接写入正式工作区

