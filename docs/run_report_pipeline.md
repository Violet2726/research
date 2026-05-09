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
experiment_core.reporting.run_figures
            |
            +--> runs/<family>/<experiment>/<phase>/<run_id>/figures/*.svg
            +--> runs/<family>/<experiment>/<phase>/<run_id>/figures/*.csv
            +--> runs/<family>/<experiment>/<phase>/<run_id>/figure_manifest.json
            |
            v
experiment_core.reporting.report_pipeline
            |
            +--> runs/.../report.md
            +--> runs/.../frontier_report.md 等附录报告
            +--> reports/<family>/<date>-<experiment>-<phase>-<backbone>-report.md
            |
            v
faithful_matrix / paper_package / reports/summary
```

## 关键约定

- 正式图资产只落在 `runs/.../figures/`；`reports/` 只引用这些 canonical 图文件。
- 各实验家族统一通过 `render_report()` 输出正式科研报告，并通过 CLI 子命令 `render-report` 重渲染。
- 正式科研报告统一使用中文结构：摘要、实验概览、研究问题与实验设计、总体结果、机制诊断、结论与建议、局限性、复现与产物说明。
- `figure_manifest.json` 是 run 级图资产的唯一索引，校验、发布和论文打包都基于它工作。

## 维护建议

- 新增实验家族时，优先复用 `experiment_core.reporting.report_pipeline`，不要再手写 local report / published report / figure manifest 的重复逻辑。
- 新增图表时，优先扩展 `run_figures.py` 的 figure spec，而不是在各家族 `reporting.py` 中重复写 SVG。
- 临时实验应优先通过 `RESEARCH_*_ROOT` 输出到隔离目录，避免把未确认产物混入正式 `runs/`。
