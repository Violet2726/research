"""single_agent ??????????"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """单次运行目录下各类产物文件的固定路径集合。"""

    root: Path
    manifest: Path
    raw_responses: Path
    predictions: Path
    metrics: Path
    run_summary: Path
    report_markdown: Path
    paper_tables: Path
    run_validation: Path
    progress: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录，并返回其中所有固定产物路径。"""
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        raw_responses=root / "raw_responses.jsonl",
        predictions=root / "predictions.jsonl",
        metrics=root / "metrics.json",
        run_summary=root / "run_summary.json",
        report_markdown=root / "report.md",
        paper_tables=root / "paper_tables.md",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )
