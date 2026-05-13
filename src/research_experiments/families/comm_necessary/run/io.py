"""comm_necessary ??????????"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """comm_necessary 运行目录下的固定产物路径。"""

    root: Path
    manifest: Path
    sample_views: Path
    stage_a_turns: Path
    message_packets: Path
    stage_b_turns: Path
    final_predictions: Path
    hotpot_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        sample_views=root / "sample_views.jsonl",
        stage_a_turns=root / "stage_a_turns.jsonl",
        message_packets=root / "message_packets.jsonl",
        stage_b_turns=root / "stage_b_turns.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        hotpot_predictions=root / "hotpot_predictions",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )
