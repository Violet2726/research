"""`budget_comm` 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """`budget_comm` 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    sample_views: Path
    stage_a_turns: Path
    candidate_packets: Path
    auction_decisions: Path
    belief_updates: Path
    final_predictions: Path
    metrics: Path
    budget_diagnostics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录，并返回其中所有固定产物路径。"""
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        sample_views=root / "sample_views.jsonl",
        stage_a_turns=root / "stage_a_turns.jsonl",
        candidate_packets=root / "candidate_packets.jsonl",
        auction_decisions=root / "auction_decisions.jsonl",
        belief_updates=root / "belief_updates.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        budget_diagnostics=root / "budget_diagnostics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )
