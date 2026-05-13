"""free_mad_lite ??????????"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """Free-MAD-lite 运行目录下的固定产物路径。"""

    root: Path
    manifest: Path
    agent_turns: Path
    debate_messages: Path
    trajectory_scores: Path
    final_predictions: Path
    metrics: Path
    diagnostics: Path
    progress: Path
    run_summary: Path
    run_validation: Path
    report_markdown: Path
    paper_summary: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        debate_messages=root / "debate_messages.jsonl",
        trajectory_scores=root / "trajectory_scores.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        diagnostics=root / "diagnostics.json",
        progress=root / "progress.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
        paper_summary=root / "paper_summary.csv",
    )
