"""`dmad` run-directory helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """Fixed artifact paths for one DMAD run."""

    root: Path
    manifest: Path
    agent_turns: Path
    debate_messages: Path
    final_predictions: Path
    metrics: Path
    strategy_diagnostics: Path
    cost_breakdown: Path
    paper_tables: Path
    run_summary: Path
    run_validation: Path
    progress: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        debate_messages=root / "debate_messages.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        strategy_diagnostics=root / "strategy_diagnostics.json",
        cost_breakdown=root / "cost_breakdown.json",
        paper_tables=root / "paper_tables.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )
