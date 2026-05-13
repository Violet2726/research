"""multi_agent ??????????"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """多智能体运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    agent_turns: Path
    debate_messages: Path
    final_predictions: Path
    metrics: Path
    cost_breakdown: Path
    debate_diagnostics: Path
    run_summary: Path
    run_validation: Path
    progress: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建多智能体运行目录和固定产物路径。"""
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        debate_messages=root / "debate_messages.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        cost_breakdown=root / "cost_breakdown.json",
        debate_diagnostics=root / "debate_diagnostics.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )
