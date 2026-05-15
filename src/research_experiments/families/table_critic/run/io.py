"""`table_critic` 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """Table-Critic 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    agent_turns: Path
    critic_trace: Path
    refinement_trace: Path
    final_predictions: Path
    metrics: Path
    template_tree: Path
    error_analysis: Path
    run_summary: Path
    run_validation: Path
    progress: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建 Table-Critic 运行目录并返回固定产物路径。"""

    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        critic_trace=root / "critic_trace.jsonl",
        refinement_trace=root / "refinement_trace.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        template_tree=root / "template_tree.json",
        error_analysis=root / "error_analysis.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )

