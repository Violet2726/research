"""ECON 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """`econ` 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    agent_turns: Path
    belief_trace: Path
    equilibrium_trace: Path
    communication_trace: Path
    final_predictions: Path
    metrics: Path
    progress: Path
    run_validation: Path
    report_markdown: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录，并返回其中所有固定产物路径。"""

    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        agent_turns=root / "agent_turns.jsonl",
        belief_trace=root / "belief_trace.jsonl",
        equilibrium_trace=root / "equilibrium_trace.jsonl",
        communication_trace=root / "communication_trace.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        report_markdown=root / "report.md",
    )

