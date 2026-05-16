"""MacNet 运行目录约定。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    root: Path
    manifest: Path
    progress: Path
    artifact_trace: Path
    instruction_trace: Path
    final_predictions: Path
    topology_manifest: Path
    scaling_summary: Path
    metrics: Path
    report: Path
    figure_manifest: Path
    archive_manifest: Path
    run_validation: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        progress=root / "progress.json",
        artifact_trace=root / "artifact_trace.jsonl",
        instruction_trace=root / "instruction_trace.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        topology_manifest=root / "topology_manifest.json",
        scaling_summary=root / "scaling_summary.json",
        metrics=root / "metrics.json",
        report=root / "report.md",
        figure_manifest=root / "figure_manifest.json",
        archive_manifest=root / "archive_manifest.json",
        run_validation=root / "run_validation.json",
    )
