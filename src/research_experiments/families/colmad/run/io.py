"""ColMAD 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """ColMAD 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    debate_trace: Path
    judge_trace: Path
    final_predictions: Path
    metrics: Path
    protocol_diagnostics: Path
    run_summary: Path
    run_validation: Path
    progress: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建 ColMAD 运行目录并返回固定产物路径。"""

    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        debate_trace=root / "debate_trace.jsonl",
        judge_trace=root / "judge_trace.jsonl",
        final_predictions=root / "final_predictions.jsonl",
        metrics=root / "metrics.json",
        protocol_diagnostics=root / "protocol_diagnostics.json",
        run_summary=root / "run_summary.json",
        run_validation=root / "run_validation.json",
        progress=root / "progress.json",
    )
