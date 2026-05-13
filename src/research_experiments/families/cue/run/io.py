"""`cue` 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """CUE 运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    communication_turns: Path
    audit_turns: Path
    control_turns: Path
    policy_predictions: Path
    policy_metrics: Path
    policy_diagnostics: Path
    oracle_trigger_eval: Path
    progress: Path
    run_validation: Path
    cue_report: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        communication_turns=root / "communication_turns.jsonl",
        audit_turns=root / "audit_turns.jsonl",
        control_turns=root / "control_turns.jsonl",
        policy_predictions=root / "policy_predictions.jsonl",
        policy_metrics=root / "policy_metrics.json",
        policy_diagnostics=root / "policy_diagnostics.json",
        oracle_trigger_eval=root / "oracle_trigger_eval.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        cue_report=root / "report.md",
    )
