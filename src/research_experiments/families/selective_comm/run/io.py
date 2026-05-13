"""`selective_comm` 运行目录与固定产物路径。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.execution.runner_common import prepare_run_root


@dataclass(frozen=True)
class RunPaths:
    """选择性通信运行目录下的固定产物路径集合。"""

    root: Path
    manifest: Path
    stage_a_turns: Path
    stage_b_turns: Path
    control_turns: Path
    trigger_decisions: Path
    policy_predictions: Path
    policy_metrics: Path
    policy_diagnostics: Path
    oracle_trigger_eval: Path
    progress: Path
    run_validation: Path
    trigger_report: Path


def _prepare_run_paths(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> RunPaths:
    """创建运行目录和固定产物路径。"""
    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    return RunPaths(
        root=root,
        manifest=root / "manifest.json",
        stage_a_turns=root / "stage_a_turns.jsonl",
        stage_b_turns=root / "stage_b_turns.jsonl",
        control_turns=root / "control_turns.jsonl",
        trigger_decisions=root / "trigger_decisions.jsonl",
        policy_predictions=root / "policy_predictions.jsonl",
        policy_metrics=root / "policy_metrics.json",
        policy_diagnostics=root / "policy_diagnostics.json",
        oracle_trigger_eval=root / "oracle_trigger_eval.json",
        progress=root / "progress.json",
        run_validation=root / "run_validation.json",
        trigger_report=root / "report.md",
    )
