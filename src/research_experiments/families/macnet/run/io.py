"""Canonical run-layout helpers for `macnet`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "artifact_trace": "turns/artifact_trace.jsonl",
"instruction_trace": "turns/instruction_trace.jsonl",
"final_predictions": "views/predictions.jsonl",
"topology_manifest": "exports/topology_manifest.json",
"scaling_summary": "diagnostics/scaling_summary.json",
"report": "report.md",
"figure_manifest": "figure_manifest.json",
"archive_manifest": "archive_manifest.json",
"run_validation": "run_validation.json",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("macnet")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
