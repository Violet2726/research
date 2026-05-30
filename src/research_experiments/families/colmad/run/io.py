"""Canonical run-layout helpers for `colmad`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "debate_trace": "turns/debate_trace.jsonl",
"judge_trace": "turns/judge_trace.jsonl",
"final_predictions": "views/predictions.jsonl",
"protocol_diagnostics": "diagnostics/protocol_diagnostics.json",
"run_validation": "run_validation.json",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("colmad")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
