"""Canonical run-layout helpers for `consensagent`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "run_root": ".",
"turns_path": "turns/turns.jsonl",
"debate_messages_path": "turns/debate_messages.jsonl",
"predictions_path": "views/predictions.jsonl",
"metrics_path": "views/metrics.json",
"cost_breakdown_path": "diagnostics/cost_breakdown.json",
"debate_diagnostics_path": "diagnostics/debate_diagnostics.json",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("consensagent")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
