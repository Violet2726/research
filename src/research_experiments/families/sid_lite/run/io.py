"""Canonical run-layout helpers for `sid_lite`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "stage_a_turns": "turns/stage_a_turns.jsonl",
"message_packets": "turns/message_packets.jsonl",
"belief_updates": "turns/belief_updates.jsonl",
"final_predictions": "views/predictions.jsonl",
"diagnostics": "diagnostics/diagnostics.json",
"run_validation": "run_validation.json",
"paper_summary": "exports/paper_summary.csv",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("sid_lite")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
