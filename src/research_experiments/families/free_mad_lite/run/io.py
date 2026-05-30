"""Canonical run-layout helpers for `free_mad_lite`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"debate_messages": "turns/debate_messages.jsonl",
"trajectory_scores": "exports/trajectory_scores.jsonl",
"final_predictions": "views/predictions.jsonl",
"diagnostics": "diagnostics/diagnostics.json",
"run_validation": "run_validation.json",
"paper_summary": "exports/paper_summary.csv",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("free_mad_lite")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
