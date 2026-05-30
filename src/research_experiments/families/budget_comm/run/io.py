"""Canonical run-layout helpers for `budget_comm`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "sample_views": "turns/sample_views.jsonl",
"stage_a_turns": "turns/stage_a_turns.jsonl",
"candidate_packets": "turns/candidate_packets.jsonl",
"auction_decisions": "turns/auction_decisions.jsonl",
"belief_updates": "turns/belief_updates.jsonl",
"final_predictions": "views/predictions.jsonl",
"budget_diagnostics": "diagnostics/budget_diagnostics.json",
"run_validation": "run_validation.json",
"report_markdown": "report.md",
"paper_summary": "exports/paper_summary.csv",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("budget_comm")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
