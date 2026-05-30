"""Canonical run-layout helpers for `multi_agent`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"debate_messages": "turns/debate_messages.jsonl",
"final_predictions": "views/predictions.jsonl",
"cost_breakdown": "diagnostics/cost_breakdown.json",
"debate_diagnostics": "diagnostics/debate_diagnostics.json",
"run_validation": "run_validation.json",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("multi_agent")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
