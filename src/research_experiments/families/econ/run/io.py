"""Canonical run-layout helpers for `econ`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"belief_trace": "turns/belief_trace.jsonl",
"equilibrium_trace": "turns/equilibrium_trace.jsonl",
"communication_trace": "turns/communication_trace.jsonl",
"final_predictions": "views/predictions.jsonl",
"run_validation": "run_validation.json",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("econ")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
