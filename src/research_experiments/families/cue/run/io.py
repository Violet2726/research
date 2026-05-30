"""Canonical run-layout helpers for `cue`."""

from __future__ import annotations

from pathlib import Path

from research_experiments.core.families.run_layout import FamilyRunLayout, prepare_family_run_layout
from research_experiments.families.registry import get_family_registration


ALIASES = {
    "stage_a_turns": "turns/stage_a_turns.jsonl",
"communication_turns": "turns/communication_turns.jsonl",
"audit_turns": "turns/audit_turns.jsonl",
"control_turns": "turns/control_turns.jsonl",
"policy_predictions": "views/predictions.jsonl",
"policy_metrics": "views/metrics.json",
"policy_diagnostics": "diagnostics/policy_diagnostics.json",
"oracle_trigger_eval": "diagnostics/oracle_trigger_eval.json",
"run_validation": "run_validation.json",
"cue_report": "report.md",
}


def prepare_run_layout(run_root: str | Path, experiment_name: str, phase_name: str, run_id: str) -> FamilyRunLayout:
    registration = get_family_registration("cue")
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=ALIASES,
    )
