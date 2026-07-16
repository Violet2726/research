"""盲审重构辩论实验族的注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.blind_reconstructive_mad.config import (
    inspect_benchmarks,
    inspect_methods,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.blind_reconstructive_mad.prompts import BRD_PROMPT_VERSION
from research_experiments.families.blind_reconstructive_mad.run.execute import run_experiment
from research_experiments.families.blind_reconstructive_mad.run.report import render_report, summarize_run
from research_experiments.families.blind_reconstructive_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    resolved = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": inspect_benchmarks(experiment),
        "protocol": asdict(protocol),
        "controls": {name: asdict(controls[name]) for name in experiment.control_methods},
        "methods": inspect_methods(experiment),
        "method_order": experiment.method_order,
        "brd_prompt_version": BRD_PROMPT_VERSION,
        "global_seed": experiment.global_seed,
        "runtime_profiles": {name: asdict(profile) for name, profile in experiment.runtime_profiles.items()},
        "workspace_defaults": workspace_defaults("blind_reconstructive_mad"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved),
        "phases": experiment.raw["phases"],
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    return run_experiment(
        experiment=experiment,
        phase_name=request.phase_name,
        backbone=resolve_model(request.model_ref or experiment.primary_model_ref),
        run_root=request.runs_root,
        cache_root=request.cache_root,
    )


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
    "debate_messages": "turns/debate_messages.jsonl",
    "router_decisions": "turns/router_decisions.jsonl",
    "brd_diagnostics": "diagnostics/brd_diagnostics.json",
    "paired_statistics": "diagnostics/paired_statistics.json",
    "pilot_gate": "diagnostics/pilot_gate.json",
    "output_protocol_diagnostics": "diagnostics/output_protocol_diagnostics.json",
}


REGISTRATION = make_family_registration(
    family_name="blind_reconstructive_mad",
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="Blind Review-Reconstruct Debate (BRD-MAD) experiments.",
        inspect_help="Show resolved BRD-MAD configuration.",
        run_help="Run a BRD-MAD phase.",
        summarize_help="Print a BRD-MAD run summary.",
        validate_help="Validate BRD-MAD artifacts and safety invariants.",
        report_help="Regenerate the BRD-MAD report.",
    ),
    load_experiment=load_experiment_config,
    resolve_model=resolve_model,
    invoke_runner=run_experiment,
    inspect_experiment=inspect_experiment,
    run_from_cli=run_from_cli,
    summarize_run=summarize_run,
    validate_run=validate_run,
    render_report=render_report,
    artifact_aliases=ARTIFACT_ALIASES,
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=("turns/agent_turns.jsonl", "turns/debate_messages.jsonl", "turns/router_decisions.jsonl"),
    diagnostic_paths=("diagnostics/brd_diagnostics.json", "diagnostics/paired_statistics.json", "diagnostics/output_protocol_diagnostics.json", "diagnostics/pilot_gate.json"),
    export_paths=("exports/brd_comparison.json", "exports/paper_summary.csv"),
)
