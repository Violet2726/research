"""RCTA-MAD family 注册。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.risk_controlled_trace_mad.config import (
    inspect_benchmarks,
    inspect_methods,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import RCTA_PROMPT_VERSION, RCTA_SCHEMA_VERSION
from research_experiments.families.risk_controlled_trace_mad.run.execute import run_experiment
from research_experiments.families.risk_controlled_trace_mad.run.report import render_report, summarize_run
from research_experiments.families.risk_controlled_trace_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults

FAMILY_NAME = "risk_controlled_trace_mad"


def inspect_experiment(path: str, model_override: str | None):
    experiment = load_experiment_config(path)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmarks": inspect_benchmarks(experiment),
        "protocol": {**asdict(protocol), "router_artifact": str(protocol.router_artifact)},
        "controls": {name: asdict(controls[name]) for name in experiment.control_methods},
        "methods": inspect_methods(experiment),
        "method_order": experiment.method_order,
        "prompt_version": RCTA_PROMPT_VERSION,
        "schema_version": RCTA_SCHEMA_VERSION,
        "global_seed": experiment.global_seed,
        "runtime_profiles": {key: asdict(value) for key, value in experiment.runtime_profiles.items()},
        "workspace_defaults": workspace_defaults(FAMILY_NAME),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(model),
        "phases": experiment.raw["phases"],
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    return run_experiment(experiment, request.phase_name, resolve_model(request.model_ref or experiment.primary_model_ref), request.runs_root, request.cache_root)


ALIASES = {"agent_turns": "turns/agent_turns.jsonl", "debate_messages": "turns/debate_messages.jsonl", "router_decisions": "turns/router_decisions.jsonl", "rcta_diagnostics": "diagnostics/rcta_diagnostics.json", "paired_statistics": "diagnostics/paired_statistics.json", "output_protocol_diagnostics": "diagnostics/output_protocol_diagnostics.json", "rcta_comparison": "exports/rcta_comparison.json", "paper_summary": "exports/paper_summary.csv"}

REGISTRATION = make_family_registration(
    family_name=FAMILY_NAME, prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(description="Risk-Controlled Trace Aggregation MAD experiments.", inspect_help="Show resolved RCTA configuration.", run_help="Run an RCTA phase.", summarize_help="Summarize an RCTA run.", validate_help="Validate RCTA artifacts and invariants.", report_help="Regenerate the RCTA report."),
    load_experiment=load_experiment_config, resolve_model=resolve_model, invoke_runner=run_experiment,
    inspect_experiment=inspect_experiment, run_from_cli=run_from_cli, summarize_run=summarize_run, validate_run=validate_run, render_report=render_report,
    artifact_aliases=ALIASES, metrics_view_path="views/metrics.json", prediction_records_path="views/predictions.jsonl",
    turn_record_paths=("turns/agent_turns.jsonl", "turns/debate_messages.jsonl", "turns/router_decisions.jsonl"),
    diagnostic_paths=("diagnostics/rcta_diagnostics.json", "diagnostics/paired_statistics.json", "diagnostics/output_protocol_diagnostics.json"),
    export_paths=("exports/rcta_comparison.json", "exports/paper_summary.csv"),
)
