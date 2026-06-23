"""CRED-MAD family 注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.cred_mad.config import (
    inspect_benchmarks,
    inspect_methods,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.cred_mad.prompts import CRED_PROMPT_VERSION
from research_experiments.families.cred_mad.run.execute import run_experiment
from research_experiments.families.cred_mad.run.report import render_report, summarize_run
from research_experiments.families.cred_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": inspect_benchmarks(experiment),
        "protocol": asdict(protocol),
        "controls": {name: asdict(controls[name]) for name in experiment.control_methods},
        "methods": inspect_methods(experiment),
        "method_order": experiment.method_order,
        "control_prompt_version": experiment.control_prompt_version,
        "cred_prompt_version": CRED_PROMPT_VERSION,
        "cred_output_protocol": experiment.cred_output_protocol,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "workspace_defaults": workspace_defaults("cred_mad"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    resolved_model = resolve_model(request.model_ref or experiment.primary_model_ref)
    return run_experiment(
        experiment=experiment,
        phase_name=request.phase_name,
        backbone=resolved_model,
        run_root=request.runs_root,
        cache_root=request.cache_root,
    )


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
    "debate_messages": "turns/debate_messages.jsonl",
    "router_decisions": "turns/router_decisions.jsonl",
    "debate_diagnostics": "diagnostics/debate_diagnostics.json",
    "router_eval": "diagnostics/router_eval.json",
    "output_protocol_diagnostics": "diagnostics/output_protocol_diagnostics.json",
}


REGISTRATION = make_family_registration(
    family_name="cred_mad",
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="CRED-MAD 契约化反驳证据辩论实验运行器。",
        inspect_help="Show the resolved CRED-MAD experiment configuration.",
        run_help="Execute one configured CRED-MAD experiment phase.",
        summarize_help="Print a concise CRED-MAD run summary.",
        validate_help="Run CRED-MAD validation checks.",
        report_help="Regenerate the Chinese CRED-MAD markdown report.",
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
    turn_record_paths=(
        "turns/agent_turns.jsonl",
        "turns/debate_messages.jsonl",
        "turns/router_decisions.jsonl",
    ),
    diagnostic_paths=(
        "diagnostics/debate_diagnostics.json",
        "diagnostics/router_eval.json",
        "diagnostics/output_protocol_diagnostics.json",
    ),
    export_paths=(
        "exports/cred_comparison.json",
        "exports/paper_summary.csv",
    ),
)
