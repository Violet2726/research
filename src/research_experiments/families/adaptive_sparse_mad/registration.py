"""adaptive_sparse_mad family registration."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from research_experiments.cli_support.output import emit_json
from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.adaptive_sparse_mad.config import (
    inspect_benchmarks,
    inspect_methods,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.adaptive_sparse_mad.run.execute import refresh_stage_a_only_run_artifacts, run_experiment
from research_experiments.families.adaptive_sparse_mad.run.report import render_report, summarize_run
from research_experiments.families.adaptive_sparse_mad.run.validate import validate_run
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
        "controls": {name: asdict(method) for name, method in controls.items()},
        "methods": inspect_methods(experiment),
        "prompt_version": experiment.prompt_version,
        "stage_a_prompt_version": experiment.stage_a_prompt_version,
        "adaptive_prompt_version": experiment.adaptive_prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("adaptive_sparse_mad"),
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


def configure_parser(parser) -> None:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        refresh = action.add_parser(
            "refresh-run-artifacts",
            help="Refresh metrics/diagnostics/report for one completed A-SMAD run using current code.",
        )
        refresh.add_argument("--run-dir", required=True)
        return
    raise RuntimeError("Parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
    if args.command != "refresh-run-artifacts":
        return False
    run_dir = refresh_stage_a_only_run_artifacts(args.run_dir)
    emit_json({"run_dir": run_dir.as_posix(), "status": "refreshed"})
    return True


ARTIFACT_ALIASES = {
    "stage_a_turns": "turns/stage_a_turns.jsonl",
    "control_turns": "turns/control_turns.jsonl",
    "router_decisions": "turns/router_decisions.jsonl",
    "router_eval": "diagnostics/router_eval.json",
    "policy_diagnostics": "diagnostics/policy_diagnostics.json",
    "stage_a_resolver_breakdown": "diagnostics/stage_a_resolver_breakdown.json",
    "stage_a_error_buckets": "diagnostics/stage_a_error_buckets.json",
    "stage_a_solver_contributions": "diagnostics/stage_a_solver_contributions.json",
}


REGISTRATION = make_family_registration(
    family_name="adaptive_sparse_mad",
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="A-SMAD same-context experiment runner.",
        inspect_help="Show the resolved A-SMAD experiment configuration.",
        run_help="Execute one configured A-SMAD experiment phase.",
        summarize_help="Print a concise A-SMAD run summary.",
        validate_help="Run A-SMAD validation checks.",
        report_help="Regenerate the Chinese A-SMAD markdown report.",
    ),
    load_experiment=load_experiment_config,
    resolve_model=resolve_model,
    invoke_runner=run_experiment,
    inspect_experiment=inspect_experiment,
    run_from_cli=run_from_cli,
    summarize_run=summarize_run,
    validate_run=validate_run,
    render_report=render_report,
    configure_parser=configure_parser,
    dispatch_extra_command=dispatch_extra_command,
    artifact_aliases=ARTIFACT_ALIASES,
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=(
        "turns/stage_a_turns.jsonl",
        "turns/control_turns.jsonl",
        "turns/router_decisions.jsonl",
    ),
    diagnostic_paths=(
        "diagnostics/router_eval.json",
        "diagnostics/policy_diagnostics.json",
        "diagnostics/stage_a_resolver_breakdown.json",
        "diagnostics/stage_a_error_buckets.json",
        "diagnostics/stage_a_solver_contributions.json",
    ),
)
