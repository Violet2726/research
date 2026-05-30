"""`cue` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.cue.config import (
    load_control_catalog,
    load_experiment_config,
    load_policies,
    load_protocol_config,
)
from research_experiments.families.cue.run.execute import run_experiment
from research_experiments.families.cue.run.report import render_report, summarize_run
from research_experiments.families.cue.run.validate import validate_run
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.family_runtime.config_helpers import load_benchmarks, resolve_model
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": asdict(protocol),
        "policies": [asdict(policy) for policy in policies],
        "controls": {name: asdict(method) for name, method in sorted(controls.items())},
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("cue"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


ARTIFACT_ALIASES = {
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

REGISTRATION = make_family_registration(
    family_name="cue",
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="CUE experiment runner.",
        inspect_help="Show the resolved CUE experiment configuration.",
        run_help="Execute one configured CUE phase.",
        summarize_help="Print a concise run summary from views/metrics.json.",
        validate_help="Run validation checks for one CUE run.",
        report_help="Regenerate the Chinese CUE markdown report.",
    ),
    load_experiment=load_experiment_config,
    resolve_model=resolve_model,
    invoke_runner=run_experiment,
    inspect_experiment=inspect_experiment,
    run_from_cli=build_backbone_run_from_cli(
        load_experiment=load_experiment_config,
        resolve_model=resolve_model,
        invoke_runner=run_experiment,
    ),
    summarize_run=summarize_run,
    validate_run=validate_run,
    render_report=render_report,
    artifact_aliases=ARTIFACT_ALIASES,
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=('turns/stage_a_turns.jsonl', 'turns/communication_turns.jsonl', 'turns/audit_turns.jsonl', 'turns/control_turns.jsonl'),
    diagnostic_paths=('diagnostics/policy_diagnostics.json', 'diagnostics/oracle_trigger_eval.json'),
    export_paths=('exports/frontier_report.md',),
)

