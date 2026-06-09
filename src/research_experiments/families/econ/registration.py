"""`econ` family 注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.econ.config import load_experiment_config, load_protocol_config
from research_experiments.families.econ.run.execute import run_experiment
from research_experiments.families.econ.run.report import render_report, summarize_run
from research_experiments.families.econ.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in experiment.methods],
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("econ"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"belief_trace": "turns/belief_trace.jsonl",
"equilibrium_trace": "turns/equilibrium_trace.jsonl",
"communication_trace": "turns/communication_trace.jsonl",
"final_predictions": "views/predictions.jsonl",
"run_validation": "run_validation.json",
}

REGISTRATION = make_family_registration(
    family_name="econ",
    prototype="packet_belief_update",
    cli_help=FamilyCliHelp(
        description="ECON 低通信协同实验运行器。",
        inspect_help="Show the resolved ECON experiment configuration.",
        run_help="Execute one configured ECON experiment phase.",
        summarize_help="Print a concise ECON run summary.",
        validate_help="Run ECON validation checks.",
        report_help="Regenerate the Chinese ECON markdown report.",
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
    turn_record_paths=('turns/agent_turns.jsonl', 'turns/belief_trace.jsonl', 'turns/equilibrium_trace.jsonl', 'turns/communication_trace.jsonl'),
    diagnostic_paths=(),
    export_paths=(),
)

