"""`macnet` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.macnet.config import load_experiment_config, load_protocol_config
from research_experiments.families.macnet.run.execute import run_experiment
from research_experiments.families.macnet.run.report import render_report, summarize_run
from research_experiments.families.macnet.run.validate import validate_run
from research_experiments.families.registration_helpers import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.families.shared.config_loading import load_benchmarks, resolve_model
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    benchmarks = load_benchmarks(experiment)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "experiment_kind": experiment.experiment_kind,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": _serialize_protocol(protocol),
        "methods": [asdict(method) for method in experiment.methods],
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("macnet"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


def _serialize_protocol(protocol) -> dict[str, object]:
    payload = asdict(protocol)
    payload["profile_asset_path"] = str(protocol.profile_asset_path)
    return payload


REGISTRATION = make_family_registration(
    family_name="macnet",
    prototype="topology_or_graph",
    cli_help=FamilyCliHelp(
        description="MacNet topology-collaboration experiment runner.",
        inspect_help="Show the resolved MacNet experiment configuration.",
        run_help="Execute one configured MacNet experiment phase.",
        summarize_help="Print a concise MacNet run summary.",
        validate_help="Run MacNet validation checks.",
        report_help="Regenerate the Chinese MacNet markdown report.",
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
    metrics_view_path="metrics.json",
    prediction_records_path="final_predictions.jsonl",
    turn_record_paths=("artifact_trace.jsonl", "instruction_trace.jsonl"),
    extra_view_paths=("topology_manifest.json", "scaling_summary.json"),
)
