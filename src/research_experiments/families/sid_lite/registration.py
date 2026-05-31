"""`sid_lite` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.sid_lite.config import load_experiment_config, load_protocol_config
from research_experiments.families.sid_lite.run.execute import run_experiment
from research_experiments.families.sid_lite.run.report import render_report, summarize_run
from research_experiments.families.sid_lite.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    backbone = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("sid_lite"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": {
            "name": backbone.name,
            "provider": backbone.provider,
            "model_id": backbone.model_id,
            "tags": backbone.tags,
        },
        "phases": {
            phase_name: phase_metadata(experiment, phase_name)
            for phase_name in experiment.raw["phases"]
        },
    }


ARTIFACT_ALIASES = {
    "stage_a_turns": "turns/stage_a_turns.jsonl",
"message_packets": "turns/message_packets.jsonl",
"belief_updates": "turns/belief_updates.jsonl",
"final_predictions": "views/predictions.jsonl",
"diagnostics_path": "diagnostics/diagnostics.json",
"run_validation": "run_validation.json",
"paper_summary": "exports/paper_summary.csv",
}

REGISTRATION = make_family_registration(
    family_name="sid_lite",
    prototype="packet_belief_update",
    cli_help=FamilyCliHelp(
        description="SID-lite experiment runner.",
        inspect_help="Show resolved SID-lite config.",
        run_help="Execute one SID-lite phase.",
        summarize_help="Print SID-lite summary.",
        validate_help="Validate SID-lite run.",
        report_help="Regenerate SID-lite report.",
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
    turn_record_paths=('turns/stage_a_turns.jsonl', 'turns/message_packets.jsonl', 'turns/belief_updates.jsonl'),
    diagnostic_paths=('diagnostics/diagnostics.json',),
    export_paths=('exports/paper_summary.csv',),
)

