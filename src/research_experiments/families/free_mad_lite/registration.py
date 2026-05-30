"""`free_mad_lite` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.free_mad_lite.config import load_experiment_config, load_protocol_config
from research_experiments.families.free_mad_lite.prompts import anti_conformity_prompt_hash
from research_experiments.families.free_mad_lite.run.execute import run_experiment
from research_experiments.families.free_mad_lite.run.report import render_report, summarize_run
from research_experiments.families.free_mad_lite.run.validate import validate_run
from research_experiments.families.registration_helpers import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.families.shared.config_loading import load_benchmarks, phase_metadata, resolve_model
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
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("free_mad_lite"),
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


REGISTRATION = make_family_registration(
    family_name="free_mad_lite",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="Free-MAD-lite experiment runner.",
        inspect_help="Show resolved Free-MAD-lite config.",
        run_help="Execute one Free-MAD-lite phase.",
        summarize_help="Print Free-MAD-lite summary.",
        validate_help="Validate Free-MAD-lite run.",
        report_help="Regenerate Free-MAD-lite report.",
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
    turn_record_paths=("agent_turns.jsonl", "debate_messages.jsonl"),
    extra_view_paths=("diagnostics.json", "trajectory_scores.jsonl", "paper_summary.csv", "run_summary.json"),
)
