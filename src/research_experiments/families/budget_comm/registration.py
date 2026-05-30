"""`budget_comm` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.budget_comm.config import (
    load_auction_policy_config,
    load_context_view_config,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.budget_comm.run.execute import run_experiment
from research_experiments.families.budget_comm.run.report import render_report, summarize_run
from research_experiments.families.budget_comm.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    auction_policy = load_auction_policy_config(experiment.auction_policy)
    context_view = load_context_view_config(experiment.context_view)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "auction_policy": asdict(auction_policy),
        "context_view": asdict(context_view),
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "calibration_sample_size": experiment.calibration_sample_size,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("budget_comm"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": {
            "name": resolved_model.name,
            "provider": resolved_model.provider,
            "model_id": resolved_model.model_id,
            "tags": resolved_model.tags,
        },
        "phases": {
            phase_name: phase_metadata(experiment, phase_name)
            for phase_name in experiment.raw["phases"]
        },
    }


ARTIFACT_ALIASES = {
    "sample_views": "turns/sample_views.jsonl",
"stage_a_turns": "turns/stage_a_turns.jsonl",
"candidate_packets": "turns/candidate_packets.jsonl",
"auction_decisions": "turns/auction_decisions.jsonl",
"belief_updates": "turns/belief_updates.jsonl",
"final_predictions": "views/predictions.jsonl",
"budget_diagnostics": "diagnostics/budget_diagnostics.json",
"run_validation": "run_validation.json",
"report_markdown": "report.md",
"paper_summary": "exports/paper_summary.csv",
}

REGISTRATION = make_family_registration(
    family_name="budget_comm",
    prototype="packet_belief_update",
    cli_help=FamilyCliHelp(
        description="Budget-aware DALA-lite experiment runner.",
        inspect_help="Show the resolved budget_comm experiment configuration.",
        run_help="Execute one configured budget_comm experiment phase.",
        summarize_help="Print a concise budget_comm run summary.",
        validate_help="Run budget_comm validation checks.",
        report_help="Regenerate the Chinese budget_comm markdown report.",
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
    turn_record_paths=('turns/sample_views.jsonl', 'turns/stage_a_turns.jsonl', 'turns/candidate_packets.jsonl', 'turns/auction_decisions.jsonl', 'turns/belief_updates.jsonl'),
    diagnostic_paths=('diagnostics/budget_diagnostics.json',),
    export_paths=('exports/paper_summary.csv',),
)

