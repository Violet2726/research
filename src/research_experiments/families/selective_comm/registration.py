"""`selective_comm` family 注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.selective_comm.config import (
    describe_backbone_fit,
    ensure_backbone_fit,
    load_control_catalog,
    load_experiment_config,
    load_policies,
    load_protocol_config,
)
from research_experiments.families.selective_comm.run.execute import run_experiment
from research_experiments.families.selective_comm.run.report import render_report, summarize_run
from research_experiments.families.selective_comm.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.registration import (
    make_family_registration,
)
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
        "protocol": {
            "agent_count": protocol.agent_count,
            "debate_rounds": protocol.debate_rounds,
            "initial_temperature": protocol.initial_temperature,
            "debate_temperature": protocol.debate_temperature,
            "top_p": protocol.top_p,
        },
        "policies": [asdict(policy) for policy in policies],
        "controls": {name: asdict(method) for name, method in sorted(controls.items())},
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "workspace_defaults": workspace_defaults("selective_comm"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "model_fit_warnings": describe_backbone_fit(experiment, resolved_model, benchmarks),
        "phases": experiment.raw["phases"],
        "resolved_by_phase": {
            phase_name: {
                "split_suffix": phase.get("split_suffix"),
                "split_overrides": phase.get("split_overrides"),
            }
            for phase_name, phase in (
                (name, phase_metadata(experiment, name))
                for name in experiment.raw["phases"]
            )
        },
    }


def run_from_cli(request) -> object:
    experiment = load_experiment_config(request.experiment_path)
    resolved_model = resolve_model(request.model_ref or experiment.primary_model_ref)
    ensure_backbone_fit(experiment, resolved_model)
    return run_experiment(
        experiment=experiment,
        phase_name=request.phase_name,
        backbone=resolved_model,
        run_root=request.runs_root,
        cache_root=request.cache_root,
        resume_run_dir=request.resume_run_dir,
    )


ARTIFACT_ALIASES = {
    "stage_a_turns": "turns/stage_a_turns.jsonl",
    "stage_b_turns": "turns/stage_b_turns.jsonl",
    "control_turns": "turns/control_turns.jsonl",
    "trigger_decisions": "turns/trigger_decisions.jsonl",
    "policy_predictions": "views/predictions.jsonl",
    "policy_metrics": "views/metrics.json",
    "policy_diagnostics": "diagnostics/policy_diagnostics.json",
    "oracle_trigger_eval": "diagnostics/oracle_trigger_eval.json",
    "run_validation": "run_validation.json",
    "trigger_report": "report.md",
    "policy_reference_summary": "exports/policy_reference_summary.json",
}

REGISTRATION = make_family_registration(
    family_name="selective_comm",
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="选择性通信触发实验运行器。",
        inspect_help="Show the resolved selective communication experiment configuration.",
        run_help="Execute one configured selective communication phase.",
        summarize_help="Print a concise run summary from views/metrics.json.",
        validate_help="Run validation checks for one selective communication run.",
        report_help="Regenerate the Chinese trigger markdown report.",
        include_resume_run_dir=True,
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
    turn_record_paths=('turns/stage_a_turns.jsonl', 'turns/stage_b_turns.jsonl', 'turns/control_turns.jsonl', 'turns/trigger_decisions.jsonl'),
    diagnostic_paths=('diagnostics/oracle_trigger_eval.json', 'diagnostics/policy_diagnostics.json'),
    export_paths=("exports/policy_reference_summary.json",),
)

