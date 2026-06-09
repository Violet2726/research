"""`imad` family 注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.imad.config import (
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.imad.run.execute import run_experiment
from research_experiments.families.imad.run.report import render_report, summarize_run
from research_experiments.families.imad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
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
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in experiment.methods],
        "controls": {name: asdict(method) for name, method in sorted(controls.items())},
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("imad"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": {
            phase_name: phase_metadata(experiment, phase_name)
            for phase_name in experiment.raw["phases"]
        },
    }


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"debate_messages": "turns/debate_messages.jsonl",
"round_diagnostics": "turns/round_diagnostics.jsonl",
"final_predictions": "views/predictions.jsonl",
"cost_breakdown": "diagnostics/cost_breakdown.json",
"stability_diagnostics": "diagnostics/stability_diagnostics.json",
"run_validation": "run_validation.json",
}

REGISTRATION = make_family_registration(
    family_name="imad",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="自适应停止辩论实验运行器。",
        inspect_help="Show the resolved iMAD experiment configuration.",
        run_help="Execute one configured iMAD experiment phase.",
        summarize_help="Print a concise iMAD run summary.",
        validate_help="Run iMAD validation checks.",
        report_help="Regenerate the Chinese iMAD markdown report.",
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
    turn_record_paths=('turns/agent_turns.jsonl', 'turns/debate_messages.jsonl', 'turns/round_diagnostics.jsonl'),
    diagnostic_paths=('diagnostics/cost_breakdown.json', 'diagnostics/stability_diagnostics.json'),
    export_paths=(),
)

