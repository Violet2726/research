"""`dmad` family 注册入口。"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.dmad.config import (
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.dmad.run.execute import run_experiment
from research_experiments.families.dmad.run.report import render_report, summarize_run
from research_experiments.families.dmad.run.validate import validate_run
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
    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog is not None else {}
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "evaluation_scope": experiment.evaluation_scope,
        "paper_alignment_version": experiment.paper_alignment_version,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": asdict(protocol),
        "control_catalog": None if experiment.control_catalog is None else experiment.control_catalog.as_posix(),
        "controls": {name: asdict(method) for name, method in controls.items()},
        "methods": [
            {
                "name": method.name,
                "mode": method.mode,
                "roster": None if method.roster is None else method.roster.as_posix(),
                "debate_call_style": method.debate_call_style,
                "note": method.note,
                "matched_controls": list(method.matched_controls),
                "roster_config": None if method.roster is None else asdict(load_roster_config(method.roster)),
            }
            for method in experiment.methods
        ],
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults("dmad"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
"debate_messages": "turns/debate_messages.jsonl",
"final_predictions": "views/predictions.jsonl",
"strategy_diagnostics": "diagnostics/strategy_diagnostics.json",
"cost_breakdown": "diagnostics/cost_breakdown.json",
"paper_tables": "exports/paper_tables.json",
"run_validation": "run_validation.json",
}

REGISTRATION = make_family_registration(
    family_name="dmad",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="DMAD 论文主线高保真复现实验 runner.",
        inspect_help="Show the resolved DMAD experiment configuration.",
        run_help="Execute one configured DMAD experiment phase.",
        summarize_help="Print a concise DMAD run summary.",
        validate_help="Run DMAD validation checks.",
        report_help="Regenerate the Chinese DMAD markdown report.",
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
    turn_record_paths=('turns/agent_turns.jsonl', 'turns/debate_messages.jsonl'),
    diagnostic_paths=('diagnostics/cost_breakdown.json', 'diagnostics/strategy_diagnostics.json'),
    export_paths=('exports/paper_tables.json',),
)

