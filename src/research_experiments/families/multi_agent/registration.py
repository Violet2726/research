"""`multi_agent` family registration."""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.multi_agent.config import (
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.multi_agent.run.execute import run_experiment
from research_experiments.families.multi_agent.run.report import render_report, summarize_run
from research_experiments.families.multi_agent.run.validate import validate_run
from research_experiments.families.registration_helpers import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.core.families.config_loading import load_benchmarks, phase_metadata, resolve_model
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    controls = load_control_catalog(experiment.control_catalog)
    benchmarks = load_benchmarks(experiment)
    payload = {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "control_catalog": str(experiment.control_catalog),
        "control_methods": {name: asdict(method) for name, method in sorted(controls.items())},
        "workspace_defaults": workspace_defaults("multi_agent"),
        "primary_model_ref": experiment.primary_model_ref,
        "phases": experiment.raw["phases"],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
                "calls_per_question": load_roster_config(setup.roster).agent_count
                * (1 + load_protocol_config(setup.protocol).debate_rounds),
            }
            for setup in experiment.setups
        ],
    }
    payload["resolved_model"] = asdict(resolve_model(model_override or experiment.primary_model_ref))
    payload["resolved_by_phase"] = {
        phase_name: {
            "setups": phase["setups"],
            "split_suffix": phase.get("split_suffix"),
            "split_overrides": phase.get("split_overrides"),
        }
        for phase_name, phase in (
            (name, phase_metadata(experiment, name))
            for name in experiment.raw["phases"]
        )
    }
    return payload


REGISTRATION = make_family_registration(
    family_name="multi_agent",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="Vanilla MAD multi-agent baseline runner.",
        inspect_help="Show the resolved multi-agent experiment configuration.",
        run_help="Execute one configured multi-agent experiment phase.",
        summarize_help="Print a concise run summary from metrics.json.",
        validate_help="Run validation checks for one multi-agent run.",
        report_help="Generate paired Debate vs Vote analysis and a Chinese markdown report.",
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
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=('turns/agent_turns.jsonl', 'turns/debate_messages.jsonl'),
    diagnostic_paths=('diagnostics/cost_breakdown.json', 'diagnostics/debate_diagnostics.json'),
    export_paths=(),
)

