"""baseline_compare 的 family 注册入口。"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from research_experiments.cli_support.output import emit_json
from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.baseline_compare.config import (
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.baseline_compare.run.execute import refresh_run_artifacts, run_experiment
from research_experiments.families.baseline_compare.run.report import render_report, summarize_run
from research_experiments.families.baseline_compare.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    controls = load_control_catalog(experiment.control_catalog)
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "control_catalog": str(experiment.control_catalog),
        "control_method_names": experiment.control_methods,
        "control_methods": {name: asdict(controls[name]) for name in experiment.control_methods},
        "method_order": experiment.method_order,
        "control_prompt_version": experiment.control_prompt_version,
        "mad_prompt_version": experiment.mad_prompt_version,
        "control_output_protocol": experiment.control_output_protocol,
        "mad_initial_output_protocol": experiment.mad_initial_output_protocol,
        "mad_debate_output_protocol": experiment.mad_debate_output_protocol,
        "workspace_defaults": workspace_defaults("baseline_compare"),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "calls_per_question": load_roster_config(setup.roster).agent_count
                * (1 + load_protocol_config(setup.protocol).debate_rounds),
            }
            for setup in experiment.setups
        ],
        "resolved_by_phase": {
            phase_name: {
                "setups": phase["setups"],
                "split_suffix": phase.get("split_suffix"),
                "split_overrides": phase.get("split_overrides"),
            }
            for phase_name, phase in ((name, phase_metadata(experiment, name)) for name in experiment.raw["phases"])
        },
    }


def configure_parser(parser) -> None:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        refresh = action.add_parser(
            "refresh-run-artifacts",
            help="Refresh metrics/diagnostics/report for one completed baseline_compare run.",
        )
        refresh.add_argument("--run-dir", required=True)
        return
    raise RuntimeError("Parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
    if args.command != "refresh-run-artifacts":
        return False
    run_dir = refresh_run_artifacts(args.run_dir)
    emit_json({"run_dir": run_dir.as_posix(), "status": "refreshed"})
    return True


ARTIFACT_ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
    "debate_messages": "turns/debate_messages.jsonl",
    "final_predictions": "views/predictions.jsonl",
    "cost_breakdown": "diagnostics/cost_breakdown.json",
    "debate_diagnostics": "diagnostics/debate_diagnostics.json",
    "output_protocol_diagnostics": "diagnostics/output_protocol_diagnostics.json",
    "baseline_comparison": "exports/baseline_comparison.json",
    "paper_summary": "exports/paper_summary.csv",
    "run_validation": "run_validation.json",
}

REGISTRATION = make_family_registration(
    family_name="baseline_compare",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="Independent baseline comparison runner.",
        inspect_help="Show the resolved baseline comparison experiment configuration.",
        run_help="Execute one baseline comparison phase.",
        summarize_help="Print a concise baseline comparison run summary.",
        validate_help="Run validation checks for one baseline comparison run.",
        report_help="Regenerate the baseline comparison markdown report.",
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
    configure_parser=configure_parser,
    dispatch_extra_command=dispatch_extra_command,
    artifact_aliases=ARTIFACT_ALIASES,
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=("turns/agent_turns.jsonl", "turns/debate_messages.jsonl"),
    diagnostic_paths=(
        "diagnostics/cost_breakdown.json",
        "diagnostics/debate_diagnostics.json",
        "diagnostics/output_protocol_diagnostics.json",
    ),
    export_paths=("exports/baseline_comparison.json", "exports/paper_summary.csv"),
)
