"""`consensagent` family 注册入口。

本模块把 CONSENSAGENT 触发式辩论实验接入统一 family CLI，
集中登记 inspect、run、report、validate 与产物别名。
"""

from __future__ import annotations

from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.consensagent.config import (
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)
from research_experiments.families.consensagent.run.execute import run_experiment
from research_experiments.families.consensagent.run.report import render_report, summarize_run
from research_experiments.families.consensagent.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import load_benchmarks, resolve_model
from research_experiments.family_runtime.registration import (
    build_backbone_run_from_cli,
    make_family_registration,
)
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    """返回 CLI inspect-experiment 使用的解析后配置视图。"""
    experiment = load_experiment_config(experiment_path)
    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog else {}
    benchmarks = load_benchmarks(experiment)
    payload = {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "control_catalog": str(experiment.control_catalog) if experiment.control_catalog else None,
        "control_methods": {name: asdict(method) for name, method in sorted(controls.items())} if controls else {},
        "workspace_defaults": workspace_defaults("consensagent"),
        "primary_model_ref": experiment.primary_model_ref,
        "phases": experiment.raw["phases"],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
                "calls_per_question": load_roster_config(setup.roster).agent_count
                * (1 + load_protocol_config(setup.protocol).max_debate_rounds),
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


ARTIFACT_ALIASES = {
    "run_root": ".",
    "turns_path": "turns/turns.jsonl",
    "debate_messages_path": "turns/debate_messages.jsonl",
    "predictions_path": "views/predictions.jsonl",
    "metrics_path": "views/metrics.json",
    "cost_breakdown_path": "diagnostics/cost_breakdown.json",
    "debate_diagnostics_path": "diagnostics/debate_diagnostics.json",
}

REGISTRATION = make_family_registration(
    family_name="consensagent",
    prototype="debate_rounds",
    cli_help=FamilyCliHelp(
        description="CONSENSAGENT：带 sycophancy mitigation 的触发式多智能体辩论运行器。",
        inspect_help="Show the resolved CONSENSAGENT experiment configuration.",
        run_help="Execute one configured CONSENSAGENT experiment phase.",
        summarize_help="Print a concise run summary from metrics.json.",
        validate_help="Run validation checks for one CONSENSAGENT run.",
        report_help="Generate CONSENSAGENT analysis and a Chinese markdown report.",
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
    turn_record_paths=("turns/turns.jsonl", "turns/debate_messages.jsonl"),
    diagnostic_paths=("diagnostics/cost_breakdown.json", "diagnostics/debate_diagnostics.json"),
    export_paths=(),
)

