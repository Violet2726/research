"""统一 MAD 创新 family 注册与版本命令。"""

from __future__ import annotations

import argparse
from dataclasses import asdict

from research_experiments.cli_support.output import emit_json
from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.risk_controlled_trace_mad.config import (
    HsgsaProtocolConfig,
    inspect_benchmarks,
    inspect_methods,
    load_experiment_config,
    load_protocol_config,
    load_version_registry,
    require_active_version,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    EVF_AUDIT_SCHEMA_VERSION,
    EVF_PROMPT_VERSION,
    HSGSA_PROMPT_VERSION,
    HSGSA_REVIEW_SCHEMA_VERSION,
)
from research_experiments.families.risk_controlled_trace_mad.replay import write_replay_audit
from research_experiments.families.risk_controlled_trace_mad.run.execute import run_experiment
from research_experiments.families.risk_controlled_trace_mad.run.report import render_report, summarize_run
from research_experiments.families.risk_controlled_trace_mad.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults

FAMILY_NAME = "risk_controlled_trace_mad"


def inspect_experiment(path: str, model_override: str | None):
    experiment = load_experiment_config(path)
    if model_override and model_override != experiment.primary_model_ref:
        raise ValueError("The active protocol uses frozen MiMo-v2.5; --model cannot replace it.")
    protocol = load_protocol_config(experiment.protocol)
    registry = load_version_registry(experiment.version_registry)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "active_version": experiment.active_version,
        "versions": {key: asdict(value) for key, value in registry.versions.items()},
        "benchmarks": inspect_benchmarks(experiment),
        "protocol": asdict(protocol),
        "methods": inspect_methods(experiment),
        "method_order": experiment.method_order,
        "prompt_version": HSGSA_PROMPT_VERSION if isinstance(protocol, HsgsaProtocolConfig) else EVF_PROMPT_VERSION,
        "schema_version": (
            HSGSA_REVIEW_SCHEMA_VERSION if isinstance(protocol, HsgsaProtocolConfig) else EVF_AUDIT_SCHEMA_VERSION
        ),
        "global_seed": experiment.global_seed,
        "model_roster": {"primary": asdict(resolve_model(experiment.primary_model_ref))},
        "runtime_profiles": {key: asdict(value) for key, value in experiment.runtime_profiles.items()},
        "workspace_defaults": workspace_defaults(FAMILY_NAME),
        "phases": experiment.raw["phases"],
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    registry = load_version_registry(experiment.version_registry)
    require_active_version(registry, request.version)
    if request.model_ref and request.model_ref != experiment.primary_model_ref:
        raise ValueError("The active protocol uses frozen MiMo-v2.5; omit --model.")
    return run_experiment(
        experiment=experiment,
        phase_name=request.phase_name,
        backbone=resolve_model(experiment.primary_model_ref),
        run_root=request.runs_root,
        cache_root=request.cache_root,
        resume_run_dir=request.resume_run_dir,
        version=request.version,
    )


def configure_parser(parser) -> None:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        action.choices["run"].add_argument("--version", default=None)
        versions = action.add_parser("inspect-versions", help="Inspect active and retired MAD innovation versions.")
        versions.add_argument("--experiment", required=True)
        replay = action.add_parser("replay-dev", help="Replay the frozen historical BBEH-300 gate without API calls.")
        replay.add_argument("--experiment", required=True)
        replay.add_argument(
            "--output",
            default="configs/families/risk_controlled_trace_mad/retired/v5_hsgsa_dev_replay_audit.json",
        )
        return
    raise RuntimeError("Parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
    if args.command == "replay-dev":
        experiment = load_experiment_config(args.experiment)
        phase = dict(experiment.raw["phases"]["replay_dev_seed42"])
        emit_json(write_replay_audit(str(phase["source_run"]), args.output))
        return True
    if args.command != "inspect-versions":
        return False
    experiment = load_experiment_config(args.experiment)
    registry = load_version_registry(experiment.version_registry)
    emit_json(
        {
            "active_version": registry.active_version,
            "versions": {key: asdict(value) for key, value in registry.versions.items()},
        }
    )
    return True


ALIASES = {
    "agent_turns": "turns/agent_turns.jsonl",
    "debate_messages": "turns/debate_messages.jsonl",
    "router_decisions": "turns/router_decisions.jsonl",
    "evf_diagnostics": "diagnostics/evf_diagnostics.json",
    "hsgsa_diagnostics": "diagnostics/hsgsa_diagnostics.json",
    "paired_statistics": "diagnostics/paired_statistics.json",
    "output_protocol_diagnostics": "diagnostics/output_protocol_diagnostics.json",
    "evf_comparison": "exports/evf_comparison.json",
    "hsgsa_comparison": "exports/hsgsa_comparison.json",
    "paper_summary": "exports/paper_summary.csv",
}

REGISTRATION = make_family_registration(
    family_name=FAMILY_NAME,
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="唯一 MAD 创新主线：同质、支持度盲化 H-SGSA v5。",
        inspect_help="Show the frozen homogeneous H-SGSA experiment configuration.",
        run_help="Run the active MAD innovation version.",
        summarize_help="Summarize an H-SGSA run.",
        validate_help="Validate H-SGSA artifacts, safety, splits, and budgets.",
        report_help="Regenerate the manifest-driven H-SGSA report.",
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
    configure_parser=configure_parser,
    dispatch_extra_command=dispatch_extra_command,
    artifact_aliases=ALIASES,
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=("turns/agent_turns.jsonl", "turns/debate_messages.jsonl", "turns/router_decisions.jsonl"),
    diagnostic_paths=(
        "diagnostics/evf_diagnostics.json",
        "diagnostics/hsgsa_diagnostics.json",
        "diagnostics/paired_statistics.json",
        "diagnostics/output_protocol_diagnostics.json",
    ),
    export_paths=("exports/evf_comparison.json", "exports/hsgsa_comparison.json", "exports/paper_summary.csv"),
)
