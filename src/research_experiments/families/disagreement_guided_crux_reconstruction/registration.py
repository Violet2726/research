"""DGCR 的注册信息与 CLI 扩展。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.core.execution.provider_audit import run_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.families.disagreement_guided_crux_reconstruction.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.disagreement_guided_crux_reconstruction.run.execute import run_experiment
from research_experiments.families.disagreement_guided_crux_reconstruction.run.report import (
    render_report,
    summarize_run,
)
from research_experiments.families.disagreement_guided_crux_reconstruction.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults

FAMILY_NAME = "disagreement_guided_crux_reconstruction"


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    resolved = resolve_model(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "protocol": asdict(load_protocol_config(experiment.protocol)),
        "benchmarks": [benchmark.slug for benchmark in load_phase_benchmarks(experiment, "development")],
        "phases": {name: phase_metadata(experiment, name) for name in ("development", "heldout")},
        "cache_policy": experiment.cache_policy,
        "provider_audit_path": str(experiment.provider_audit_path),
        "resolved_model": asdict(resolved),
        "workspace_defaults": workspace_defaults(FAMILY_NAME),
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    backbone = resolve_model(request.model_ref or experiment.primary_model_ref)
    return run_experiment(experiment, request.phase_name, backbone, request.runs_root, request.cache_root)


def configure_parser(parser) -> None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            audit = action.add_parser("provider-audit", help="Run the required uncached MiMo provider contract audit.")
            audit.add_argument("--experiment", required=True)
            audit.add_argument("--output", default=None)
            return
    raise RuntimeError("DGCR parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
    if args.command != "provider-audit":
        return False
    experiment = load_experiment_config(args.experiment)
    backbone = resolve_model(experiment.primary_model_ref)
    if backbone.provider != "xiaomimimo":
        raise ValueError("DGCR provider audit is defined only for xiaomimimo.")
    load_dotenv(".env.local", override=False)
    provider = OpenAICompatibleProvider(backbone)
    try:
        payload = run_mimo_provider_audit(
            backbone=backbone,
            provider=provider,
        )
    finally:
        provider.close()
    target = Path(args.output or experiment.provider_audit_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return True


REGISTRATION = make_family_registration(
    family_name=FAMILY_NAME,
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="Disagreement-Guided Crux Reconstruction experiments.",
        inspect_help="Show the frozen DGCR configuration.",
        run_help="Run a gated DGCR development or held-out phase.",
        summarize_help="Print a DGCR run summary.",
        validate_help="Validate DGCR artifacts and cache isolation.",
        report_help="Return the generated DGCR report.",
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
    artifact_aliases={
        "agent_turns": "turns/agent_turns.jsonl",
        "router_decisions": "turns/router_decisions.jsonl",
        "gate": "diagnostics/gate.json",
    },
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=("turns/agent_turns.jsonl", "turns/router_decisions.jsonl"),
    diagnostic_paths=("diagnostics/gate.json",),
)
