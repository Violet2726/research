"""CATCH 实验族注册及显式网络审计、冻结命令。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.core.execution.provider_audit import run_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.replay import replay_from_experiment
from research_experiments.families.contrastive_active_testing.run.execute import (
    _frozen_config_sha,
    _load_frozen_decoding,
    finalize_partial_run_directory,
    run_experiment,
)
from research_experiments.families.contrastive_active_testing.run.report import render_report, summarize_run
from research_experiments.families.contrastive_active_testing.run.validate import validate_run
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults

FAMILY_NAME = "contrastive_active_testing"


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    resolved = resolve_model(model_override or experiment.primary_model_ref)
    protocol = load_protocol_config(experiment.protocol)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "paper_method_name": "CATCH-ICV" if protocol.protocol_version == "catch_v3" else "CATCH",
        "method_version": protocol.protocol_version,
        "protocol": asdict(protocol),
        "benchmarks": [benchmark.slug for benchmark in load_phase_benchmarks(experiment, "development")],
        "phases": {
            name: phase_metadata(experiment, name)
            for name in ("development", "heldout", "confirmation")
        },
        "cache_namespaces": experiment.cache_namespaces,
        "baseline_cache_namespaces": experiment.baseline_cache_namespaces,
        "provider_audit_path": str(experiment.provider_audit_path),
        "frozen_decoding_path": str(experiment.frozen_decoding_path),
        "human_audit_path": str(experiment.human_audit_path),
        "preflight_human_audit_path": str(experiment.preflight_human_audit_path),
        "resolved_model": asdict(resolved),
        "workspace_defaults": workspace_defaults(FAMILY_NAME),
    }


def run_from_cli(request: FamilyRunRequest):
    experiment = load_experiment_config(request.experiment_path)
    backbone = resolve_model(request.model_ref or experiment.primary_model_ref)
    return run_experiment(experiment, request.phase_name, backbone, request.runs_root, request.cache_root)


def configure_parser(parser) -> None:
    for action in parser._actions:
        if not isinstance(action, argparse._SubParsersAction):
            continue
        audit = action.add_parser("provider-audit", help="Run the required uncached MiMo provider contract audit.")
        audit.add_argument("--experiment", required=True)
        audit.add_argument("--output", default=None)
        freeze = action.add_parser("freeze-development", help="Freeze a passing development decoder configuration.")
        freeze.add_argument("--experiment", required=True)
        freeze.add_argument("--run", required=True)
        freeze.add_argument("--output", default=None)
        partial = action.add_parser(
            "finalize-partial",
            help="Finalize a hard-stopped CATCH run as a failed auditable artifact.",
        )
        partial.add_argument("--run", required=True)
        partial.add_argument("--termination-reason", default="futility_gate_impossible")
        preflight = action.add_parser(
            "structural-preflight",
            help="Run the one-shot CATCH-v3 20-disagreement structural preflight and terminate.",
        )
        preflight.add_argument("--experiment", required=True)
        preflight.add_argument("--runs-root", default=None)
        preflight.add_argument("--cache-root", default=None)
        replay = action.add_parser(
            "canonicalization-replay",
            help="Rejudge archived Stage-A/adaptive responses without network access.",
        )
        replay.add_argument("--experiment", required=True)
        replay.add_argument("--run", required=True, action="append")
        replay.add_argument("--output", required=True)
        return
    raise RuntimeError("CATCH parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
    if args.command == "provider-audit":
        _run_provider_audit(args)
        return True
    if args.command == "freeze-development":
        _freeze_development(args)
        return True
    if args.command == "finalize-partial":
        print(
            json.dumps(
                finalize_partial_run_directory(
                    args.run,
                    termination_reason=args.termination_reason,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    if args.command == "structural-preflight":
        experiment = load_experiment_config(args.experiment)
        backbone = resolve_model(experiment.primary_model_ref)
        result = run_experiment(
            experiment,
            "development",
            backbone,
            args.runs_root,
            args.cache_root,
            run_mode="structural_preflight",
        )
        print(json.dumps({"run_dir": str(result), "run_mode": "structural_preflight"}, indent=2))
        return True
    if args.command == "canonicalization-replay":
        experiment = load_experiment_config(args.experiment)
        payload = replay_from_experiment(args.run, experiment, output_path=args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "passed": payload["passed"],
                    "network_requests": payload["network_requests"],
                    "metrics": payload["metrics"],
                    "feasibility_conditions": payload["feasibility_conditions"],
                    "changed_turn_count": payload["changed_turn_count"],
                    "hashes": payload["hashes"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    return False


def _run_provider_audit(args) -> None:
    experiment = load_experiment_config(args.experiment)
    backbone = resolve_model(experiment.primary_model_ref)
    if backbone.provider != "xiaomimimo":
        raise ValueError("CATCH provider audit is defined only for xiaomimimo.")
    load_dotenv(".env.local", override=False)
    provider = OpenAICompatibleProvider(backbone)
    try:
        payload = run_mimo_provider_audit(
            backbone=backbone,
            provider=provider,
            cache_namespace=experiment.cache_namespaces["provider_audit"],
        )
    finally:
        provider.close()
    target = Path(args.output or experiment.provider_audit_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def _freeze_development(args) -> None:
    experiment = load_experiment_config(args.experiment)
    run_dir = Path(args.run)
    gate_path = run_dir / "diagnostics" / "gate.json"
    candidate_path = run_dir / "diagnostics" / "frozen_decoding_candidate.json"
    validation_path = run_dir / "run_validation.json"
    if not gate_path.exists() or not candidate_path.exists() or not validation_path.exists():
        raise RuntimeError("CATCH freeze requires gate, frozen candidate, and run validation artifacts.")
    gate = json.loads(gate_path.read_text(encoding="utf-8"))
    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    _load_frozen_decoding(candidate_path, config_sha=_frozen_config_sha(experiment))
    if not gate.get("passed") or not validation.get("passed") or not candidate.get("selection_constraints_passed"):
        raise RuntimeError("CATCH development run did not pass every freeze requirement.")
    target = Path(args.output or experiment.frozen_decoding_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(candidate, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"frozen": True, "output": str(target), "sha256": candidate.get("sha256")}, indent=2))


REGISTRATION = make_family_registration(
    family_name=FAMILY_NAME,
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="CATCH: contrastive active testing with error-correcting candidate decoding.",
        inspect_help="Show the frozen CATCH protocol and gate configuration.",
        run_help="Run a gated CATCH development, held-out, or confirmation phase.",
        summarize_help="Print a CATCH run summary.",
        validate_help="Validate CATCH artifacts, usage, and cache isolation.",
        report_help="Return or regenerate the CATCH report.",
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
        "preflight_turns": "turns/preflight_turns.jsonl",
        "gate": "diagnostics/gate.json",
        "preflight": "diagnostics/preflight.json",
        "frozen_decoding_candidate": "diagnostics/frozen_decoding_candidate.json",
        "canonicalization_replay": "diagnostics/canonicalization_replay.json",
    },
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=(
        "turns/agent_turns.jsonl",
        "turns/router_decisions.jsonl",
        "turns/preflight_turns.jsonl",
    ),
    diagnostic_paths=(
        "diagnostics/gate.json",
        "diagnostics/preflight.json",
        "diagnostics/frozen_decoding_candidate.json",
        "diagnostics/canonicalization_replay.json",
    ),
)
