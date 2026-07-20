"""CATCH 实验族注册及显式网络审计、冻结命令。"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp, FamilyRunRequest
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.mechanism_factorial import (
    summarize_factorial_results,
    write_factorial_template,
)
from research_experiments.families.contrastive_active_testing.replay import replay_from_experiment
from research_experiments.families.contrastive_active_testing.run.execute import run_experiment
from research_experiments.families.contrastive_active_testing.run.report import render_report, summarize_run
from research_experiments.families.contrastive_active_testing.run.validate import validate_run
from research_experiments.families.contrastive_active_testing.v1_failure_audit import write_v1_mechanism_audit
from research_experiments.families.contrastive_active_testing.v2_readiness import (
    write_cert_v2_readiness_assessment,
)
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.registration import make_family_registration
from research_experiments.workspace.layout import workspace_defaults

FAMILY_NAME = "contrastive_active_testing"


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    resolved = resolve_model(model_override or experiment.primary_model_ref)
    protocol = load_protocol_config(experiment.protocol)
    phase_names = (
        ("boundary_audit",)
        if experiment.study_type == "post_failure_cross_domain_boundary_audit"
        else ("development", "heldout", "confirmation")
    )
    return {
        "name": experiment.name,
        "description": experiment.description,
        "paper_method_name": (
            "CATCH-Cert v2"
            if protocol.protocol_version == "catch_cert_v2"
            else "CATCH-Cert"
            if protocol.protocol_version == "catch_cert_v1"
            else "CATCH-ICV"
            if protocol.protocol_version == "catch_v3"
            else "CATCH"
        ),
        "method_version": protocol.protocol_version,
        "protocol": asdict(protocol),
        "benchmarks": [benchmark.slug for benchmark in load_phase_benchmarks(experiment, phase_names[0])],
        "study_type": experiment.study_type,
        "confirmatory": experiment.confirmatory,
        "execution_policy": "best_effort_non_blocking",
        "config_warnings": list(experiment.config_warnings),
        "phases": {name: phase_metadata(experiment, name) for name in phase_names},
        "cache_namespaces": experiment.cache_namespaces,
        "baseline_cache_namespaces": experiment.baseline_cache_namespaces,
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
        replay = action.add_parser(
            "canonicalization-replay",
            help="Legacy offline rejudging tool for archived CATCH runs.",
        )
        replay.add_argument("--experiment", required=True)
        replay.add_argument("--run", required=True, action="append")
        replay.add_argument("--output", required=True)
        audit = action.add_parser(
            "cert-v1-audit",
            help="Build the 120-case Chinese CATCH-Cert v1 mechanism-audit queue without API calls.",
        )
        audit.add_argument("--run", required=True)
        audit.add_argument("--output-dir", required=True)
        readiness = action.add_parser(
            "assess-cert-v2-readiness",
            help="Write a non-blocking development/audit diagnostic for interpreting later CATCH-Cert v2 runs.",
        )
        readiness.add_argument("--run", required=True)
        readiness.add_argument("--audit", required=True)
        readiness.add_argument("--output", required=True)
        readiness_compat = action.add_parser(
            "freeze-cert-v2-readiness",
            help="Deprecated alias of assess-cert-v2-readiness; it does not block later runs.",
        )
        readiness_compat.add_argument("--run", required=True)
        readiness_compat.add_argument("--audit", required=True)
        readiness_compat.add_argument("--output", required=True)
        factorial = action.add_parser(
            "cert-v2-factorial-template",
            help="Create the frozen 120-case by four-cell mechanism audit template.",
        )
        factorial.add_argument("--audit", required=True)
        factorial.add_argument("--output", required=True)
        factorial_summary = action.add_parser(
            "summarize-cert-v2-factorial",
            help="Summarize a completed CATCH-Cert v2 2x2 mechanism audit.",
        )
        factorial_summary.add_argument("--results", required=True)
        factorial_summary.add_argument("--output", required=True)
        return
    raise RuntimeError("CATCH parser is missing subcommands.")


def dispatch_extra_command(args) -> bool:
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
    if args.command == "cert-v1-audit":
        payload = write_v1_mechanism_audit(args.run, args.output_dir)
        print(
            json.dumps(
                {
                    "output_dir": str(args.output_dir),
                    "actual_counts": payload["actual_counts"],
                    "automatic_error_counts": payload["automatic_error_counts"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    if args.command in {"assess-cert-v2-readiness", "freeze-cert-v2-readiness"}:
        payload = write_cert_v2_readiness_assessment(args.run, args.audit, args.output)
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "enforcement": payload["enforcement"],
                    "blocks_execution": payload["blocks_execution"],
                    "all_recommended_conditions_met": payload[
                        "all_recommended_conditions_met"
                    ],
                    "unmet_conditions": payload["unmet_conditions"],
                    "conditions": payload["conditions"],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    if args.command == "cert-v2-factorial-template":
        payload = write_factorial_template(args.audit, args.output)
        print(json.dumps({"output": args.output, "rows": payload["row_count"]}, ensure_ascii=False, indent=2))
        return True
    if args.command == "summarize-cert-v2-factorial":
        payload = summarize_factorial_results(args.results, args.output)
        print(
            json.dumps(
                {"output": args.output, "paper_branch": payload["paper_branch"], "attribution": payload["attribution"]},
                ensure_ascii=False,
                indent=2,
            )
        )
        return True
    return False


REGISTRATION = make_family_registration(
    family_name=FAMILY_NAME,
    prototype="shared_stage_policy",
    cli_help=FamilyCliHelp(
        description="CATCH: contrastive active testing with error-correcting candidate decoding.",
        inspect_help="Show the CATCH protocol and execution configuration.",
        run_help="Run a best-effort CATCH phase without prerequisite gates.",
        summarize_help="Print a CATCH run summary.",
        validate_help="Summarize CATCH result readability and recoverable execution warnings.",
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
        "screening_samples": "diagnostics/screening_samples.jsonl",
        "dataset_checkpoints": "diagnostics/dataset_checkpoints.json",
        "boundary_human_audit_sample": "diagnostics/human_audit_sample.json",
        "boundary_human_audit_completed": "diagnostics/human_audit_completed.json",
        "boundary_human_audit_evaluation": "diagnostics/human_audit_evaluation.json",
        "selector_funnel": "selector_funnel.json",
        "witness_analysis": "witness_analysis.json",
        "sample_outcomes": "sample_outcomes.jsonl",
        "boundary_reproducibility_manifest": "reproducibility_manifest.json",
        "failure_cases": "failure_cases.md",
        "boundary_index": "index.md",
    },
    metrics_view_path="views/metrics.json",
    prediction_records_path="views/predictions.jsonl",
    turn_record_paths=(
        "turns/agent_turns.jsonl",
        "turns/router_decisions.jsonl",
    ),
    diagnostic_paths=(),
)
