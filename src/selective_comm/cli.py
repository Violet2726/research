from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from experiment_core.workspace import (
    default_cache_path,
    default_reports_root,
    default_runs_root,
    workspace_defaults,
)
from selective_comm.config import (
    describe_backbone_fit,
    ensure_backbone_fit,
    load_benchmarks,
    load_control_catalog,
    load_experiment_config,
    load_policies,
    load_protocol_config,
    phase_metadata,
    resolve_backbone,
)
from selective_comm.reporting import render_trigger_report, summarize_run
from selective_comm.runner import run_experiment
from selective_comm.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="Selective communication trigger experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved selective communication experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--backbone", default=None)

    run = subparsers.add_parser("run", help="Execute one configured selective communication phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--backbone", default=None)
    run.add_argument("--runs-root", default=default_runs_root("selective_comm"))
    run.add_argument("--cache-path", default=default_cache_path("selective_comm"))
    run.add_argument("--resume-run-dir", default=None)

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary from policy_metrics.json.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Run validation checks for one selective communication run.")
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("report-trigger", help="Regenerate the Chinese trigger markdown report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root("selective_comm"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        benchmarks = load_benchmarks(experiment)
        protocol = load_protocol_config(experiment.protocol)
        policies = load_policies(experiment.policy_configs)
        controls = load_control_catalog(experiment.control_catalog)
        resolved_backbone = resolve_backbone(args.backbone or experiment.primary_backbone)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in benchmarks],
            "protocol": _serialize_protocol(protocol),
            "policies": [_serialize_policy(policy) for policy in policies],
            "controls": {name: _serialize_control(method) for name, method in sorted(controls.items())},
            "prompt_version": experiment.prompt_version,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "workspace_defaults": workspace_defaults("selective_comm"),
            "primary_backbone": experiment.primary_backbone,
            "resolved_backbone": _serialize_backbone(resolved_backbone),
            "backbone_fit_warnings": describe_backbone_fit(experiment, resolved_backbone, benchmarks),
            "phases": experiment.raw["phases"],
        }
        for phase_name in experiment.raw["phases"]:
            phase = phase_metadata(experiment, phase_name)
            payload.setdefault("resolved_by_phase", {})[phase_name] = {
                "split_suffix": phase.get("split_suffix"),
                "split_overrides": phase.get("split_overrides"),
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        backbone_ref = args.backbone or experiment.primary_backbone
        resolved_backbone = resolve_backbone(backbone_ref)
        ensure_backbone_fit(experiment, resolved_backbone)
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            backbone=resolved_backbone,
            run_root=args.runs_root,
            cache_path=args.cache_path,
            resume_run_dir=args.resume_run_dir,
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-run":
        print(json.dumps(validate_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "report-trigger":
        print(json.dumps(render_trigger_report(args.run_dir, publish_dir=args.publish_dir), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


def _serialize_protocol(protocol: object) -> dict[str, object]:
    return {
        "agent_count": protocol.agent_count,
        "debate_rounds": protocol.debate_rounds,
        "initial_temperature": protocol.initial_temperature,
        "debate_temperature": protocol.debate_temperature,
        "top_p": protocol.top_p,
        "max_output_tokens": protocol.max_output_tokens,
    }


def _serialize_policy(policy: object) -> dict[str, object]:
    return {
        "policy_name": policy.policy_name,
        "trigger_type": policy.trigger_type,
        "mean_conf_threshold": policy.mean_conf_threshold,
        "conf_spread_threshold": policy.conf_spread_threshold,
        "claim_divergence_threshold": policy.claim_divergence_threshold,
        "uncertainty_type_diversity_threshold": policy.uncertainty_type_diversity_threshold,
        "fail_open_to_always": policy.fail_open_to_always,
    }


def _serialize_control(method: object) -> dict[str, object]:
    return {
        "family": method.family,
        "budget_calls": method.budget_calls,
        "temperature": method.temperature,
        "top_p": method.top_p,
        "max_output_tokens": method.max_output_tokens,
    }


def _serialize_backbone(backbone: object) -> dict[str, object]:
    return {
        "name": backbone.name,
        "provider": backbone.provider,
        "model_id": backbone.model_id,
        "base_url": backbone.base_url,
        "api_key_env": backbone.api_key_env,
        "chat_path": backbone.chat_path,
        "default_temperature": backbone.default_temperature,
        "default_top_p": backbone.default_top_p,
        "default_max_output_tokens": backbone.default_max_output_tokens,
        "reasoning_effort": backbone.reasoning_effort,
        "supports_response_format": backbone.supports_response_format,
        "response_format": backbone.response_format,
        "timeout_seconds": backbone.timeout_seconds,
        "max_retries": backbone.max_retries,
        "tags": backbone.tags,
    }


if __name__ == "__main__":
    main()
