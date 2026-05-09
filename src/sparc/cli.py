"""SPARC 命令行入口。

CLI 暴露实验检查、执行、摘要、校验与报告生成能力，
方便在内容压缩、局部审计和完整 SPARC 方案之间切换分析。
"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from experiment_core.foundation.workspace import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
    workspace_defaults,
)
from sparc.config import load_benchmarks, load_experiment_config, load_protocol_config, phase_metadata, resolve_model
from sparc.reporting import render_report, summarize_run
from sparc.runner import run_experiment
from sparc.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="SPARC v1 experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved SPARC experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help="Execute one configured SPARC experiment phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root("sparc"))
    run.add_argument("--cache-root", default=default_cache_root())

    summarize = subparsers.add_parser("summarize-run", help="Print a concise SPARC run summary.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Run SPARC validation checks.")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--compare-run-dir", default=None)

    report = subparsers.add_parser("render-report", help="Regenerate the Chinese SPARC markdown report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root("sparc"))

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        protocol = load_protocol_config(experiment.protocol)
        resolved_model = resolve_model(args.model or experiment.primary_model_ref)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "experiment_kind": experiment.experiment_kind,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
            "protocol": {
                "agent_count": protocol.agent_count,
                "debate_rounds": protocol.debate_rounds,
                "initial_temperature": protocol.initial_temperature,
                "debate_temperature": protocol.debate_temperature,
                "top_p": protocol.top_p,
                "max_output_tokens": protocol.max_output_tokens,
            },
            "global_seed": experiment.global_seed,
            "prompt_version": experiment.prompt_version,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "workspace_defaults": workspace_defaults("sparc"),
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": {
                "name": resolved_model.name,
                "provider": resolved_model.provider,
                "model_id": resolved_model.model_id,
                "tags": resolved_model.tags,
            },
            "fixed_trigger_policy": experiment.fixed_trigger_policy,
            "message_modes": experiment.message_modes,
            "fixed_message_modes": experiment.fixed_message_modes,
            "aggregation_methods": experiment.aggregation_methods,
            "default_trigger_policy": experiment.default_trigger_policy,
            "fallback_trigger_policy": experiment.fallback_trigger_policy,
            "trigger_drop_questions": experiment.trigger_drop_questions,
            "phases": {
                phase_name: phase_metadata(experiment, phase_name)
                for phase_name in experiment.raw["phases"]
            },
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        backbone = resolve_model(args.model or experiment.primary_model_ref)
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            backbone=backbone,
            run_root=args.runs_root,
            cache_root=args.cache_root,
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-run":
        print(json.dumps(validate_run(args.run_dir, compare_run_dir=args.compare_run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "render-report":
        print(json.dumps(render_report(args.run_dir, publish_dir=args.publish_dir), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

