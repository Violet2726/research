"""Free-MAD-lite 命令行入口。

CLI 暴露实验检查、执行、摘要、校验与报告生成能力，
方便单独运行 Free-MAD-lite 机制验证实验。
"""

from __future__ import annotations

from dataclasses import asdict
import argparse

from dotenv import load_dotenv

from experiment_core.foundation.cli_output import configure_utf8_stdio, emit_json
from experiment_core.foundation.workspace import (
    default_cache_root,
    default_reports_root,
    default_runs_root,
    workspace_defaults,
)
from free_mad_lite.config import (
    load_benchmarks,
    load_experiment_config,
    load_protocol_config,
    phase_metadata,
    resolve_model,
)
from free_mad_lite.prompting import anti_conformity_prompt_hash
from free_mad_lite.reporting import render_report, summarize_run
from free_mad_lite.runner import run_experiment
from free_mad_lite.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="Free-MAD-lite experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show resolved Free-MAD-lite config.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help="Execute one Free-MAD-lite phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root("free_mad_lite"))
    run.add_argument("--cache-root", default=default_cache_root())

    summarize = subparsers.add_parser("summarize-run", help="Print Free-MAD-lite summary.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Validate Free-MAD-lite run.")
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("render-report", help="Regenerate Free-MAD-lite report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root("free_mad_lite"))
    return parser


def main() -> None:
    configure_utf8_stdio()
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        protocol = load_protocol_config(experiment.protocol)
        backbone = resolve_model(args.model or experiment.primary_model_ref)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
            "protocol": asdict(protocol),
            "methods": experiment.methods,
            "global_seed": experiment.global_seed,
            "prompt_version": experiment.prompt_version,
            "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "workspace_defaults": workspace_defaults("free_mad_lite"),
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": {
                "name": backbone.name,
                "provider": backbone.provider,
                "model_id": backbone.model_id,
                "tags": backbone.tags,
            },
            "phases": {
                phase_name: phase_metadata(experiment, phase_name)
                for phase_name in experiment.raw["phases"]
            },
        }
        emit_json(payload)
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
        emit_json(summarize_run(args.run_dir))
        return

    if args.command == "validate-run":
        emit_json(validate_run(args.run_dir))
        return

    if args.command == "render-report":
        emit_json(render_report(args.run_dir, publish_dir=args.publish_dir))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

