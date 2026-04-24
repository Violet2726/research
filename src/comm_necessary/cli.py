"""comm_necessary CLI。"""

from __future__ import annotations

import argparse
import json

from comm_necessary.config import (
    load_benchmarks,
    load_experiment_config,
    load_protocol_config,
    phase_metadata,
    resolve_backbone,
)
from comm_necessary.reporting import render_report, summarize_run
from comm_necessary.runner import run_experiment
from comm_necessary.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Communication-necessary HotpotQA experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show resolved comm_necessary experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--backbone", default=None)

    run = subparsers.add_parser("run", help="Execute one configured comm_necessary phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--backbone", default=None)
    run.add_argument("--runs-root", default="local/runs/comm_necessary")
    run.add_argument("--cache-path", default="cache/comm_necessary_requests.sqlite")

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Validate a comm_necessary run directory.")
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("report-run", help="Regenerate the Chinese Markdown report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default="local/reports/comm_necessary")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        protocol = load_protocol_config(experiment.protocol)
        backbone = resolve_backbone(args.backbone or experiment.primary_backbone)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
            "protocol": {
                "agent_count": protocol.agent_count,
                "debate_rounds": protocol.debate_rounds,
                "initial_temperature": protocol.initial_temperature,
                "update_temperature": protocol.update_temperature,
                "top_p": protocol.top_p,
                "max_output_tokens": protocol.max_output_tokens,
                "answer_only_token_cap": protocol.answer_only_token_cap,
                "evidence_token_cap": protocol.evidence_token_cap,
                "full_packet_token_cap": protocol.full_packet_token_cap,
            },
            "methods": experiment.methods,
            "global_seed": experiment.global_seed,
            "prompt_version": experiment.prompt_version,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "primary_backbone": experiment.primary_backbone,
            "resolved_backbone": {
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
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        backbone = resolve_backbone(args.backbone or experiment.primary_backbone)
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            backbone=backbone,
            run_root=args.runs_root,
            cache_path=args.cache_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-run":
        print(json.dumps(validate_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "report-run":
        print(json.dumps(render_report(args.run_dir, publish_dir=args.publish_dir), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


if __name__ == "__main__":
    main()

