"""budget_comm CLI。"""

from __future__ import annotations

import argparse
import json

from budget_comm.config import (
    load_auction_policy_config,
    load_benchmarks,
    load_context_view_config,
    load_experiment_config,
    load_protocol_config,
    phase_metadata,
    resolve_backbone,
)
from budget_comm.reporting import render_report, summarize_run
from budget_comm.runner import run_experiment
from budget_comm.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Budget-aware DALA-lite experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved budget_comm experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--backbone", default=None)

    run = subparsers.add_parser("run", help="Execute one configured budget_comm experiment phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--backbone", default=None)
    run.add_argument("--runs-root", default="local/runs/budget_comm")
    run.add_argument("--cache-path", default="cache/budget_comm_requests.sqlite")

    summarize = subparsers.add_parser("summarize-run", help="Print a concise budget_comm run summary.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Run budget_comm validation checks.")
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("report-run", help="Regenerate the Chinese budget_comm markdown report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default="local/reports/budget_comm")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        protocol = load_protocol_config(experiment.protocol)
        auction_policy = load_auction_policy_config(experiment.auction_policy)
        context_view = load_context_view_config(experiment.context_view)
        resolved_backbone = resolve_backbone(args.backbone or experiment.primary_backbone)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
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
            "auction_policy": {
                "calibration_fraction": auction_policy.calibration_fraction,
                "disagreement_weight": auction_policy.disagreement_weight,
                "evidence_weight": auction_policy.evidence_weight,
                "novelty_weight": auction_policy.novelty_weight,
                "confidence_weight": auction_policy.confidence_weight,
                "positive_density_threshold": auction_policy.positive_density_threshold,
                "full_token_cap": auction_policy.full_token_cap,
                "summary_token_cap": auction_policy.summary_token_cap,
                "keywords_token_cap": auction_policy.keywords_token_cap,
            },
            "context_view": {
                "track_name": context_view.track_name,
                "strategyqa_mode": context_view.strategyqa_mode,
                "hotpotqa_mode": context_view.hotpotqa_mode,
                "allow_full_context": context_view.allow_full_context,
            },
            "global_seed": experiment.global_seed,
            "prompt_version": experiment.prompt_version,
            "calibration_sample_size": experiment.calibration_sample_size,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "primary_backbone": experiment.primary_backbone,
            "resolved_backbone": {
                "name": resolved_backbone.name,
                "provider": resolved_backbone.provider,
                "model_id": resolved_backbone.model_id,
                "tags": resolved_backbone.tags,
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
