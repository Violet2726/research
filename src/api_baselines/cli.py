from __future__ import annotations

import argparse
import json
from pathlib import Path

from api_baselines.config import load_benchmark_config, load_experiment_config, load_model_config
from api_baselines.reporting import budget_fairness_check, export_paper_tables, summarize_run
from api_baselines.runner import generate_split_manifests, run_experiment
from api_baselines.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Professional API baseline experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-splits", help="Generate frozen benchmark split manifests.")
    generate.add_argument(
        "--benchmarks",
        nargs="*",
        default=[
            "configs/benchmarks/gsm8k.toml",
            "configs/benchmarks/strategyqa.toml",
            "configs/benchmarks/hotpotqa.toml",
        ],
    )
    generate.add_argument("--output-dir", default="configs/benchmarks/splits")

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved experiment configuration.")
    inspect.add_argument("--experiment", required=True)

    run = subparsers.add_parser("run", help="Execute one configured experiment phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--runs-root", default="runs")
    run.add_argument("--cache-path", default="cache/requests.sqlite")

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary from metrics.json.")
    summarize.add_argument("--run-dir", required=True)

    export = subparsers.add_parser("export-paper-tables", help="Export markdown tables for paper drafting.")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output", default=None)

    fairness = subparsers.add_parser("check-budget-fairness", help="Check SC vs MV token parity by budget.")
    fairness.add_argument("--run-dir", required=True)
    fairness.add_argument("--threshold", type=float, default=0.10)

    validate = subparsers.add_parser("validate-run", help="Run strict post-hoc validation checks for one run.")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--parse-success-threshold", type=float, default=0.95)
    validate.add_argument("--budget-threshold", type=float, default=0.10)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-splits":
        benchmarks = [load_benchmark_config(path) for path in args.benchmarks]
        created = generate_split_manifests(benchmarks, args.output_dir)
        for path in created:
            print(path.as_posix())
        return

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "model_configs": [str(path) for path in experiment.model_configs],
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "global_seed": experiment.global_seed,
            "reruns_per_method": experiment.reruns_per_method,
            "prompt_version": experiment.prompt_version,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "phases": experiment.raw["phases"],
            "methods": experiment.raw["methods"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        models = [load_model_config(path) for path in experiment.model_configs]
        benchmarks = [load_benchmark_config(path) for path in experiment.benchmark_configs]
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            models=models,
            benchmarks=benchmarks,
            run_root=args.runs_root,
            cache_path=args.cache_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "summarize-run":
        print(json.dumps(summarize_run(args.run_dir), ensure_ascii=False, indent=2))
        return

    if args.command == "export-paper-tables":
        output = args.output or str(Path(args.run_dir) / "paper_tables.md")
        path = export_paper_tables(args.run_dir, output)
        print(path.as_posix())
        return

    if args.command == "check-budget-fairness":
        print(json.dumps(budget_fairness_check(args.run_dir, threshold=args.threshold), ensure_ascii=False, indent=2))
        return

    if args.command == "validate-run":
        print(
            json.dumps(
                validate_run(
                    args.run_dir,
                    parse_success_threshold=args.parse_success_threshold,
                    budget_threshold=args.budget_threshold,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    parser.error(f"Unsupported command: {args.command}")
