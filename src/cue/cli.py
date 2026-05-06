"""CUE 实验的命令行入口。"""

from __future__ import annotations

import argparse
import json

from dotenv import load_dotenv

from cue.config import (
    load_benchmarks,
    load_control_catalog,
    load_experiment_config,
    load_policies,
    load_protocol_config,
    resolve_model,
)
from cue.reporting import render_cue_report, summarize_run
from cue.runner import run_experiment
from cue.validation import validate_run
from experiment_core.workspace import default_cache_path, default_reports_root, default_runs_root, workspace_defaults


def build_parser() -> argparse.ArgumentParser:
    """构建 CUE 子命令解析器。"""
    load_dotenv(".env.local", override=False)
    parser = argparse.ArgumentParser(description="CUE experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved CUE experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help="Execute one configured CUE phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", default=None)
    run.add_argument("--runs-root", default=default_runs_root("cue"))
    run.add_argument("--cache-path", default=default_cache_path("cue"))

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary from policy_metrics.json.")
    summarize.add_argument("--run-dir", required=True)

    validate = subparsers.add_parser("validate-run", help="Run validation checks for one CUE run.")
    validate.add_argument("--run-dir", required=True)

    report = subparsers.add_parser("report-cue", help="Regenerate the Chinese CUE markdown report.")
    report.add_argument("--run-dir", required=True)
    report.add_argument("--publish-dir", default=default_reports_root("cue"))
    return parser


def main() -> None:
    """命令行入口。"""
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "inspect-experiment":
        experiment = load_experiment_config(args.experiment)
        benchmarks = load_benchmarks(experiment)
        protocol = load_protocol_config(experiment.protocol)
        policies = load_policies(experiment.policy_configs)
        controls = load_control_catalog(experiment.control_catalog)
        resolved_model = resolve_model(args.model or experiment.primary_model_ref)
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "benchmarks": [benchmark.slug for benchmark in benchmarks],
            "protocol": asdict_like(protocol),
            "policies": [asdict_like(policy) for policy in policies],
            "controls": {name: asdict_like(method) for name, method in sorted(controls.items())},
            "prompt_version": experiment.prompt_version,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "workspace_defaults": workspace_defaults("cue"),
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": asdict_like(resolved_model),
            "phases": experiment.raw["phases"],
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        model_ref = args.model or experiment.primary_model_ref
        resolved_model = resolve_model(model_ref)
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            backbone=resolved_model,
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

    if args.command == "report-cue":
        print(json.dumps(render_cue_report(args.run_dir, publish_dir=args.publish_dir), ensure_ascii=False, indent=2))
        return

    parser.error(f"Unsupported command: {args.command}")


def asdict_like(obj: object) -> dict[str, object]:
    """把 dataclass 或普通对象转换成可 JSON 序列化的字典视图。"""
    return dict(vars(obj))


if __name__ == "__main__":
    main()
