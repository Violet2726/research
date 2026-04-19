"""单智能体实验 CLI。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiment_core.config import (
    DEFAULT_MODEL_CATALOG_PATH,
    load_benchmark_config,
    load_model_catalog,
    resolve_model_ref,
)
from experiment_core.datasets import generate_split_manifests
from experiment_core.methods import load_method_catalog
from single_agent_baselines.config import (
    load_experiment_config,
    required_benchmark_tags,
    required_model_tags,
)
from single_agent_baselines.reporting import budget_fairness_check, export_paper_tables, summarize_run
from single_agent_baselines.runner import run_experiment
from single_agent_baselines.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    """构建单智能体实验命令行解析器。"""
    parser = argparse.ArgumentParser(description="Single-agent baseline experiment runner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-splits", help="Generate frozen benchmark splits.")
    generate.add_argument(
        "--benchmarks",
        nargs="*",
        default=[
            "configs/shared/benchmarks/gsm8k.toml",
            "configs/shared/benchmarks/strategyqa.toml",
            "configs/shared/benchmarks/hotpotqa.toml",
        ],
    )
    generate.add_argument("--output-dir", default="configs/shared/benchmarks/splits")

    inspect = subparsers.add_parser("inspect-experiment", help="Show the resolved experiment configuration.")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument("--model", default=None)

    run = subparsers.add_parser("run", help="Execute one configured experiment phase.")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument("--model", required=True)
    run.add_argument("--runs-root", default="runs/single_agent")
    run.add_argument("--cache-path", default="cache/single_agent_requests.sqlite")

    list_models = subparsers.add_parser("list-models", help="List registered models from the catalog.")
    list_models.add_argument("--catalog", default=str(DEFAULT_MODEL_CATALOG_PATH))

    summarize = subparsers.add_parser("summarize-run", help="Print a concise run summary from metrics.json.")
    summarize.add_argument("--run-dir", required=True)

    export = subparsers.add_parser("export-paper-tables", help="Export paper-ready markdown tables.")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output", default=None)

    fairness = subparsers.add_parser("check-budget-fairness", help="Check SC vs MV token fairness.")
    fairness.add_argument("--run-dir", required=True)
    fairness.add_argument("--threshold", type=float, default=0.10)

    validate = subparsers.add_parser("validate-run", help="Run validation checks for one run.")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--parse-success-threshold", type=float, default=0.95)
    validate.add_argument("--budget-threshold", type=float, default=0.10)

    return parser


def main() -> None:
    """解析参数并分发到对应子命令。"""
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
        methods = load_method_catalog(experiment.method_catalog)
        benchmark_slugs = [load_benchmark_config(path).slug for path in experiment.benchmark_configs]
        payload = {
            "name": experiment.name,
            "description": experiment.description,
            "method_catalog": str(experiment.method_catalog),
            "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
            "required_model_tags": experiment.required_model_tags,
            "benchmark_required_tags": experiment.benchmark_required_tags,
            "global_seed": experiment.global_seed,
            "reruns_per_method": experiment.reruns_per_method,
            "prompt_version": experiment.prompt_version,
            "max_concurrent_requests": experiment.max_concurrent_requests,
            "requests_per_minute_limit": experiment.requests_per_minute_limit,
            "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
            "phases": experiment.raw["phases"],
            "methods": {name: _serialize_method(method) for name, method in sorted(methods.items())},
        }
        if args.model:
            resolved_model = resolve_model_ref(args.model)
            payload["resolved_model"] = _serialize_model(resolved_model)
            payload["resolved_requirements_by_phase"] = {
                phase_name: {
                    "required_model_tags": required_model_tags(experiment, phase_name),
                    "benchmark_required_tags": {
                        benchmark_slug: required_benchmark_tags(experiment, phase_name, benchmark_slug)
                        for benchmark_slug in benchmark_slugs
                    },
                }
                for phase_name in experiment.raw["phases"]
            }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    if args.command == "run":
        experiment = load_experiment_config(args.experiment)
        resolved_model = resolve_model_ref(args.model)
        benchmarks = [load_benchmark_config(path) for path in experiment.benchmark_configs]
        run_dir = run_experiment(
            experiment=experiment,
            phase_name=args.phase,
            models=[resolved_model],
            benchmarks=benchmarks,
            run_root=args.runs_root,
            cache_path=args.cache_path,
        )
        print(run_dir.as_posix())
        return

    if args.command == "list-models":
        catalog = load_model_catalog(args.catalog)
        payload = []
        for model_ref, entry in sorted(catalog.items()):
            provider_name, _ = model_ref.split("/", 1)
            payload.append(
                {
                    "model_ref": model_ref,
                    "provider": provider_name,
                    "tags": entry.tags,
                    "overrides": {
                        key: value
                        for key, value in {
                            "supports_response_format": entry.supports_response_format,
                            "response_format": entry.response_format,
                            "default_max_output_tokens": entry.default_max_output_tokens,
                            "timeout_seconds": entry.timeout_seconds,
                            "max_retries": entry.max_retries,
                        }.items()
                        if value is not None
                    },
                }
            )
        print(json.dumps(payload, ensure_ascii=False, indent=2))
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


def _serialize_model(model) -> dict[str, object]:
    """把解析后的模型配置转成可 JSON 序列化结构。"""
    return {
        "name": model.name,
        "provider": model.provider,
        "model_id": model.model_id,
        "base_url": model.base_url,
        "api_key_env": model.api_key_env,
        "chat_path": model.chat_path,
        "default_temperature": model.default_temperature,
        "default_top_p": model.default_top_p,
        "default_max_output_tokens": model.default_max_output_tokens,
        "supports_response_format": model.supports_response_format,
        "response_format": model.response_format,
        "timeout_seconds": model.timeout_seconds,
        "max_retries": model.max_retries,
        "tags": model.tags,
    }


def _serialize_method(method) -> dict[str, object]:
    """把方法配置转成可 JSON 序列化结构。"""
    return {
        "family": method.family,
        "budget_calls": method.budget_calls,
        "temperature": method.temperature,
        "top_p": method.top_p,
        "max_output_tokens": method.max_output_tokens,
    }


if __name__ == "__main__":
    main()
