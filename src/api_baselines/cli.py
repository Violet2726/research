"""命令行入口。

CLI 负责把用户的显式输入转换成配置解析与实验运行调用，
是项目配置链最靠前的一层。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from api_baselines.config import (
    DEFAULT_MODEL_CATALOG_PATH,
    load_benchmark_config,
    load_experiment_config,
    load_method_catalog,
    load_model_catalog,
    required_benchmark_tags,
    required_model_tags,
    resolve_model_ref,
)
from api_baselines.reporting import budget_fairness_check, export_paper_tables, summarize_run
from api_baselines.runner import generate_split_manifests, run_experiment
from api_baselines.validation import validate_run


def build_parser() -> argparse.ArgumentParser:
    """构建 baseline-cli 的命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="API 基线实验运行器。")
    subparsers = parser.add_subparsers(dest="command", required=True)

    generate = subparsers.add_parser("generate-splits", help="生成冻结后的 benchmark split 清单。")
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

    inspect = subparsers.add_parser("inspect-experiment", help="查看实验规格，必要时同时解析一个显式模型。")
    inspect.add_argument("--experiment", required=True)
    inspect.add_argument(
        "--model",
        default=None,
        help="显式解析一个模型引用，例如 'dashscope/qwen2.5-7b-instruct'。",
    )

    run = subparsers.add_parser("run", help="执行一个实验 phase。")
    run.add_argument("--experiment", required=True)
    run.add_argument("--phase", required=True)
    run.add_argument(
        "--model",
        required=True,
        help="显式指定要运行的模型引用，例如 'dashscope/qwen2.5-7b-instruct'。",
    )
    run.add_argument("--runs-root", default="runs")
    run.add_argument("--cache-path", default="cache/requests.sqlite")

    list_models = subparsers.add_parser("list-models", help="列出模型目录中已登记的常用模型。")
    list_models.add_argument("--catalog", default=str(DEFAULT_MODEL_CATALOG_PATH))

    summarize = subparsers.add_parser("summarize-run", help="根据 metrics.json 输出运行摘要。")
    summarize.add_argument("--run-dir", required=True)

    export = subparsers.add_parser("export-paper-tables", help="导出论文草稿可用的 Markdown 表格。")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output", default=None)

    fairness = subparsers.add_parser("check-budget-fairness", help="检查 SC 与 MV 在同预算下的 token 公平性。")
    fairness.add_argument("--run-dir", required=True)
    fairness.add_argument("--threshold", type=float, default=0.10)

    validate = subparsers.add_parser("validate-run", help="对单次运行结果执行严格校验。")
    validate.add_argument("--run-dir", required=True)
    validate.add_argument("--parse-success-threshold", type=float, default=0.95)
    validate.add_argument("--budget-threshold", type=float, default=0.10)

    return parser


def main() -> None:
    """根据子命令分发到对应的实验或报告逻辑。"""
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
    """把解析后的模型对象转成可打印的 JSON 结构。"""
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
    """把方法配置转成稳定的 JSON 输出。"""
    return {
        "family": method.family,
        "budget_calls": method.budget_calls,
        "temperature": method.temperature,
        "top_p": method.top_p,
        "max_output_tokens": method.max_output_tokens,
    }
