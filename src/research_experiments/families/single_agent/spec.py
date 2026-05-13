"""CLI entrypoint for single-agent experiments."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import argparse

from research_experiments.cli_support.output import emit_json
from research_experiments.core.config import DEFAULT_MODEL_CATALOG_PATH, load_model_catalog
from research_experiments.core.config import load_benchmark_config
from research_experiments.core.data.datasets import generate_split_manifests
from research_experiments.families.shared.method_catalog import load_method_catalog
from research_experiments.workspace.layout import workspace_defaults
from research_experiments.families.shared.cli import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.registry import get_family_spec
from research_experiments.families.single_agent.config import (
    load_benchmarks,
    required_benchmark_tags,
    required_model_tags,
)
from research_experiments.families.single_agent.run.report import export_paper_tables


SPEC = get_family_spec("single_agent")


def build_parser() -> argparse.ArgumentParser:
    parser = build_standard_family_parser(
        family_name=SPEC.family_name,
        description="Single-agent baseline experiment runner.",
        inspect_help="Show the resolved experiment configuration.",
        run_help="Execute one configured experiment phase.",
        summarize_help="Print a concise run summary from metrics.json.",
        validate_help="Run validation checks for one run.",
        report_help="Render the formal single-agent markdown report with figures.",
    )
    subcommands = _subcommands(parser)
    subparsers_action = _subparsers_action(parser)
    subcommands["validate-run"].add_argument("--output-success-threshold", type=float, default=0.95)

    generate = subparsers_action.add_parser("generate-splits", help="Generate frozen benchmark splits.")
    generate.add_argument(
        "--benchmarks",
        nargs="*",
        default=[
            "configs/core/shared/benchmarks/gsm8k.toml",
            "configs/core/shared/benchmarks/strategyqa.toml",
            "configs/core/shared/benchmarks/hotpotqa.toml",
            "configs/core/shared/benchmarks/math500.toml",
            "configs/core/shared/benchmarks/mmlu_pro.toml",
            "configs/core/shared/benchmarks/gpqa_diamond.toml",
            "configs/core/shared/benchmarks/gsm_symbolic.toml",
        ],
    )
    generate.add_argument("--output-dir", default="configs/core/shared/benchmarks/splits")

    list_models = subparsers_action.add_parser("list-models", help="List registered models from the catalog.")
    list_models.add_argument("--catalog", default=str(DEFAULT_MODEL_CATALOG_PATH))

    export = subparsers_action.add_parser("export-paper-tables", help="Export paper-ready markdown tables.")
    export.add_argument("--run-dir", required=True)
    export.add_argument("--output", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    dispatch_standard_family_cli(
        parser=build_parser(),
        inspect_payload_builder=_build_inspect_payload,
        run_command=_run_command,
        summarize_run=SPEC.summarizer,
        validate_run=SPEC.validator,
        render_report=SPEC.report_renderer,
        extra_dispatch=_dispatch_extra_command,
        validate_command=lambda args: SPEC.validator(
            args.run_dir,
            output_success_threshold=args.output_success_threshold,
        ),
        argv=argv,
    )


def _build_inspect_payload(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = SPEC.config_loader(experiment_path)
    methods = load_method_catalog(experiment.method_catalog)
    benchmarks = load_benchmarks(experiment)
    benchmark_slugs = [benchmark.slug for benchmark in benchmarks]
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
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
        "primary_model_ref": experiment.primary_model_ref,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "phases": experiment.raw["phases"],
        "methods": {name: asdict(method) for name, method in sorted(methods.items())},
        "resolved_model": asdict(resolved_model),
        "resolved_requirements_by_phase": {
            phase_name: {
                "required_model_tags": required_model_tags(experiment, phase_name),
                "benchmark_required_tags": {
                    benchmark_slug: required_benchmark_tags(experiment, phase_name, benchmark_slug)
                    for benchmark_slug in benchmark_slugs
                },
            }
            for phase_name in experiment.raw["phases"]
        },
    }


def _run_command(args: argparse.Namespace):
    experiment = SPEC.config_loader(args.experiment)
    resolved_model = SPEC.model_resolver(args.model or experiment.primary_model_ref)
    benchmarks = load_benchmarks(experiment)
    return SPEC.runner(
        experiment=experiment,
        phase_name=args.phase,
        models=[resolved_model],
        benchmarks=benchmarks,
        run_root=args.runs_root,
        cache_root=args.cache_root,
    )


def _dispatch_extra_command(args: argparse.Namespace) -> bool:
    if args.command == "generate-splits":
        benchmarks = [load_benchmark_config(path) for path in args.benchmarks]
        created = generate_split_manifests(benchmarks, args.output_dir)
        for path in created:
            print(path.as_posix())
        return True

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
                            "reasoning_effort": entry.reasoning_effort,
                            "default_max_output_tokens": entry.default_max_output_tokens,
                            "timeout_seconds": entry.timeout_seconds,
                            "max_retries": entry.max_retries,
                        }.items()
                        if value is not None
                    },
                }
            )
        emit_json(payload)
        return True

    if args.command == "export-paper-tables":
        output = args.output or str(Path(args.run_dir) / "paper_tables.md")
        print(export_paper_tables(args.run_dir, output).as_posix())
        return True

    return False


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    return _subparsers_action(parser).choices


def _subparsers_action(parser: argparse.ArgumentParser) -> argparse._SubParsersAction:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action
    raise RuntimeError("Parser is missing subcommands.")


if __name__ == "__main__":
    main()

