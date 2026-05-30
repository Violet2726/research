"""`single_agent` family registration."""

from __future__ import annotations

import argparse
from dataclasses import asdict

from research_experiments.core.contracts import FamilyCliHelp
from research_experiments.families.registration_helpers import (
    build_single_agent_run_from_cli,
    make_family_registration,
)
from research_experiments.families.shared.config_loading import load_benchmarks, resolve_model
from research_experiments.families.shared.method_catalog import load_method_catalog
from research_experiments.families.single_agent.config import (
    load_experiment_config,
    required_benchmark_tags,
    required_model_tags,
)
from research_experiments.families.single_agent.run.execute import run_experiment
from research_experiments.families.single_agent.run.report import render_report, summarize_run
from research_experiments.families.single_agent.run.validate import validate_run
from research_experiments.workspace.layout import workspace_defaults


def inspect_experiment(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = load_experiment_config(experiment_path)
    methods = load_method_catalog(experiment.method_catalog)
    benchmarks = load_benchmarks(experiment)
    benchmark_slugs = [benchmark.slug for benchmark in benchmarks]
    resolved_model = resolve_model(model_override or experiment.primary_model_ref)
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
        "workspace_defaults": workspace_defaults("single_agent"),
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


def validate_from_cli(args) -> dict[str, object]:
    return validate_run(args.run_dir, output_success_threshold=args.output_success_threshold)


def configure_parser(parser) -> None:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            action.choices["validate-run"].add_argument("--output-success-threshold", type=float, default=0.95)
            return
    raise RuntimeError("Parser is missing subcommands.")


REGISTRATION = make_family_registration(
    family_name="single_agent",
    prototype="independent_sampling",
    cli_help=FamilyCliHelp(
        description="Single-agent baseline experiment runner.",
        inspect_help="Show the resolved experiment configuration.",
        run_help="Execute one configured experiment phase.",
        summarize_help="Print a concise run summary from metrics.json.",
        validate_help="Run validation checks for one run.",
        report_help="Render the formal single-agent markdown report with figures.",
    ),
    load_experiment=load_experiment_config,
    resolve_model=resolve_model,
    invoke_runner=run_experiment,
    inspect_experiment=inspect_experiment,
    run_from_cli=build_single_agent_run_from_cli(
        load_experiment=load_experiment_config,
        resolve_model=resolve_model,
        invoke_runner=run_experiment,
    ),
    summarize_run=summarize_run,
    validate_run=validate_run,
    render_report=render_report,
    configure_parser=configure_parser,
    validate_from_cli=validate_from_cli,
    metrics_view_path="metrics.json",
    prediction_records_path="predictions.jsonl",
    turn_record_paths=("raw_responses.jsonl",),
    extra_view_paths=("paper_tables.md", "run_summary.json"),
)
