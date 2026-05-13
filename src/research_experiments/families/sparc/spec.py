"""`sparc` 实验的 CLI 入口。"""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.workspace.layout import workspace_defaults
from research_experiments.families.shared.cli import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.registry import get_family_spec
from research_experiments.families.sparc.config import load_benchmarks, load_protocol_config, phase_metadata


SPEC = get_family_spec("sparc")


def build_parser() -> argparse.ArgumentParser:
    parser = build_standard_family_parser(
        family_name=SPEC.family_name,
        description="SPARC v1 experiment runner.",
        inspect_help="Show the resolved SPARC experiment configuration.",
        run_help="Execute one configured SPARC experiment phase.",
        summarize_help="Print a concise SPARC run summary.",
        validate_help="Run SPARC validation checks.",
        report_help="Regenerate the Chinese SPARC markdown report.",
    )
    _subcommands(parser)["validate-run"].add_argument("--compare-run-dir", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    dispatch_standard_family_cli(
        parser=build_parser(),
        inspect_payload_builder=_build_inspect_payload,
        run_command=_run_command,
        summarize_run=SPEC.summarizer,
        validate_run=SPEC.validator,
        render_report=SPEC.report_renderer,
        validate_command=lambda args: SPEC.validator(args.run_dir, compare_run_dir=args.compare_run_dir),
        argv=argv,
    )


def _build_inspect_payload(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = SPEC.config_loader(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "variant_name": experiment.variant_name,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
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
        "trigger_reference": asdict(experiment.trigger_reference) if experiment.trigger_reference is not None else None,
        "phases": {
            phase_name: phase_metadata(experiment, phase_name)
            for phase_name in experiment.raw["phases"]
        },
    }


def _run_command(args: argparse.Namespace):
    experiment = SPEC.config_loader(args.experiment)
    backbone = SPEC.model_resolver(args.model or experiment.primary_model_ref)
    return SPEC.runner(
        experiment=experiment,
        phase_name=args.phase,
        backbone=backbone,
        run_root=args.runs_root,
        cache_root=args.cache_root,
    )


def _subcommands(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return action.choices
    raise RuntimeError("Parser is missing subcommands.")


if __name__ == "__main__":
    main()

