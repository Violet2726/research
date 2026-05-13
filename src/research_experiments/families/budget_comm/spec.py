"""CLI entrypoint for budget-aware communication experiments."""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.families.budget_comm.config import (
    load_auction_policy_config,
    load_benchmarks,
    load_context_view_config,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.workspace.layout import workspace_defaults
from research_experiments.families.shared.cli import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.registry import get_family_spec


SPEC = get_family_spec("budget_comm")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="Budget-aware DALA-lite experiment runner.",
        inspect_help="Show the resolved budget_comm experiment configuration.",
        run_help="Execute one configured budget_comm experiment phase.",
        summarize_help="Print a concise budget_comm run summary.",
        validate_help="Run budget_comm validation checks.",
        report_help="Regenerate the Chinese budget_comm markdown report.",
    )


def main(argv: list[str] | None = None) -> None:
    dispatch_standard_family_cli(
        parser=build_parser(),
        inspect_payload_builder=_build_inspect_payload,
        run_command=_run_command,
        summarize_run=SPEC.summarizer,
        validate_run=SPEC.validator,
        render_report=SPEC.report_renderer,
        argv=argv,
    )


def _build_inspect_payload(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = SPEC.config_loader(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    auction_policy = load_auction_policy_config(experiment.auction_policy)
    context_view = load_context_view_config(experiment.context_view)
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "auction_policy": asdict(auction_policy),
        "context_view": asdict(context_view),
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "calibration_sample_size": experiment.calibration_sample_size,
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


if __name__ == "__main__":
    main()

