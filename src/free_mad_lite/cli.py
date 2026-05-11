"""CLI entrypoint for Free-MAD-lite experiments."""

from __future__ import annotations

from dataclasses import asdict
import argparse

from experiment_core.foundation.workspace import workspace_defaults
from experiment_core.orchestration.cli_scaffold import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from experiment_core.orchestration.registry import get_family_spec
from free_mad_lite.config import load_benchmarks, load_protocol_config, phase_metadata
from free_mad_lite.prompting import anti_conformity_prompt_hash


SPEC = get_family_spec("free_mad_lite")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="Free-MAD-lite experiment runner.",
        inspect_help="Show resolved Free-MAD-lite config.",
        run_help="Execute one Free-MAD-lite phase.",
        summarize_help="Print Free-MAD-lite summary.",
        validate_help="Validate Free-MAD-lite run.",
        report_help="Regenerate Free-MAD-lite report.",
    )


def main() -> None:
    dispatch_standard_family_cli(
        parser=build_parser(),
        inspect_payload_builder=_build_inspect_payload,
        run_command=_run_command,
        summarize_run=SPEC.summarizer,
        validate_run=SPEC.validator,
        render_report=SPEC.report_renderer,
    )


def _build_inspect_payload(experiment_path: str, model_override: str | None) -> dict[str, object]:
    experiment = SPEC.config_loader(experiment_path)
    protocol = load_protocol_config(experiment.protocol)
    backbone = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "methods": experiment.methods,
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": {
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
