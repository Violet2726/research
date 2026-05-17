"""`colmad` 实验的 CLI 入口。"""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.workspace.layout import workspace_defaults
from research_experiments.families.shared.cli import build_standard_family_parser, dispatch_standard_family_cli
from research_experiments.families.registry import get_family_spec
from research_experiments.families.colmad.config import load_benchmarks, load_protocol_config, phase_metadata


SPEC = get_family_spec("colmad")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="ColMAD collaborative oversight reproduction runner.",
        inspect_help="Show the resolved ColMAD experiment configuration.",
        run_help="Execute one configured ColMAD experiment phase.",
        summarize_help="Print a concise ColMAD run summary.",
        validate_help="Run ColMAD validation checks.",
        report_help="Regenerate the Chinese ColMAD markdown report.",
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
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "experiment_kind": experiment.experiment_kind,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in load_benchmarks(experiment)],
        "protocol": asdict(protocol),
        "methods": [asdict(method) for method in experiment.methods],
        "global_seed": experiment.global_seed,
        "prompt_version": experiment.prompt_version,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
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
