"""`dmad` experiment CLI entrypoint."""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.families.dmad.config import (
    load_benchmarks,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
)
from research_experiments.families.registry import get_family_spec
from research_experiments.families.shared.cli import build_standard_family_parser, dispatch_standard_family_cli
from research_experiments.workspace.layout import workspace_defaults


SPEC = get_family_spec("dmad")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="DMAD strategy-diverse debate experiment runner.",
        inspect_help="Show the resolved DMAD experiment configuration.",
        run_help="Execute one configured DMAD experiment phase.",
        summarize_help="Print a concise DMAD run summary.",
        validate_help="Run DMAD validation checks.",
        report_help="Regenerate the Chinese DMAD markdown report.",
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
    benchmarks = load_benchmarks(experiment)
    controls = load_control_catalog(experiment.control_catalog) if experiment.control_catalog is not None else {}
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    return {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": asdict(protocol),
        "control_catalog": None if experiment.control_catalog is None else experiment.control_catalog.as_posix(),
        "controls": {name: asdict(method) for name, method in controls.items()},
        "methods": [
            {
                "name": method.name,
                "mode": method.mode,
                "roster": None if method.roster is None else method.roster.as_posix(),
                "note": method.note,
                "matched_controls": list(method.matched_controls),
                "roster_config": None if method.roster is None else asdict(load_roster_config(method.roster)),
            }
            for method in experiment.methods
        ],
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "phases": experiment.raw["phases"],
    }


def _run_command(args: argparse.Namespace):
    experiment = SPEC.config_loader(args.experiment)
    resolved_model = SPEC.model_resolver(args.model or experiment.primary_model_ref)
    return SPEC.runner(
        experiment=experiment,
        phase_name=args.phase,
        backbone=resolved_model,
        run_root=args.runs_root,
        cache_root=args.cache_root,
    )


if __name__ == "__main__":
    main()
