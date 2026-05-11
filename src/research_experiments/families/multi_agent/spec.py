"""CLI entrypoint for multi-agent experiments."""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.core.foundation.workspace import workspace_defaults
from research_experiments.families.cli_scaffold import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.registry import get_family_spec
from research_experiments.families.multi_agent.config import (
    load_benchmarks,
    load_control_catalog,
    load_protocol_config,
    load_roster_config,
    phase_metadata,
)


SPEC = get_family_spec("multi_agent")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="Vanilla MAD multi-agent baseline runner.",
        inspect_help="Show the resolved multi-agent experiment configuration.",
        run_help="Execute one configured multi-agent experiment phase.",
        summarize_help="Print a concise run summary from metrics.json.",
        validate_help="Run validation checks for one multi-agent run.",
        report_help="Generate paired Debate vs Vote analysis and a Chinese markdown report.",
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
    controls = load_control_catalog(experiment.control_catalog)
    benchmarks = load_benchmarks(experiment)
    payload = {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "control_catalog": str(experiment.control_catalog),
        "control_methods": {name: asdict(method) for name, method in sorted(controls.items())},
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "primary_model_ref": experiment.primary_model_ref,
        "phases": experiment.raw["phases"],
        "setups": [
            {
                "name": setup.name,
                "protocol": asdict(load_protocol_config(setup.protocol)),
                "roster": asdict(load_roster_config(setup.roster)),
                "matched_controls": setup.matched_controls,
                "calls_per_question": load_roster_config(setup.roster).agent_count
                * (1 + load_protocol_config(setup.protocol).debate_rounds),
            }
            for setup in experiment.setups
        ],
    }
    payload["resolved_model"] = asdict(SPEC.model_resolver(model_override or experiment.primary_model_ref))
    payload["resolved_by_phase"] = {
        phase_name: {
            "setups": phase["setups"],
            "split_suffix": phase.get("split_suffix"),
            "split_overrides": phase.get("split_overrides"),
        }
        for phase_name, phase in (
            (name, phase_metadata(experiment, name))
            for name in experiment.raw["phases"]
        )
    }
    return payload


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
