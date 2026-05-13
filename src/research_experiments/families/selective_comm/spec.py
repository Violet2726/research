"""CLI entrypoint for selective communication experiments."""

from __future__ import annotations

from dataclasses import asdict
import argparse

from research_experiments.workspace.layout import workspace_defaults
from research_experiments.families.shared.cli import (
    build_standard_family_parser,
    dispatch_standard_family_cli,
)
from research_experiments.families.registry import get_family_spec
from research_experiments.families.selective_comm.config import (
    describe_backbone_fit,
    ensure_backbone_fit,
    load_benchmarks,
    load_control_catalog,
    load_policies,
    load_protocol_config,
    phase_metadata,
)


SPEC = get_family_spec("selective_comm")


def build_parser() -> argparse.ArgumentParser:
    return build_standard_family_parser(
        family_name=SPEC.family_name,
        description="Selective communication trigger experiment runner.",
        inspect_help="Show the resolved selective communication experiment configuration.",
        run_help="Execute one configured selective communication phase.",
        summarize_help="Print a concise run summary from policy_metrics.json.",
        validate_help="Run validation checks for one selective communication run.",
        report_help="Regenerate the Chinese trigger markdown report.",
        include_resume_run_dir=True,
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
    benchmarks = load_benchmarks(experiment)
    protocol = load_protocol_config(experiment.protocol)
    policies = load_policies(experiment.policy_configs)
    controls = load_control_catalog(experiment.control_catalog)
    resolved_model = SPEC.model_resolver(model_override or experiment.primary_model_ref)
    payload = {
        "name": experiment.name,
        "description": experiment.description,
        "benchmark_configs": [str(path) for path in experiment.benchmark_configs],
        "benchmarks": [benchmark.slug for benchmark in benchmarks],
        "protocol": {
            "agent_count": protocol.agent_count,
            "debate_rounds": protocol.debate_rounds,
            "initial_temperature": protocol.initial_temperature,
            "debate_temperature": protocol.debate_temperature,
            "top_p": protocol.top_p,
            "max_output_tokens": protocol.max_output_tokens,
        },
        "policies": [asdict(policy) for policy in policies],
        "controls": {name: asdict(method) for name, method in sorted(controls.items())},
        "prompt_version": experiment.prompt_version,
        "global_seed": experiment.global_seed,
        "max_concurrent_requests": experiment.max_concurrent_requests,
        "requests_per_minute_limit": experiment.requests_per_minute_limit,
        "tokens_per_minute_limit": experiment.tokens_per_minute_limit,
        "workspace_defaults": workspace_defaults(SPEC.family_name),
        "primary_model_ref": experiment.primary_model_ref,
        "resolved_model": asdict(resolved_model),
        "model_fit_warnings": describe_backbone_fit(experiment, resolved_model, benchmarks),
        "phases": experiment.raw["phases"],
        "resolved_by_phase": {
            phase_name: {
                "split_suffix": phase.get("split_suffix"),
                "split_overrides": phase.get("split_overrides"),
            }
            for phase_name, phase in (
                (name, phase_metadata(experiment, name))
                for name in experiment.raw["phases"]
            )
        },
    }
    return payload


def _run_command(args: argparse.Namespace):
    experiment = SPEC.config_loader(args.experiment)
    resolved_model = SPEC.model_resolver(args.model or experiment.primary_model_ref)
    ensure_backbone_fit(experiment, resolved_model)
    return SPEC.runner(
        experiment=experiment,
        phase_name=args.phase,
        backbone=resolved_model,
        run_root=args.runs_root,
        cache_root=args.cache_root,
        resume_run_dir=args.resume_run_dir,
    )


if __name__ == "__main__":
    main()

