"""Single source of truth for experiment family registration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from budget_comm.config import load_experiment_config as load_budget_experiment_config
from budget_comm.config import resolve_model as resolve_budget_model
from budget_comm.reporting import render_report as render_budget_report
from budget_comm.reporting import summarize_run as summarize_budget_run
from budget_comm.runner import run_experiment as run_budget_experiment
from budget_comm.validation import validate_run as validate_budget_run
from comm_necessary.config import load_experiment_config as load_comm_necessary_experiment_config
from comm_necessary.config import resolve_model as resolve_comm_necessary_model
from comm_necessary.reporting import render_report as render_comm_necessary_report
from comm_necessary.reporting import summarize_run as summarize_comm_necessary_run
from comm_necessary.runner import run_experiment as run_comm_necessary_experiment
from comm_necessary.validation import validate_run as validate_comm_necessary_run
from cue.config import load_experiment_config as load_cue_experiment_config
from cue.config import resolve_model as resolve_cue_model
from cue.reporting import render_report as render_cue_report
from cue.reporting import summarize_run as summarize_cue_run
from cue.runner import run_experiment as run_cue_experiment
from cue.validation import validate_run as validate_cue_run
from free_mad_lite.config import load_experiment_config as load_free_mad_experiment_config
from free_mad_lite.config import resolve_model as resolve_free_mad_model
from free_mad_lite.reporting import render_report as render_free_mad_report
from free_mad_lite.reporting import summarize_run as summarize_free_mad_run
from free_mad_lite.runner import run_experiment as run_free_mad_experiment
from free_mad_lite.validation import validate_run as validate_free_mad_run
from multi_agent.config import load_experiment_config as load_multi_agent_experiment_config
from multi_agent.config import resolve_model as resolve_multi_agent_model
from multi_agent.reporting import render_report as render_multi_agent_report
from multi_agent.reporting import summarize_run as summarize_multi_agent_run
from multi_agent.runner import run_experiment as run_multi_agent_experiment
from multi_agent.validation import validate_run as validate_multi_agent_run
from selective_comm.config import load_experiment_config as load_selective_experiment_config
from selective_comm.config import resolve_model as resolve_selective_model
from selective_comm.reporting import render_report as render_selective_report
from selective_comm.reporting import summarize_run as summarize_selective_run
from selective_comm.runner import run_experiment as run_selective_experiment
from selective_comm.validation import validate_run as validate_selective_run
from sid_lite.config import load_experiment_config as load_sid_experiment_config
from sid_lite.config import resolve_model as resolve_sid_model
from sid_lite.reporting import render_report as render_sid_report
from sid_lite.reporting import summarize_run as summarize_sid_run
from sid_lite.runner import run_experiment as run_sid_experiment
from sid_lite.validation import validate_run as validate_sid_run
from single_agent.config import load_experiment_config as load_single_agent_experiment_config
from single_agent.config import resolve_model as resolve_single_agent_model
from single_agent.reporting import render_report as render_single_agent_report
from single_agent.reporting import summarize_run as summarize_single_agent_run
from single_agent.runner import run_experiment as run_single_agent_experiment
from single_agent.validation import validate_run as validate_single_agent_run
from sparc.config import load_experiment_config as load_sparc_experiment_config
from sparc.config import resolve_model as resolve_sparc_model
from sparc.reporting import render_report as render_sparc_report
from sparc.reporting import summarize_run as summarize_sparc_run
from sparc.runner import run_experiment as run_sparc_experiment
from sparc.validation import validate_run as validate_sparc_run


RunnerFn = Callable[..., Path]
ValidatorFn = Callable[[str | Path], dict[str, Any]]
SummarizerFn = Callable[[str | Path], dict[str, Any]]
ReportRendererFn = Callable[[str | Path, str | Path | None], dict[str, Any]]
ConfigLoaderFn = Callable[[str | Path], Any]
ModelResolverFn = Callable[[str], Any]


@dataclass(frozen=True)
class FamilySpec:
    """Registered entry for one experiment family."""

    family_name: str
    cli_script: str
    config_loader: ConfigLoaderFn
    model_resolver: ModelResolverFn
    runner: RunnerFn
    validator: ValidatorFn
    summarizer: SummarizerFn
    report_renderer: ReportRendererFn
    standard_cli: bool = True


FAMILY_SPECS: dict[str, FamilySpec] = {
    "budget_comm": FamilySpec(
        family_name="budget_comm",
        cli_script="budget_comm_cli",
        config_loader=load_budget_experiment_config,
        model_resolver=resolve_budget_model,
        runner=run_budget_experiment,
        validator=validate_budget_run,
        summarizer=summarize_budget_run,
        report_renderer=render_budget_report,
    ),
    "comm_necessary": FamilySpec(
        family_name="comm_necessary",
        cli_script="comm_necessary_cli",
        config_loader=load_comm_necessary_experiment_config,
        model_resolver=resolve_comm_necessary_model,
        runner=run_comm_necessary_experiment,
        validator=validate_comm_necessary_run,
        summarizer=summarize_comm_necessary_run,
        report_renderer=render_comm_necessary_report,
    ),
    "cue": FamilySpec(
        family_name="cue",
        cli_script="cue_cli",
        config_loader=load_cue_experiment_config,
        model_resolver=resolve_cue_model,
        runner=run_cue_experiment,
        validator=validate_cue_run,
        summarizer=summarize_cue_run,
        report_renderer=render_cue_report,
    ),
    "free_mad_lite": FamilySpec(
        family_name="free_mad_lite",
        cli_script="free_mad_lite_cli",
        config_loader=load_free_mad_experiment_config,
        model_resolver=resolve_free_mad_model,
        runner=run_free_mad_experiment,
        validator=validate_free_mad_run,
        summarizer=summarize_free_mad_run,
        report_renderer=render_free_mad_report,
    ),
    "multi_agent": FamilySpec(
        family_name="multi_agent",
        cli_script="multi_agent_cli",
        config_loader=load_multi_agent_experiment_config,
        model_resolver=resolve_multi_agent_model,
        runner=run_multi_agent_experiment,
        validator=validate_multi_agent_run,
        summarizer=summarize_multi_agent_run,
        report_renderer=render_multi_agent_report,
    ),
    "selective_comm": FamilySpec(
        family_name="selective_comm",
        cli_script="selective_comm_cli",
        config_loader=load_selective_experiment_config,
        model_resolver=resolve_selective_model,
        runner=run_selective_experiment,
        validator=validate_selective_run,
        summarizer=summarize_selective_run,
        report_renderer=render_selective_report,
    ),
    "sid_lite": FamilySpec(
        family_name="sid_lite",
        cli_script="sid_lite_cli",
        config_loader=load_sid_experiment_config,
        model_resolver=resolve_sid_model,
        runner=run_sid_experiment,
        validator=validate_sid_run,
        summarizer=summarize_sid_run,
        report_renderer=render_sid_report,
    ),
    "single_agent": FamilySpec(
        family_name="single_agent",
        cli_script="single_agent_cli",
        config_loader=load_single_agent_experiment_config,
        model_resolver=resolve_single_agent_model,
        runner=run_single_agent_experiment,
        validator=validate_single_agent_run,
        summarizer=summarize_single_agent_run,
        report_renderer=render_single_agent_report,
    ),
    "sparc": FamilySpec(
        family_name="sparc",
        cli_script="sparc_cli",
        config_loader=load_sparc_experiment_config,
        model_resolver=resolve_sparc_model,
        runner=run_sparc_experiment,
        validator=validate_sparc_run,
        summarizer=summarize_sparc_run,
        report_renderer=render_sparc_report,
    ),
}


def get_family_spec(family_name: str) -> FamilySpec:
    """Return one registered family spec."""

    return FAMILY_SPECS[family_name]


def registered_family_names() -> tuple[str, ...]:
    """Return registered family names in stable order."""

    return tuple(sorted(FAMILY_SPECS))


def standard_cli_family_names() -> tuple[str, ...]:
    """Return families that expose the standard family CLI surface."""

    return tuple(
        spec.family_name
        for spec in sorted(FAMILY_SPECS.values(), key=lambda item: item.family_name)
        if spec.standard_cli
    )


def validator_map() -> dict[str, ValidatorFn]:
    """Return the registered validator map keyed by family name."""

    return {
        family_name: spec.validator
        for family_name, spec in FAMILY_SPECS.items()
    }
