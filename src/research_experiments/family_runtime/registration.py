"""family 注册对象的共享构造辅助。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.core.contracts import (
    FamilyArtifactSchema,
    FamilyCliHelp,
    FamilyPrototype,
    FamilyRegistration,
    FamilyRunRequest,
)
from research_experiments.family_runtime.config_helpers import load_benchmarks

_RESERVED_LAYOUT_ATTRIBUTE_NAMES = {
    "root",
    "schema",
    "manifest",
    "progress",
    "validation",
    "report",
    "figure_manifest",
    "archive_manifest",
    "metrics",
    "predictions",
    "run_summary",
    "turns",
    "diagnostics",
    "exports",
    "aliases",
    "turn_path",
    "diagnostic_path",
    "export_path",
}


def build_backbone_run_from_cli(
    *,
    load_experiment,
    resolve_model,
    invoke_runner,
) -> Any:
    """构造以单 backbone 运行的 family CLI 执行函数。"""

    def _run(request: FamilyRunRequest) -> Path:
        experiment = load_experiment(request.experiment_path)
        resolved_model = resolve_model(request.model_ref or experiment.primary_model_ref)
        kwargs = {
            "experiment": experiment,
            "phase_name": request.phase_name,
            "backbone": resolved_model,
            "run_root": request.runs_root,
            "cache_root": request.cache_root,
        }
        if request.resume_run_dir is not None:
            kwargs["resume_run_dir"] = request.resume_run_dir
        return invoke_runner(**kwargs)

    return _run


def build_single_agent_run_from_cli(
    *,
    load_experiment,
    resolve_model,
    invoke_runner,
) -> Any:
    """构造 single_agent 风格的 CLI 执行函数。"""

    def _run(request: FamilyRunRequest) -> Path:
        experiment = load_experiment(request.experiment_path)
        resolved_model = resolve_model(request.model_ref or experiment.primary_model_ref)
        return invoke_runner(
            experiment=experiment,
            phase_name=request.phase_name,
            models=[resolved_model],
            benchmarks=load_benchmarks(experiment),
            run_root=request.runs_root,
            cache_root=request.cache_root,
        )

    return _run


def make_family_registration(
    *,
    family_name: str,
    prototype: FamilyPrototype,
    cli_help: FamilyCliHelp,
    load_experiment,
    resolve_model,
    invoke_runner,
    inspect_experiment,
    run_from_cli,
    summarize_run,
    validate_run,
    render_report,
    artifact_aliases: dict[str, str],
    metrics_view_path: str,
    prediction_records_path: str,
    turn_record_paths: tuple[str, ...] = (),
    diagnostic_paths: tuple[str, ...] = (),
    export_paths: tuple[str, ...] = (),
    configure_parser=None,
    dispatch_extra_command=None,
    validate_from_cli=None,
    render_from_cli=None,
) -> FamilyRegistration:
    """按统一默认值构造一个 family registration。"""

    reserved_aliases = sorted(set(artifact_aliases) & _RESERVED_LAYOUT_ATTRIBUTE_NAMES)
    if reserved_aliases:
        raise ValueError(
            f"{family_name} artifact aliases shadow FamilyRunLayout attributes: {reserved_aliases}"
        )

    return FamilyRegistration(
        family_name=family_name,
        prototype=prototype,
        cli_help=cli_help,
        artifact_aliases=artifact_aliases,
        load_experiment=load_experiment,
        resolve_model=resolve_model,
        invoke_runner=invoke_runner,
        inspect_experiment=inspect_experiment,
        run_from_cli=run_from_cli,
        summarize_run=summarize_run,
        validate_run=validate_run,
        render_report=render_report,
        configure_parser=configure_parser,
        dispatch_extra_command=dispatch_extra_command,
        validate_from_cli=validate_from_cli,
        render_from_cli=render_from_cli,
        artifact_schema=FamilyArtifactSchema(
            metrics_view_path=metrics_view_path,
            prediction_records_path=prediction_records_path,
            turn_record_paths=turn_record_paths,
            diagnostic_paths=diagnostic_paths,
            export_paths=export_paths,
        ),
    )

