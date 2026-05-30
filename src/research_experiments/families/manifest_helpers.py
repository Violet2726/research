"""family manifest 构造辅助函数。"""

from __future__ import annotations

from research_experiments.core.contracts import FamilyArtifactContract, FamilyManifest, FamilyPrototype


def make_family_manifest(
    *,
    family_name: str,
    prototype: FamilyPrototype,
    config_loader_path: str,
    model_resolver_path: str,
    runner_path: str,
    validator_path: str,
    summarizer_path: str,
    report_renderer_path: str,
    cli_main_path: str,
    metrics_view_path: str,
    prediction_records_path: str,
    turn_record_paths: tuple[str, ...] = (),
    extra_view_paths: tuple[str, ...] = (),
    historical_labels: tuple[str, ...] = (),
) -> FamilyManifest:
    """按统一默认值构造一个 family manifest。"""

    return FamilyManifest(
        family_name=family_name,
        prototype=prototype,
        historical_labels=historical_labels,
        config_loader_path=config_loader_path,
        model_resolver_path=model_resolver_path,
        runner_path=runner_path,
        validator_path=validator_path,
        summarizer_path=summarizer_path,
        report_renderer_path=report_renderer_path,
        cli_main_path=cli_main_path,
        artifact_contract=FamilyArtifactContract(
            metrics_view_path=metrics_view_path,
            prediction_records_path=prediction_records_path,
            turn_record_paths=turn_record_paths,
            extra_view_paths=extra_view_paths,
        ),
    )
