"""实验家族注册合同。"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

FamilyPrototype = Literal[
    "independent_sampling",
    "debate_rounds",
    "shared_stage_policy",
    "packet_belief_update",
    "topology_or_graph",
]

ARTIFACT_SCHEMA_VERSION = 1

ArtifactPaths = dict[str, Path | tuple[Path, ...]]


@dataclass(frozen=True)
class FamilyArtifactSchema:
    """声明某个 family 对外暴露的正式运行产物布局。"""

    manifest_path: str = "manifest.json"
    progress_path: str = "progress.json"
    validation_path: str = "run_validation.json"
    report_path: str = "report.md"
    figure_manifest_path: str = "figure_manifest.json"
    archive_manifest_path: str = "archive_manifest.json"
    metrics_view_path: str = "views/metrics.json"
    prediction_records_path: str = "views/predictions.jsonl"
    run_summary_path: str = "views/run_summary.json"
    turn_record_paths: tuple[str, ...] = ()
    diagnostic_paths: tuple[str, ...] = ()
    export_paths: tuple[str, ...] = ()

    def build_paths(self, run_dir: str | Path) -> ArtifactPaths:
        """把相对合同路径映射成某个 run 目录下的绝对路径。"""

        root = Path(run_dir)
        return {
            "manifest_path": root / self.manifest_path,
            "progress_path": root / self.progress_path,
            "validation_path": root / self.validation_path,
            "report_path": root / self.report_path,
            "figure_manifest_path": root / self.figure_manifest_path,
            "archive_manifest_path": root / self.archive_manifest_path,
            "metrics_view_path": root / self.metrics_view_path,
            "prediction_records_path": root / self.prediction_records_path,
            "run_summary_path": root / self.run_summary_path,
            "turn_record_paths": tuple(root / path for path in self.turn_record_paths),
            "diagnostic_paths": tuple(root / path for path in self.diagnostic_paths),
            "export_paths": tuple(root / path for path in self.export_paths),
        }

    def to_manifest_payload(self) -> dict[str, object]:
        """序列化为 manifest 内的稳定 artifact 合同。"""

        return {
            "version": ARTIFACT_SCHEMA_VERSION,
            "manifest_path": self.manifest_path,
            "progress_path": self.progress_path,
            "validation_path": self.validation_path,
            "report_path": self.report_path,
            "figure_manifest_path": self.figure_manifest_path,
            "archive_manifest_path": self.archive_manifest_path,
            "metrics_view_path": self.metrics_view_path,
            "prediction_records_path": self.prediction_records_path,
            "run_summary_path": self.run_summary_path,
            "turn_record_paths": list(self.turn_record_paths),
            "diagnostic_paths": list(self.diagnostic_paths),
            "export_paths": list(self.export_paths),
        }

    @classmethod
    def from_manifest_payload(cls, payload: dict[str, object] | None) -> "FamilyArtifactSchema":
        """从 manifest 载荷恢复 artifact 合同。"""

        source = payload or {}
        return cls(
            manifest_path=str(source.get("manifest_path") or "manifest.json"),
            progress_path=str(source.get("progress_path") or "progress.json"),
            validation_path=str(source.get("validation_path") or "run_validation.json"),
            report_path=str(source.get("report_path") or "report.md"),
            figure_manifest_path=str(source.get("figure_manifest_path") or "figure_manifest.json"),
            archive_manifest_path=str(source.get("archive_manifest_path") or "archive_manifest.json"),
            metrics_view_path=str(source.get("metrics_view_path") or "views/metrics.json"),
            prediction_records_path=str(source.get("prediction_records_path") or "views/predictions.jsonl"),
            run_summary_path=str(source.get("run_summary_path") or "views/run_summary.json"),
            turn_record_paths=tuple(str(item) for item in source.get("turn_record_paths", []) or ()),
            diagnostic_paths=tuple(str(item) for item in source.get("diagnostic_paths", []) or ()),
            export_paths=tuple(str(item) for item in source.get("export_paths", []) or ()),
        )


@dataclass(frozen=True)
class FamilyCliHelp:
    """统一 family CLI 的命令帮助文案。"""

    description: str
    inspect_help: str
    run_help: str
    summarize_help: str
    validate_help: str
    report_help: str
    include_resume_run_dir: bool = False


@dataclass(frozen=True)
class FamilyRunRequest:
    """从根 CLI 传入 family runner 的标准请求。"""

    experiment_path: str
    phase_name: str
    model_ref: str | None = None
    runs_root: str | Path | None = None
    cache_root: str | Path | None = None
    resume_run_dir: str | Path | None = None


@dataclass(frozen=True)
class FamilyRegistration:
    """单个实验家族的统一注册对象。"""

    family_name: str
    prototype: FamilyPrototype
    cli_help: FamilyCliHelp
    artifact_schema: FamilyArtifactSchema
    artifact_aliases: dict[str, str]
    load_experiment: Callable[[str | Path], Any]
    resolve_model: Callable[[str], Any]
    invoke_runner: Callable[..., Path]
    inspect_experiment: Callable[[str, str | None], dict[str, object]]
    run_from_cli: Callable[[FamilyRunRequest], Path]
    summarize_run: Callable[[str | Path], dict[str, Any]]
    validate_run: Callable[..., dict[str, Any]]
    render_report: Callable[[str | Path, str | Path | None], dict[str, Any]]
    configure_parser: Callable[[argparse.ArgumentParser], None] | None = None
    dispatch_extra_command: Callable[[argparse.Namespace], bool] | None = None
    validate_from_cli: Callable[[argparse.Namespace], dict[str, Any]] | None = None
    render_from_cli: Callable[[argparse.Namespace], dict[str, Any]] | None = None

    def build_artifact_paths(self, run_dir: str | Path) -> ArtifactPaths:
        """映射当前 family 的正式产物路径。"""

        return self.artifact_schema.build_paths(run_dir)
