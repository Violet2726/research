"""family manifest 驱动的运行产物索引读取接口。"""

from __future__ import annotations

from pathlib import Path
from typing import cast

from research_experiments.core.contracts import (
    FamilyArtifactSchema,
    PredictionRecord,
    RunArtifactIndex,
)
from research_experiments.core.io import read_json


def resolve_run_artifact_index(
    run_dir: str | Path,
    *,
    family_name: str | None = None,
) -> RunArtifactIndex:
    """根据 run manifest 解析正式产物索引。"""

    root = Path(run_dir)
    manifest = load_run_manifest(root)
    resolved_family = family_name or _resolve_family_name(root, manifest)
    schema = FamilyArtifactSchema.from_manifest_payload(_manifest_artifact_schema(manifest))
    paths = schema.build_paths(root)
    return RunArtifactIndex(
        family_name=resolved_family,
        prototype=str(manifest.get("prototype") or ""),
        run_dir=root,
        manifest_path=cast(Path, paths["manifest_path"]),
        progress_path=cast(Path, paths["progress_path"]),
        validation_path=cast(Path, paths["validation_path"]),
        report_path=cast(Path, paths["report_path"]),
        figure_manifest_path=cast(Path, paths["figure_manifest_path"]),
        archive_manifest_path=cast(Path, paths["archive_manifest_path"]),
        metrics_view_path=cast(Path, paths["metrics_view_path"]),
        prediction_records_path=cast(Path, paths["prediction_records_path"]),
        run_summary_path=cast(Path, paths["run_summary_path"]),
        turn_record_paths=cast(tuple[Path, ...], paths["turn_record_paths"]),
        diagnostic_paths=cast(tuple[Path, ...], paths["diagnostic_paths"]),
        export_paths=cast(tuple[Path, ...], paths["export_paths"]),
    )


def load_run_manifest(run_dir: str | Path) -> dict[str, object]:
    """读取某个 run 的 manifest.json。"""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return read_json(manifest_path)


def load_metrics_payload(run_dir: str | Path, *, family_name: str | None = None) -> dict[str, object]:
    """按 family manifest 读取规范指标视图。"""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    if not index.metrics_view_path.exists():
        return {}
    return read_json(index.metrics_view_path)


def load_prediction_records(
    run_dir: str | Path,
    *,
    family_name: str | None = None,
) -> list[PredictionRecord]:
    """按 family manifest 读取规范题级预测记录。"""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    if not index.prediction_records_path.exists():
        return []
    rows: list[PredictionRecord] = []
    with index.prediction_records_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(PredictionRecord.model_validate_json(stripped))
    return rows


def named_turn_record_paths(run_dir: str | Path, *, family_name: str | None = None) -> dict[str, Path]:
    """Return turn-record paths indexed by basename."""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    return {path.name: path for path in index.turn_record_paths}


def named_diagnostic_paths(run_dir: str | Path, *, family_name: str | None = None) -> dict[str, Path]:
    """Return diagnostic artifact paths indexed by basename."""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    return {path.name: path for path in index.diagnostic_paths}


def named_export_paths(run_dir: str | Path, *, family_name: str | None = None) -> dict[str, Path]:
    """Return export artifact paths indexed by basename."""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    return {path.name: path for path in index.export_paths}


def _resolve_family_name(run_dir: Path, manifest: dict[str, object]) -> str:
    for key in ("family_name", "family"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    if len(run_dir.parts) >= 4:
        return run_dir.parts[-4]
    raise RuntimeError(f"无法从运行目录 {run_dir.as_posix()} 解析 family_name。")


def _manifest_artifact_schema(manifest: dict[str, object]) -> dict[str, object] | None:
    payload = manifest.get("artifact_schema")
    return payload if isinstance(payload, dict) else None
