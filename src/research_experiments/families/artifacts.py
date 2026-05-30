"""family manifest 驱动的运行产物索引读取接口。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

from research_experiments.core.contracts import PredictionRecord, RunArtifactIndex
from research_experiments.families.registry import get_family_manifest, registered_family_names


def resolve_run_artifact_index(
    run_dir: str | Path,
    *,
    family_name: str | None = None,
) -> RunArtifactIndex:
    """根据 family manifest 解析某个 run 目录的正式产物索引。"""

    root = Path(run_dir)
    resolved_family = family_name or _resolve_family_name(root)
    manifest = get_family_manifest(resolved_family)
    paths = manifest.build_artifact_paths(root)
    return RunArtifactIndex(
        family_name=manifest.family_name,
        prototype=manifest.prototype,
        run_dir=root,
        manifest_path=cast(Path, paths["manifest_path"]),
        progress_path=cast(Path, paths["progress_path"]),
        validation_path=cast(Path, paths["validation_path"]),
        report_path=cast(Path, paths["report_path"]),
        figure_manifest_path=cast(Path, paths["figure_manifest_path"]),
        metrics_view_path=cast(Path, paths["metrics_view_path"]),
        prediction_records_path=cast(Path, paths["prediction_records_path"]),
        turn_record_paths=cast(tuple[Path, ...], paths["turn_record_paths"]),
        extra_view_paths=cast(tuple[Path, ...], paths["extra_view_paths"]),
    )


def load_run_manifest(run_dir: str | Path) -> dict[str, object]:
    """读取某个 run 的 manifest.json。"""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {}
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_metrics_payload(run_dir: str | Path, *, family_name: str | None = None) -> dict[str, object]:
    """按 family manifest 读取规范指标视图。"""

    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    if not index.metrics_view_path.exists():
        return {}
    return json.loads(index.metrics_view_path.read_text(encoding="utf-8"))


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


def _resolve_family_name(run_dir: Path) -> str:
    manifest = load_run_manifest(run_dir)
    for key in ("family_name", "family"):
        value = str(manifest.get(key) or "").strip()
        if value:
            return value
    known = set(registered_family_names())
    for part in run_dir.parts:
        if part in known:
            return part
    raise RuntimeError(f"无法从运行目录 {run_dir.as_posix()} 解析 family_name。")
