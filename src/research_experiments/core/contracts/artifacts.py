"""运行产物目录的规范读模型。"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict


class RunArtifactIndex(BaseModel):
    """某个 run 目录在平台层暴露的规范产物索引。"""

    model_config = ConfigDict(frozen=True)

    family_name: str
    prototype: str
    run_dir: Path
    manifest_path: Path
    progress_path: Path
    validation_path: Path
    report_path: Path
    figure_manifest_path: Path
    archive_manifest_path: Path
    metrics_view_path: Path
    prediction_records_path: Path
    run_summary_path: Path
    turn_record_paths: tuple[Path, ...] = ()
    diagnostic_paths: tuple[Path, ...] = ()
    export_paths: tuple[Path, ...] = ()


__all__ = ["RunArtifactIndex"]
