"""family 运行目录布局的共享构造器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from research_experiments.core.contracts import FamilyArtifactSchema
from research_experiments.core.execution.runner_common import prepare_run_root
from research_experiments.families.registry import get_family_registration


@dataclass(frozen=True)
class FamilyRunLayout:
    """描述某个 family 单次运行的规范产物路径集合。"""

    root: Path
    schema: FamilyArtifactSchema
    manifest: Path
    progress: Path
    validation: Path
    report: Path
    figure_manifest: Path
    archive_manifest: Path
    metrics: Path
    predictions: Path
    run_summary: Path
    turns: dict[str, Path]
    diagnostics: dict[str, Path]
    exports: dict[str, Path]
    aliases: dict[str, Path]

    def turn_path(self, name: str) -> Path:
        return self.turns[name]

    def diagnostic_path(self, name: str) -> Path:
        return self.diagnostics[name]

    def export_path(self, name: str) -> Path:
        return self.exports[name]

    def __getattr__(self, name: str) -> Path:
        try:
            return self.aliases[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def prepare_family_run_layout(
    run_root: str | Path,
    experiment_name: str,
    phase_name: str,
    run_id: str,
    *,
    artifact_schema: FamilyArtifactSchema,
    aliases: dict[str, str] | None = None,
) -> FamilyRunLayout:
    """创建运行目录并解析规范产物路径。"""

    root = prepare_run_root(run_root, experiment_name, phase_name, run_id)
    paths = artifact_schema.build_paths(root)
    alias_paths = {name: root / relative_path for name, relative_path in (aliases or {}).items()}
    layout = FamilyRunLayout(
        root=root,
        schema=artifact_schema,
        manifest=paths["manifest_path"],
        progress=paths["progress_path"],
        validation=paths["validation_path"],
        report=paths["report_path"],
        figure_manifest=paths["figure_manifest_path"],
        archive_manifest=paths["archive_manifest_path"],
        metrics=paths["metrics_view_path"],
        predictions=paths["prediction_records_path"],
        run_summary=paths["run_summary_path"],
        turns={path.name: path for path in paths["turn_record_paths"]},
        diagnostics={path.name: path for path in paths["diagnostic_paths"]},
        exports={path.name: path for path in paths["export_paths"]},
        aliases=alias_paths,
    )
    _ensure_layout_directories(layout)
    return layout


def prepare_registered_run_layout(
    family_name: str,
    run_root: str | Path,
    experiment_name: str,
    phase_name: str,
    run_id: str,
) -> FamilyRunLayout:
    """按 family 注册信息构造运行目录布局。"""

    registration = get_family_registration(family_name)
    return prepare_family_run_layout(
        run_root,
        experiment_name,
        phase_name,
        run_id,
        artifact_schema=registration.artifact_schema,
        aliases=registration.artifact_aliases,
    )


def _ensure_layout_directories(layout: FamilyRunLayout) -> None:
    """提前创建布局涉及的全部父目录。"""

    required_dirs = {
        layout.root,
        layout.metrics.parent,
        layout.predictions.parent,
        layout.run_summary.parent,
        layout.report.parent,
        layout.figure_manifest.parent,
        layout.archive_manifest.parent,
    }
    required_dirs.update(path.parent for path in layout.turns.values())
    required_dirs.update(path.parent for path in layout.diagnostics.values())
    required_dirs.update(path.parent for path in layout.exports.values())
    required_dirs.update(path.parent for path in layout.aliases.values())
    required_dirs.add(layout.root / "figures")
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)
