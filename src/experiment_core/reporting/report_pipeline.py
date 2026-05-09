"""共享的科研报告总装器。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from experiment_core.reporting.reporting_utils import build_published_report_name
from experiment_core.reporting.run_figures import append_figure_gallery_markdown, write_figure_bundle


@dataclass(frozen=True)
class SupplementalReport:
    """除正式 `report.md` 之外的附加报告。"""

    result_key: str
    filename: str
    content: str


def render_report_bundle(
    *,
    run_dir: str | Path,
    publish_dir: str | Path,
    manifest: dict[str, Any],
    base_markdown: str,
    figure_specs: list[dict[str, Any]],
    published_stem: str = "report",
    supplemental_reports: list[SupplementalReport] | None = None,
) -> dict[str, Any]:
    """统一写出 run 本地报告、发布报告、图资产与附加报告。"""

    root = Path(run_dir)
    target_publish_dir = Path(publish_dir)
    extras = supplemental_reports or []

    figure_bundle = write_figure_bundle(root, figure_specs)

    local_markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(local_markdown, encoding="utf-8")

    publish_path = target_publish_dir / build_published_report_name(manifest, stem=published_stem)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    published_markdown = append_figure_gallery_markdown(
        base_markdown,
        figure_bundle["figures"],
        run_dir=root,
        published_path=publish_path,
    )
    publish_path.write_text(published_markdown, encoding="utf-8")

    payload: dict[str, Any] = {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "figure_manifest": str(root / "figure_manifest.json"),
        "figures_dir": str(root / "figures"),
    }
    for report in extras:
        output_path = root / report.filename
        output_path.write_text(report.content, encoding="utf-8")
        payload[report.result_key] = str(output_path)
    return payload
