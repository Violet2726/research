"""family 报告入口的共享包装。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.workspace.layout import default_reports_root
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.reporting.scientific_report import render_scientific_report


def render_family_report_bundle(
    *,
    family_name: str,
    run_dir: str | Path,
    manifest: dict[str, Any],
    base_markdown: str,
    figure_specs: list[dict[str, Any]],
    publish_dir: str | Path | None = None,
    supplemental_reports: list[SupplementalReport] | None = None,
) -> dict[str, Any]:
    """统一处理 family 报告的默认发布目录与 bundle 渲染。"""

    resolved_publish_dir = publish_dir or default_reports_root(family_name)
    return render_report_bundle(
        run_dir=run_dir,
        publish_dir=resolved_publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=figure_specs,
        supplemental_reports=supplemental_reports,
    )


def render_family_scientific_report(*args: Any, **kwargs: Any) -> dict[str, Any]:
    """保留 family 侧统一调用点，便于后续继续收敛报告模板。"""

    return render_scientific_report(*args, **kwargs)
