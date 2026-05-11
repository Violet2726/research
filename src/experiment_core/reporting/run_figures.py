"""统一管理 run 级科研图资产。"""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
import math
import os
from typing import Any

from experiment_core.reporting.report_views import coerce_summary_rows


FONT_FAMILY = "Helvetica, Arial, sans-serif"
COLOR_TEXT = "#111827"
COLOR_MUTED = "#6b7280"
COLOR_GRID = "#d1d5db"
COLOR_AXIS = "#374151"
COLOR_BLUE = "#0072B2"
COLOR_ORANGE = "#E69F00"
COLOR_GREEN = "#009E73"
COLOR_RED = "#D55E00"
COLOR_PURPLE = "#CC79A7"
COLOR_GRAY = "#9ca3af"

CORE_FIGURE_IDS = (
    "frontier_overall",
    "efficiency_rank_overall",
    "score_by_dataset",
)


def write_figure_bundle(run_dir: str | Path, figure_specs: list[dict[str, Any]]) -> dict[str, Any]:
    """把 figure specs 渲染成 run 本地 `figures/` 资产与 manifest。"""
    root = Path(run_dir)
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows: list[dict[str, Any]] = []
    for spec in figure_specs:
        figure_id = str(spec["figure_id"])
        svg_path = figures_dir / f"{figure_id}.svg"
        csv_path = figures_dir / f"{figure_id}.csv"
        normalized = _normalize_figure_spec(spec)
        svg_path.write_text(_render_svg(normalized), encoding="utf-8")
        csv_path.write_text(_render_points_csv(normalized.get("data", [])), encoding="utf-8")
        manifest_rows.append(
            {
                "figure_id": figure_id,
                "title": str(normalized["title"]),
                "caption": str(normalized["caption"]),
                "takeaway": str(normalized.get("takeaway") or ""),
                "svg_path": f"figures/{figure_id}.svg",
                "csv_path": f"figures/{figure_id}.csv",
                "source_kind": str(normalized["source_kind"]),
                "dataset_scope": str(normalized["dataset_scope"]),
                "primary_metric": str(normalized["primary_metric"]),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "figure_count": len(manifest_rows),
        "figures": manifest_rows,
    }
    manifest_path = root / "figure_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "figures_dir": figures_dir.as_posix(),
        "figure_manifest": manifest_path.as_posix(),
        "figures": manifest_rows,
    }


def append_figure_gallery_markdown(
    markdown: str,
    figures: list[dict[str, Any]],
    *,
    run_dir: str | Path,
    published_path: str | Path | None = None,
) -> str:
    """在 Markdown 末尾追加图表画廊。"""
    if not figures:
        return markdown
    body = markdown.rstrip() + "\n\n## 图表资产\n"
    run_root = Path(run_dir)
    publish_target = Path(published_path) if published_path is not None else None
    for figure in figures:
        svg_path = str(figure["svg_path"])
        if publish_target is None:
            link = svg_path
        else:
            link = os.path.relpath(run_root / svg_path, publish_target.parent).replace("\\", "/")
        body += (
            "\n"
            f"### {figure['title']}\n\n"
            f"![{figure['title']}]({link})\n\n"
            f"*{figure['caption']}*\n"
        )
        takeaway = str(figure.get("takeaway") or "").strip()
        if takeaway:
            body += f"\n要点：{takeaway}\n"
    return body.rstrip() + "\n"


def validate_figure_contract(
    run_dir: str | Path,
    *,
    report_name: str = "report.md",
    required_core_ids: tuple[str, ...] = CORE_FIGURE_IDS,
) -> dict[str, Any]:
    """检查 run 是否满足统一图资产合同。"""
    root = Path(run_dir)
    manifest_path = root / "figure_manifest.json"
    report_path = root / report_name
    manifest_payload = _safe_load_json(manifest_path)
    figures = manifest_payload.get("figures", []) if isinstance(manifest_payload, dict) else []
    figure_lookup = {
        str(row.get("figure_id")): row
        for row in figures
        if isinstance(row, dict) and row.get("figure_id")
    }

    missing_core_ids = [figure_id for figure_id in required_core_ids if figure_id not in figure_lookup]
    missing_files: list[str] = []
    for row in figure_lookup.values():
        for key in ("svg_path", "csv_path"):
            target = root / str(row.get(key) or "")
            if not target.exists():
                missing_files.append(target.as_posix())

    report_text = report_path.read_text(encoding="utf-8") if report_path.exists() else ""
    referenced_svg_paths = [
        str(row.get("svg_path"))
        for row in figure_lookup.values()
        if str(row.get("svg_path") or "") in report_text
    ]
    return {
        "passed": manifest_path.exists()
        and not missing_core_ids
        and not missing_files
        and bool(referenced_svg_paths),
        "figure_count": len(figure_lookup),
        "missing_core_ids": missing_core_ids,
        "missing_files": missing_files,
        "report_references_count": len(referenced_svg_paths),
        "report_name": report_name,
    }


def build_frontier_figure_spec(
    summary_rows: list[dict[str, Any]],
    *,
    title: str,
    caption: str,
    score_field: str,
    primary_metric: str,
    method_label_field: str = "display_name",
    figure_id: str = "frontier_overall",
) -> dict[str, Any]:
    """构建统一的成本-效果前沿图。"""
    rows = [row for row in coerce_summary_rows(summary_rows) if row.dataset == "overall"]
    data = [
        {
            "label": row.label(method_label_field),
            "short_label": row.short_label(method_label_field),
            "x": _as_float(row.number("total_tokens_mean")),
            "y": _as_float(row.number(score_field)),
            "value": _as_float(row.number(score_field)),
            "method_name": row.method_name,
        }
        for row in rows
        if row.number(score_field) is not None
    ]
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _frontier_takeaway(data, primary_metric),
        "renderer": "scatter",
        "source_kind": "metrics.summary",
        "dataset_scope": "overall",
        "primary_metric": primary_metric,
        "x_label": "平均总 token / 题",
        "y_label": primary_metric,
        "note": "横轴越小表示成本越低，纵轴越高表示性能越好。",
        "reference_y": 0.0 if primary_metric.lower().endswith("delta") else None,
        "data": sorted(data, key=lambda row: (float(row["x"]), -float(row["y"]), str(row["label"]))),
    }


def build_efficiency_rank_figure_spec(
    summary_rows: list[dict[str, Any]],
    *,
    title: str,
    caption: str,
    efficiency_field: str,
    primary_metric: str,
    method_label_field: str = "display_name",
    figure_id: str = "efficiency_rank_overall",
) -> dict[str, Any]:
    """构建统一的效率排序图。"""
    rows = [row for row in coerce_summary_rows(summary_rows) if row.dataset == "overall"]
    data = [
        {
            "label": row.label(method_label_field),
            "short_label": row.short_label(method_label_field),
            "value": _as_float(row.number(efficiency_field)),
            "method_name": row.method_name,
        }
        for row in rows
        if row.number(efficiency_field) is not None
    ]
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _rank_takeaway(data, primary_metric),
        "renderer": "rank_bar",
        "source_kind": "metrics.summary",
        "dataset_scope": "overall",
        "primary_metric": primary_metric,
        "x_label": primary_metric,
        "note": "方法按效率从高到低排序，便于直接比较成本收益。",
        "data": sorted(data, key=lambda row: (-float(row["value"]), str(row["label"]))),
    }


def build_score_by_dataset_figure_spec(
    summary_rows: list[dict[str, Any]],
    *,
    title: str,
    caption: str,
    score_field: str,
    primary_metric: str,
    method_label_field: str = "display_name",
    figure_id: str = "score_by_dataset",
) -> dict[str, Any]:
    """构建统一的跨数据集得分矩阵图。"""
    rows = [row for row in coerce_summary_rows(summary_rows) if row.dataset != "overall"]
    data = [
        {
            "dataset": row.dataset,
            "label": row.label(method_label_field),
            "short_label": row.short_label(method_label_field),
            "value": _as_float(row.number(score_field)),
            "method_name": row.method_name,
        }
        for row in rows
        if row.number(score_field) is not None
    ]
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _dataset_takeaway(data, primary_metric),
        "renderer": "matrix_dot",
        "source_kind": "metrics.summary",
        "dataset_scope": "per_dataset",
        "primary_metric": primary_metric,
        "note": "圆点大小和颜色深浅共同编码分数，便于观察跨数据集稳定性。",
        "data": sorted(data, key=lambda row: (str(row["dataset"]), str(row["label"]))),
    }


def build_scatter_figure_spec(
    *,
    figure_id: str,
    title: str,
    caption: str,
    primary_metric: str,
    data: list[dict[str, Any]],
    x_label: str,
    y_label: str,
    source_kind: str,
    dataset_scope: str,
    note: str,
    reference_x: float | None = None,
    reference_y: float | None = None,
) -> dict[str, Any]:
    """构建通用散点图。"""
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _scatter_takeaway(data, primary_metric),
        "renderer": "scatter",
        "source_kind": source_kind,
        "dataset_scope": dataset_scope,
        "primary_metric": primary_metric,
        "x_label": x_label,
        "y_label": y_label,
        "reference_x": reference_x,
        "reference_y": reference_y,
        "note": note,
        "data": data,
    }


def build_grouped_bar_figure_spec(
    *,
    figure_id: str,
    title: str,
    caption: str,
    primary_metric: str,
    data: list[dict[str, Any]],
    series: list[tuple[str, str]],
    x_label: str,
    source_kind: str,
    dataset_scope: str,
    note: str,
) -> dict[str, Any]:
    """构建通用分组条形图。"""
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _grouped_bar_takeaway(data, series),
        "renderer": "grouped_bar",
        "source_kind": source_kind,
        "dataset_scope": dataset_scope,
        "primary_metric": primary_metric,
        "x_label": x_label,
        "series": [{"key": key, "label": label} for key, label in series],
        "note": note,
        "data": data,
    }


def build_interval_figure_spec(
    *,
    figure_id: str,
    title: str,
    caption: str,
    primary_metric: str,
    data: list[dict[str, Any]],
    x_label: str,
    source_kind: str,
    dataset_scope: str,
    note: str,
) -> dict[str, Any]:
    """构建置信区间型比较图。"""
    return {
        "figure_id": figure_id,
        "title": title,
        "caption": caption,
        "takeaway": _interval_takeaway(data, primary_metric),
        "renderer": "interval",
        "source_kind": source_kind,
        "dataset_scope": dataset_scope,
        "primary_metric": primary_metric,
        "x_label": x_label,
        "note": note,
        "reference_x": 0.0,
        "data": data,
    }


def _normalize_figure_spec(spec: dict[str, Any]) -> dict[str, Any]:
    payload = dict(spec)
    payload.setdefault("caption", "")
    payload.setdefault("takeaway", "")
    payload.setdefault("source_kind", "metrics.summary")
    payload.setdefault("dataset_scope", "overall")
    payload.setdefault("primary_metric", "")
    payload.setdefault("data", [])
    payload.setdefault("note", "")
    return payload


def _render_svg(spec: dict[str, Any]) -> str:
    renderer = str(spec.get("renderer") or "")
    if renderer == "scatter":
        return _render_scatter_svg(spec)
    if renderer == "rank_bar":
        return _render_rank_bar_svg(spec)
    if renderer == "matrix_dot":
        return _render_matrix_dot_svg(spec)
    if renderer == "grouped_bar":
        return _render_grouped_bar_svg(spec)
    if renderer == "interval":
        return _render_interval_svg(spec)
    raise ValueError(f"Unsupported figure renderer: {renderer}")


def _render_scatter_svg(spec: dict[str, Any]) -> str:
    title = str(spec["title"])
    caption = str(spec["caption"])
    data = list(spec.get("data", []))
    if not data:
        return _render_empty_svg(title, caption)

    width = 960
    height = 560
    left = 88
    right = 42
    top = 78
    bottom = 86
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(row["x"]) for row in data]
    y_values = [float(row["y"]) for row in data]
    include_x = [float(spec["reference_x"])] if spec.get("reference_x") is not None else []
    include_y = [float(spec["reference_y"])] if spec.get("reference_y") is not None else []
    x_min, x_max = _expand_domain(x_values, include_values=include_x, pad_fraction=0.08)
    y_min, y_max = _expand_domain(y_values, include_values=include_y, pad_fraction=0.15)
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    y_ticks = _nice_ticks(y_min, y_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    y_min, y_max = y_ticks[0], y_ticks[-1]

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    lines.extend(_svg_axes_and_grid(left, top, plot_width, plot_height, x_ticks, y_ticks, x_min, x_max, y_min, y_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label=str(spec["x_label"]),
            y_label=str(spec["y_label"]),
        )
    )
    if spec.get("reference_x") is not None:
        x = _scale_linear(float(spec["reference_x"]), x_min, x_max, left, left + plot_width)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
        )
    if spec.get("reference_y") is not None:
        y = _scale_linear(float(spec["reference_y"]), y_min, y_max, top + plot_height, top)
        lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
        )

    for index, row in enumerate(sorted(data, key=lambda item: (float(item["x"]), -float(item["y"])))):
        cx = _scale_linear(float(row["x"]), x_min, x_max, left, left + plot_width)
        cy = _scale_linear(float(row["y"]), y_min, y_max, top + plot_height, top)
        color = _series_color(index)
        label_dx = 10 if index % 2 == 0 else -12
        anchor = "start" if label_dx > 0 else "end"
        label_y = cy - 8 if index % 3 != 1 else cy + 18
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5.5" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{cx + label_dx:.2f}" y="{label_y:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" text-anchor="{anchor}" fill="{COLOR_TEXT}">{escape(str(row.get("short_label") or row.get("label") or ""))}</text>'
        )
    lines.extend(_svg_note_block(height, str(spec["note"])))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_rank_bar_svg(spec: dict[str, Any]) -> str:
    title = str(spec["title"])
    caption = str(spec["caption"])
    data = list(spec.get("data", []))
    if not data:
        return _render_empty_svg(title, caption)

    width = 1040
    row_height = 30
    top = 86
    bottom = 72
    left = 260
    right = 40
    height = max(420, top + bottom + row_height * len(data))
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_value = max(float(row["value"]) for row in data)
    x_min, x_max = 0.0, _expand_domain([max_value], include_values=[0.0], pad_fraction=0.15)[1]
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    band_step = plot_height / max(1, len(data))
    bar_height = max(12.0, band_step * 0.56)

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    lines.extend(_svg_vertical_grid(left, top, plot_width, plot_height, x_ticks, x_min, x_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label=str(spec["x_label"]),
            y_label=None,
        )
    )
    for index, row in enumerate(data):
        value = float(row["value"])
        y = top + index * band_step + (band_step - bar_height) / 2
        x_end = _scale_linear(value, x_min, x_max, left, left + plot_width)
        color = _series_color(index)
        lines.append(
            f'<text x="{left - 12}" y="{y + bar_height / 2 + 4:.2f}" text-anchor="end" '
            f'font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(str(row.get("short_label") or row.get("label") or ""))}</text>'
        )
        lines.append(
            f'<rect x="{left:.2f}" y="{y:.2f}" width="{max(x_end - left, 0.8):.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" opacity="0.92"/>'
        )
        lines.append(
            f'<text x="{x_end + 8:.2f}" y="{y + bar_height / 2 + 4:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{value:.3f}</text>'
        )
    lines.extend(_svg_note_block(height, str(spec["note"])))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_matrix_dot_svg(spec: dict[str, Any]) -> str:
    title = str(spec["title"])
    caption = str(spec["caption"])
    data = list(spec.get("data", []))
    if not data:
        return _render_empty_svg(title, caption)

    datasets = sorted({str(row["dataset"]) for row in data})
    methods = sorted({str(row.get("short_label") or row.get("label") or "") for row in data})
    value_lookup = {
        (str(row["dataset"]), str(row.get("short_label") or row.get("label") or "")): float(row["value"])
        for row in data
    }
    width = max(760, 240 + 130 * len(datasets))
    height = max(420, 140 + 42 * len(methods))
    left = 210
    right = 36
    top = 88
    bottom = 84
    plot_width = width - left - right
    plot_height = height - top - bottom
    cell_width = plot_width / max(1, len(datasets))
    cell_height = plot_height / max(1, len(methods))
    max_value = max(value_lookup.values()) if value_lookup else 1.0

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    for col, dataset in enumerate(datasets):
        x = left + col * cell_width + cell_width / 2
        lines.append(
            f'<text x="{x:.2f}" y="{top - 14}" text-anchor="middle" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(dataset)}</text>'
        )
        lines.append(
            f'<line x1="{left + col * cell_width:.2f}" y1="{top}" x2="{left + col * cell_width:.2f}" y2="{top + plot_height}" '
            f'stroke="{COLOR_GRID}" stroke-width="1"/>'
        )
    lines.append(
        f'<line x1="{left + plot_width:.2f}" y1="{top}" x2="{left + plot_width:.2f}" y2="{top + plot_height}" '
        f'stroke="{COLOR_GRID}" stroke-width="1"/>'
    )
    for row_index, method in enumerate(methods):
        y = top + row_index * cell_height + cell_height / 2
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(method)}</text>'
        )
        lines.append(
            f'<line x1="{left}" y1="{top + row_index * cell_height:.2f}" x2="{left + plot_width}" y2="{top + row_index * cell_height:.2f}" '
            f'stroke="{COLOR_GRID}" stroke-width="1"/>'
        )
        for col, dataset in enumerate(datasets):
            value = value_lookup.get((dataset, method))
            if value is None:
                continue
            cx = left + col * cell_width + cell_width / 2
            radius = 5.0 + 13.0 * (value / max_value if max_value else 0.0)
            color = COLOR_BLUE if value >= max_value * 0.66 else COLOR_ORANGE if value >= max_value * 0.33 else COLOR_GRAY
            lines.append(
                f'<circle cx="{cx:.2f}" cy="{y:.2f}" r="{radius:.2f}" fill="{color}" opacity="0.85" stroke="#ffffff" stroke-width="1.2"/>'
            )
            lines.append(
                f'<text x="{cx:.2f}" y="{y + 4:.2f}" text-anchor="middle" font-size="10" '
                f'font-family="{FONT_FAMILY}" fill="#ffffff">{value:.2f}</text>'
            )
    lines.append(
        f'<line x1="{left}" y1="{top + plot_height:.2f}" x2="{left + plot_width}" y2="{top + plot_height:.2f}" '
        f'stroke="{COLOR_GRID}" stroke-width="1"/>'
    )
    lines.extend(_svg_note_block(height, str(spec["note"])))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_grouped_bar_svg(spec: dict[str, Any]) -> str:
    title = str(spec["title"])
    caption = str(spec["caption"])
    data = list(spec.get("data", []))
    series = list(spec.get("series", []))
    if not data or not series:
        return _render_empty_svg(title, caption)

    width = 1080
    group_height = 24 + 18 * len(series)
    top = 88
    bottom = 84
    left = 260
    right = 48
    height = max(460, top + bottom + group_height * len(data))
    plot_width = width - left - right
    plot_height = height - top - bottom
    all_values = [
        _as_float(row.get(item["key"]))
        for row in data
        for item in series
    ]
    x_min, x_max = 0.0, _expand_domain(all_values, include_values=[0.0], pad_fraction=0.15)[1]
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    band_step = plot_height / max(1, len(data))
    bar_height = max(7.0, band_step * 0.16)

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    lines.extend(_svg_vertical_grid(left, top, plot_width, plot_height, x_ticks, x_min, x_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label=str(spec["x_label"]),
            y_label=None,
        )
    )
    lines.extend(
        _svg_series_legend(
            x=width - right - 180,
            y=44,
            items=[(str(item["label"]), _series_color(index)) for index, item in enumerate(series)],
        )
    )

    for row_index, row in enumerate(data):
        group_y = top + row_index * band_step
        lines.append(
            f'<text x="{left - 12}" y="{group_y + band_step * 0.46:.2f}" text-anchor="end" '
            f'font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(str(row.get("short_label") or row.get("label") or ""))}</text>'
        )
        for series_index, item in enumerate(series):
            value = _as_float(row.get(item["key"]))
            y = group_y + band_step * (0.18 + 0.24 * series_index)
            x_end = _scale_linear(value, x_min, x_max, left, left + plot_width)
            lines.append(
                f'<rect x="{left:.2f}" y="{y:.2f}" width="{max(x_end - left, 0.8):.2f}" height="{bar_height:.2f}" '
                f'fill="{_series_color(series_index)}" opacity="0.9"/>'
            )
            lines.append(
                f'<text x="{x_end + 8:.2f}" y="{y + bar_height - 1:.2f}" font-size="10" '
                f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{value:.3f}</text>'
            )
    lines.extend(_svg_note_block(height, str(spec["note"])))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_interval_svg(spec: dict[str, Any]) -> str:
    title = str(spec["title"])
    caption = str(spec["caption"])
    data = list(spec.get("data", []))
    if not data:
        return _render_empty_svg(title, caption)

    width = 1040
    row_height = 32
    top = 86
    bottom = 74
    left = 280
    right = 42
    height = max(420, top + bottom + row_height * len(data))
    plot_width = width - left - right
    plot_height = height - top - bottom

    values = []
    for row in data:
        values.extend([_as_float(row.get("low")), _as_float(row.get("high")), _as_float(row.get("value"))])
    include_x = [float(spec["reference_x"])] if spec.get("reference_x") is not None else []
    x_min, x_max = _expand_domain(values, include_values=include_x, pad_fraction=0.15)
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    band_step = plot_height / max(1, len(data))

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    lines.extend(_svg_vertical_grid(left, top, plot_width, plot_height, x_ticks, x_min, x_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label=str(spec["x_label"]),
            y_label=None,
        )
    )
    if spec.get("reference_x") is not None:
        x = _scale_linear(float(spec["reference_x"]), x_min, x_max, left, left + plot_width)
        lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
        )

    for index, row in enumerate(data):
        y = top + index * band_step + band_step / 2
        low = _scale_linear(_as_float(row.get("low")), x_min, x_max, left, left + plot_width)
        high = _scale_linear(_as_float(row.get("high")), x_min, x_max, left, left + plot_width)
        value = _scale_linear(_as_float(row.get("value")), x_min, x_max, left, left + plot_width)
        lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(str(row.get("short_label") or row.get("label") or ""))}</text>'
        )
        lines.append(
            f'<line x1="{low:.2f}" y1="{y:.2f}" x2="{high:.2f}" y2="{y:.2f}" stroke="{COLOR_BLUE}" stroke-width="2.2"/>'
        )
        lines.append(
            f'<circle cx="{value:.2f}" cy="{y:.2f}" r="5.5" fill="{COLOR_ORANGE}" stroke="#ffffff" stroke-width="1.2"/>'
        )
        lines.append(
            f'<text x="{high + 8:.2f}" y="{y + 4:.2f}" font-size="10" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">'
            f'{_as_float(row.get("value")):.3f} [{_as_float(row.get("low")):.3f}, {_as_float(row.get("high")):.3f}]</text>'
        )
    lines.extend(_svg_note_block(height, str(spec["note"])))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_points_csv(points: list[dict[str, Any]]) -> str:
    headers = _points_csv_headers(points)
    lines = [",".join(headers)]
    for point in points:
        lines.append(",".join(_csv_cell(str(point.get(header, ""))) for header in headers))
    return "\n".join(lines) + "\n"


def _render_empty_svg(title: str, caption: str) -> str:
    width = 960
    height = 220
    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, caption))
    lines.append(
        f'<text x="48" y="132" font-size="14" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">No eligible data points.</text>'
    )
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _svg_canvas(width: int, height: int) -> list[str]:
    return [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
    ]


def _svg_title_block(title: str, caption: str) -> list[str]:
    return [
        f'<text x="48" y="34" font-size="20" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}" font-weight="600">{escape(title)}</text>',
        f'<text x="48" y="56" font-size="12" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{escape(caption)}</text>',
    ]


def _svg_note_block(height: int, note: str) -> list[str]:
    return [
        f'<text x="48" y="{height - 18}" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{escape(note)}</text>'
    ]


def _svg_axes_and_grid(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_ticks: list[float],
    y_ticks: list[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> list[str]:
    lines: list[str] = []
    x0 = left
    x1 = left + plot_width
    y0 = top
    y1 = top + plot_height
    for tick in x_ticks:
        x = _scale_linear(tick, x_min, x_max, x0, x1)
        lines.append(f'<line x1="{x:.2f}" y1="{y0}" x2="{x:.2f}" y2="{y1}" stroke="{COLOR_GRID}" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{y1 + 20}" text-anchor="middle" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{_format_tick(tick)}</text>'
        )
    for tick in y_ticks:
        y = _scale_linear(tick, y_min, y_max, y1, y0)
        lines.append(f'<line x1="{x0}" y1="{y:.2f}" x2="{x1}" y2="{y:.2f}" stroke="{COLOR_GRID}" stroke-width="1"/>')
        lines.append(
            f'<text x="{x0 - 10}" y="{y + 4:.2f}" text-anchor="end" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{_format_tick(tick)}</text>'
        )
    lines.append(f'<rect x="{x0}" y="{y0}" width="{plot_width}" height="{plot_height}" fill="none" stroke="{COLOR_AXIS}" stroke-width="1.2"/>')
    return lines


def _svg_vertical_grid(
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_ticks: list[float],
    x_min: float,
    x_max: float,
) -> list[str]:
    lines: list[str] = []
    x1 = left + plot_width
    y1 = top + plot_height
    for tick in x_ticks:
        x = _scale_linear(tick, x_min, x_max, left, x1)
        lines.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{y1}" stroke="{COLOR_GRID}" stroke-width="1"/>')
        lines.append(
            f'<text x="{x:.2f}" y="{y1 + 20}" text-anchor="middle" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{_format_tick(tick)}</text>'
        )
    lines.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{y1}" stroke="{COLOR_AXIS}" stroke-width="1.2"/>')
    lines.append(f'<line x1="{left}" y1="{y1}" x2="{x1}" y2="{y1}" stroke="{COLOR_AXIS}" stroke-width="1.2"/>')
    return lines


def _svg_axis_labels(
    *,
    left: int,
    top: int,
    plot_width: int,
    plot_height: int,
    x_label: str,
    y_label: str | None,
) -> list[str]:
    lines = [
        f'<text x="{left + plot_width / 2:.2f}" y="{top + plot_height + 48:.2f}" text-anchor="middle" font-size="12" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(x_label)}</text>'
    ]
    if y_label:
        lines.append(
            f'<text x="24" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="12" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}" transform="rotate(-90 24,{top + plot_height / 2:.2f})">{escape(y_label)}</text>'
        )
    return lines


def _svg_series_legend(x: int, y: int, items: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for index, (label, color) in enumerate(items):
        y_offset = y + index * 18
        lines.append(f'<rect x="{x}" y="{y_offset - 9}" width="12" height="12" fill="{color}" opacity="0.92"/>')
        lines.append(
            f'<text x="{x + 18}" y="{y_offset + 1}" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
    return lines


def _points_csv_headers(points: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "label",
        "short_label",
        "dataset",
        "method_name",
        "x",
        "y",
        "value",
        "low",
        "high",
    ]
    seen = {key for point in points for key in point}
    headers = [key for key in preferred if key in seen]
    headers.extend(sorted(key for key in seen if key not in headers))
    return headers or ["label", "value"]


def _series_color(index: int) -> str:
    palette = [COLOR_BLUE, COLOR_ORANGE, COLOR_GREEN, COLOR_RED, COLOR_PURPLE, COLOR_GRAY]
    return palette[index % len(palette)]


def _frontier_takeaway(data: list[dict[str, Any]], primary_metric: str) -> str:
    if not data:
        return ""
    best = max(data, key=lambda row: float(row.get("y") or 0.0))
    cheapest = min(data, key=lambda row: float(row.get("x") or 0.0))
    return (
        f"{primary_metric}最高的方法是 `{best.get('label')}`；成本最低的方法是 `{cheapest.get('label')}`。"
    )


def _rank_takeaway(data: list[dict[str, Any]], primary_metric: str) -> str:
    if not data:
        return ""
    best = max(data, key=lambda row: float(row.get("value") or 0.0))
    return f"{primary_metric}最高的方法是 `{best.get('label')}`，数值为 {float(best.get('value') or 0.0):.4f}。"


def _dataset_takeaway(data: list[dict[str, Any]], primary_metric: str) -> str:
    if not data:
        return ""
    datasets = sorted({str(row.get('dataset') or '') for row in data if row.get("dataset")})
    methods = sorted({str(row.get('label') or '') for row in data if row.get("label")})
    return f"该图覆盖 `{len(datasets)}` 个数据集与 `{len(methods)}` 个方法，用于观察 {primary_metric} 的跨数据集稳定性。"


def _scatter_takeaway(data: list[dict[str, Any]], primary_metric: str) -> str:
    if not data:
        return ""
    best = max(data, key=lambda row: float(row.get("y") or row.get("value") or 0.0))
    return f"图中表现最优的点对应 `{best.get('label')}`，其 {primary_metric} 为 {float(best.get('y') or best.get('value') or 0.0):.4f}。"


def _grouped_bar_takeaway(data: list[dict[str, Any]], series: list[tuple[str, str]]) -> str:
    if not data or not series:
        return ""
    first_key, first_label = series[0]
    best = max(data, key=lambda row: float(row.get(first_key) or 0.0))
    return f"按 `{first_label}` 指标看，最高的对象是 `{best.get('label')}`。"


def _interval_takeaway(data: list[dict[str, Any]], primary_metric: str) -> str:
    if not data:
        return ""
    best = max(data, key=lambda row: float(row.get("value") or 0.0))
    excludes_zero = [
        row for row in data
        if float(row.get("low") or 0.0) > 0.0 or float(row.get("high") or 0.0) < 0.0
    ]
    qualifier = f"其中有 `{len(excludes_zero)}` 个比较的区间整体不跨 0。" if excludes_zero else "当前所有比较的区间都覆盖或接触 0。"
    return f"`{best.get('label')}` 的 {primary_metric}差值最高；{qualifier}"


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _csv_cell(value: str) -> str:
    if any(char in value for char in [",", "\"", "\n"]):
        return '"' + value.replace('"', '""') + '"'
    return value


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _nice_ticks(min_value: float, max_value: float, *, target_ticks: int) -> list[float]:
    if math.isclose(min_value, max_value):
        return [min_value]
    span = max_value - min_value
    raw_step = span / max(target_ticks - 1, 1)
    step = _nice_step(raw_step)
    tick_min = math.floor(min_value / step) * step
    tick_max = math.ceil(max_value / step) * step
    ticks: list[float] = []
    cursor = tick_min
    while cursor <= tick_max + step * 0.5:
        ticks.append(round(cursor, 10))
        cursor += step
    return ticks


def _nice_step(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1.0:
        nice_fraction = 1.0
    elif fraction <= 2.0:
        nice_fraction = 2.0
    elif fraction <= 2.5:
        nice_fraction = 2.5
    elif fraction <= 5.0:
        nice_fraction = 5.0
    else:
        nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def _expand_domain(values: list[float], *, include_values: list[float], pad_fraction: float) -> tuple[float, float]:
    materialized = list(values) + list(include_values)
    lower = min(materialized)
    upper = max(materialized)
    if math.isclose(lower, upper):
        padding = max(abs(lower) * pad_fraction, 0.1)
        return lower - padding, upper + padding
    span = upper - lower
    padding = span * pad_fraction
    return lower - padding, upper + padding


def _scale_linear(value: float, domain_min: float, domain_max: float, range_min: float, range_max: float) -> float:
    if math.isclose(domain_min, domain_max):
        return (range_min + range_max) / 2.0
    ratio = (value - domain_min) / (domain_max - domain_min)
    return range_min + ratio * (range_max - range_min)


def _format_tick(value: float) -> str:
    if math.isclose(value, round(value), abs_tol=1e-9):
        return str(int(round(value)))
    if abs(value) >= 1:
        return f"{value:.2f}".rstrip("0").rstrip(".")
    return f"{value:.3f}".rstrip("0").rstrip(".")
