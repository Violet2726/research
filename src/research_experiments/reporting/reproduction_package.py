"""为 reproduction matrix 生成复现导向的报告包。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from research_experiments.matrix.reproduction_analysis import render_reproduction_analysis
from research_experiments.workspace.layout import default_reports_root


def render_reproduction_package(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
) -> dict[str, str]:
    state_path = _resolve_state_path(state_path_or_root)
    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)

    analysis_path = root / "reproduction_analysis.json"
    if not analysis_path.exists():
        render_reproduction_analysis(state_path, output_root=root)

    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    package = build_reproduction_package_payload(analysis)

    package_json = root / "reproduction_package.json"
    package_md = root / "reproduction_package.md"
    published = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("reproduction_matrix")) / f"{state_path.parent.name}-reproduction_package.md"
    )

    markdown = render_reproduction_package_markdown(package)
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package_md.write_text(markdown, encoding="utf-8")
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(markdown, encoding="utf-8")
    return {
        "package_json": package_json.as_posix(),
        "package_markdown": package_md.as_posix(),
        "published_path": published.as_posix(),
    }


def build_reproduction_package_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    entries = analysis.get("entries", [])
    canonical_rows = [entry["overall_row"] for entry in entries if entry.get("entry_role") == "canonical" and entry.get("overall_row")]
    auxiliary_rows = [
        entry["overall_row"]
        for entry in entries
        if entry.get("entry_role") in {"ablation", "control", "reference"} and entry.get("overall_row")
    ]
    track_sections: dict[str, list[dict[str, Any]]] = {}
    scaling_sections: list[dict[str, Any]] = []
    for entry in entries:
        track_sections.setdefault(str(entry.get("track_name") or "other"), [])
        if entry.get("overall_row"):
            track_sections[str(entry.get("track_name") or "other")].append(entry["overall_row"])
        if entry.get("analysis_mode") == "scaling_summary":
            scaling_sections.append(
                {
                    "family": entry.get("family"),
                    "experiment_name": entry.get("experiment_name"),
                    "track_name": entry.get("track_name"),
                    "scaling_summary": entry.get("scaling_summary") or {},
                }
            )
    for rows in track_sections.values():
        rows.sort(key=lambda row: (str(row.get("entry_role") or ""), -(row.get("primary_metric_value") or 0.0), str(row.get("experiment_name") or "")))
    canonical_rows.sort(key=lambda row: (str(row.get("track_name") or ""), -(row.get("primary_metric_value") or 0.0)))
    auxiliary_rows.sort(key=lambda row: (str(row.get("track_name") or ""), str(row.get("entry_role") or ""), -(row.get("primary_metric_value") or 0.0)))
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_id": analysis.get("matrix_id"),
        "phase_name": analysis.get("phase_name"),
        "model_ref": analysis.get("model_ref"),
        "counts": analysis.get("counts", {}),
        "canonical_board": canonical_rows,
        "auxiliary_board": auxiliary_rows,
        "track_sections": track_sections,
        "scaling_sections": scaling_sections,
    }


def render_reproduction_package_markdown(package: dict[str, Any]) -> str:
    lines = [
        "# Reproduction Package",
        "",
        f"- generated_at: `{package.get('generated_at')}`",
        f"- matrix_id: `{package.get('matrix_id')}`",
        f"- phase_name: `{package.get('phase_name')}`",
        f"- model_ref: `{package.get('model_ref')}`",
        "",
        "## Canonical Board",
        "",
    ]
    lines.extend(_render_table(package.get("canonical_board", [])))
    lines.extend(["", "## Auxiliary Board", ""])
    lines.extend(_render_table(package.get("auxiliary_board", [])))
    for track_name, rows in sorted(package.get("track_sections", {}).items()):
        lines.extend(["", f"## {track_name}", ""])
        lines.extend(_render_table(rows))
    if package.get("scaling_sections"):
        lines.extend(["", "## Scaling Sections", ""])
        for section in package["scaling_sections"]:
            lines.append(f"- `{section['experiment_name']}` / `{section['track_name']}`")
            for series in section.get("scaling_summary", {}).get("series", []):
                scale_parts = ", ".join(
                    f"{item['node_scale']}=>{item['quality_mean']:.4f}"
                    for item in series.get("scales", [])
                )
                lines.append(f"  - `{series['method_name']}` / `{series['topology_direction_mode']}`: {scale_parts}")
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- reproduction matrix 只允许 track 内比较，不生成跨 graph/table/topology 的单一总榜。",
            "- scaling 条目用于趋势解释，不会被压进 canonical 排名表。",
        ]
    )
    return "\n".join(lines) + "\n"


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "family",
        "experiment_name",
        "track_name",
        "entry_role",
        "primary_method_name",
        "primary_metric_label",
        "primary_metric_value",
        "total_tokens_mean",
        "calls_per_question_mean",
    ]
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display(row.get(header)) for header in headers) + " |")
    return lines


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _resolve_state_path(path_or_root: str | Path) -> Path:
    path = Path(path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path
