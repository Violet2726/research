"""为 reproduction matrix 生成横向景观视图。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from research_experiments.matrix.reproduction_analysis import render_reproduction_analysis
from research_experiments.workspace.layout import default_reports_root


def render_reproduction_landscape(
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
    payload = build_reproduction_landscape_payload(analysis)

    json_path = root / "reproduction_landscape.json"
    markdown_path = root / "reproduction_landscape.md"
    published = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("reproduction_matrix")) / f"{state_path.parent.name}-reproduction_landscape.md"
    )

    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_reproduction_landscape_markdown(payload)
    markdown_path.write_text(markdown, encoding="utf-8")
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(markdown, encoding="utf-8")
    return {
        "json_path": json_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
        "published_path": published.as_posix(),
    }


def build_reproduction_landscape_payload(analysis: dict[str, Any]) -> dict[str, Any]:
    entries = analysis.get("entries", [])
    track_boards: dict[str, list[dict[str, Any]]] = defaultdict(list)
    family_rollup: list[dict[str, Any]] = []
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for entry in entries:
        if entry.get("overall_row"):
            track_boards[str(entry.get("track_name") or "other")].append(entry["overall_row"])
        by_family[str(entry.get("family") or "unknown")].append(entry)
    for rows in track_boards.values():
        rows.sort(key=lambda row: (str(row.get("entry_role") or ""), -(row.get("primary_metric_value") or 0.0), str(row.get("experiment_name") or "")))
    for family, rows in sorted(by_family.items()):
        tracks = sorted({str(row.get("track_name") or "") for row in rows})
        roles = sorted({str(row.get("entry_role") or "") for row in rows})
        experiments = sorted({str(row.get("experiment_name") or "") for row in rows})
        family_rollup.append(
            {
                "family": family,
                "row_count": len(rows),
                "tracks": ", ".join(tracks),
                "entry_roles": ", ".join(roles),
                "experiments": ", ".join(experiments),
            }
        )
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_id": analysis.get("matrix_id"),
        "phase_name": analysis.get("phase_name"),
        "model_ref": analysis.get("model_ref"),
        "counts": analysis.get("counts", {}),
        "track_boards": dict(track_boards),
        "family_rollup": family_rollup,
    }


def render_reproduction_landscape_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reproduction Landscape",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- matrix_id: `{payload.get('matrix_id')}`",
        f"- phase_name: `{payload.get('phase_name')}`",
        f"- model_ref: `{payload.get('model_ref')}`",
        "",
        "## 解释边界",
        "",
        "- 这里只做 track 内比较，不生成跨 track 的统一总榜。",
        "- scaling 条目只会出现在各自 track 的辅助位置，不会挤进 canonical 排名。",
    ]
    for track_name, rows in sorted(payload.get("track_boards", {}).items()):
        lines.extend(["", f"## {track_name}", ""])
        lines.extend(_render_table(rows))
    lines.extend(["", "## Family Rollup", ""])
    lines.extend(_render_rollup_table(payload.get("family_rollup", [])))
    return "\n".join(lines) + "\n"


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "family",
        "experiment_name",
        "entry_role",
        "primary_method_name",
        "primary_metric_label",
        "primary_metric_value",
        "total_tokens_mean",
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


def _render_rollup_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = ["family", "row_count", "tracks", "entry_roles", "experiments"]
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
