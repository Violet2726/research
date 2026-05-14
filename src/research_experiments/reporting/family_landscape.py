"""为 faithful matrix 运行生成 family 横向景观总览。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from research_experiments.workspace.layout import default_reports_root
from research_experiments.matrix.faithful_analysis import render_faithful_analysis
from research_experiments.matrix.matrix_specs import (
    EVIDENCE_DIAGNOSTIC,
    EVIDENCE_HEADLINE,
    EVIDENCE_REFERENCE,
    EVIDENCE_SUPPORTING,
    TRACK_SAME_CONTEXT,
    TRACK_SPLIT_CONTEXT,
)
from research_experiments.reporting.report_views import MatrixAnalysisTableView, load_json_payload


TRACK_ORDER = (TRACK_SAME_CONTEXT, TRACK_SPLIT_CONTEXT)
TIER_ORDER = (
    EVIDENCE_HEADLINE,
    EVIDENCE_SUPPORTING,
    EVIDENCE_DIAGNOSTIC,
    EVIDENCE_REFERENCE,
)
LANDSCAPE_TABLE_HEADERS = [
    "global_rank_by_score",
    "track_rank_by_score",
    "tier_track_rank_by_score",
    "family",
    "experiment_name",
    "evaluation_track",
    "evidence_tier",
    "faithful_score",
    "delta_vs_best_no_comm",
    "delta_vs_full_comm",
    "delta_vs_full_context",
    "total_tokens_mean",
    "communication_tokens_mean",
    "calls_per_question_mean",
    "stage_ceiling_gap",
    "helpful_rate",
    "harmful_rate",
    "acceptance_status",
]


def render_family_landscape(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
) -> dict[str, str]:
    """为既有 matrix 运行写出 family 景观 JSON 与 Markdown 产物。"""

    state_path = _resolve_state_path(state_path_or_root)
    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)

    analysis_path = root / "faithful_analysis.json"
    if not analysis_path.exists():
        render_faithful_analysis(state_path, output_root=root)

    state_payload = load_json_payload(state_path)
    analysis_payload = load_json_payload(analysis_path)
    package_payload = load_json_payload(root / "paper_package.json")
    acceptance_payload = load_json_payload(root / "acceptance_summary.json")
    landscape = build_family_landscape_payload(
        state_payload,
        analysis_payload,
        package_payload=package_payload,
        acceptance_payload=acceptance_payload,
    )

    json_path = root / "family_landscape.json"
    markdown_path = root / "family_landscape.md"
    published_output = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("faithful_matrix")) / f"{state_path.parent.name}-family_landscape.md"
    )

    json_path.write_text(json.dumps(landscape, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_family_landscape_markdown(landscape)
    markdown_path.write_text(markdown, encoding="utf-8")
    published_output.parent.mkdir(parents=True, exist_ok=True)
    published_output.write_text(markdown, encoding="utf-8")
    return {
        "state_path": state_path.as_posix(),
        "analysis_path": analysis_path.as_posix(),
        "json_path": json_path.as_posix(),
        "markdown_path": markdown_path.as_posix(),
        "published_path": published_output.as_posix(),
    }


def build_family_landscape_payload(
    state_payload: dict[str, Any],
    analysis_payload: dict[str, Any],
    *,
    package_payload: dict[str, Any] | None = None,
    acceptance_payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """把 faithful analysis 整理成可扫描的 family 横向景观视图。"""

    package_payload = package_payload or {}
    acceptance_payload = acceptance_payload or {}
    table = MatrixAnalysisTableView.from_analysis_payload(analysis_payload)
    helpful_lookup = _build_helpful_lookup(package_payload)
    acceptance_lookup = _build_acceptance_lookup(acceptance_payload)

    rows = [
        _normalize_landscape_row(row.to_dict(), helpful_lookup, acceptance_lookup)
        for row in table.rows
    ]
    _assign_rank_fields(rows)

    global_total_board = _sorted_board(rows)
    track_boards = {
        track: _sorted_board([row for row in rows if row["evaluation_track"] == track])
        for track in TRACK_ORDER
    }
    tier_track_boards = {
        track: {
            tier: _sorted_board(
                [
                    row
                    for row in rows
                    if row["evaluation_track"] == track and row["evidence_tier"] == tier
                ]
            )
            for tier in TIER_ORDER
        }
        for track in TRACK_ORDER
    }

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase_name": state_payload.get("overrides", {}).get("phase_name"),
        "model_ref": state_payload.get("overrides", {}).get("model_ref"),
        "counts": state_payload.get("counts", {}),
        "global_total_board": global_total_board,
        "track_boards": track_boards,
        "tier_track_boards": tier_track_boards,
        "family_rollup": _build_family_rollup(rows),
    }


def render_family_landscape_markdown(payload: dict[str, Any]) -> str:
    """把 family 景观载荷渲染成便于人工审阅的 Markdown。"""

    lines = [
        "# Family Landscape",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- phase_name: `{payload.get('phase_name')}`",
        f"- model_ref: `{payload.get('model_ref')}`",
        f"- counts: `{json.dumps(payload.get('counts', {}), ensure_ascii=False)}`",
        "",
        "## 解释边界",
        "",
        "- `global_total_board` 是景观视图，不是论文主结论表。",
        "- `accepted` 不等于 `headline`；`headline / supporting / diagnostic / reference` 仍需分层解释。",
        "- same-context 与 split-context 共享一张总览表，但不应被误读成单一研究问题上的统一冠军榜。",
        "",
        "## Global Total Board",
        "",
    ]
    lines.extend(_render_table(payload.get("global_total_board", []), LANDSCAPE_TABLE_HEADERS))

    for track in TRACK_ORDER:
        lines.extend(["", f"## {track}", ""])
        lines.extend(_render_table(payload.get("track_boards", {}).get(track, []), LANDSCAPE_TABLE_HEADERS))
        for tier in TIER_ORDER:
            lines.extend(["", f"### {track} / {tier}", ""])
            lines.extend(
                _render_table(
                    payload.get("tier_track_boards", {}).get(track, {}).get(tier, []),
                    LANDSCAPE_TABLE_HEADERS,
                )
            )

    lines.extend(["", "## Family Rollup", ""])
    lines.extend(
        _render_table(
            payload.get("family_rollup", []),
            [
                "family",
                "row_count",
                "tracks",
                "evidence_tiers",
                "acceptance_statuses",
                "experiments",
            ],
        )
    )
    return "\n".join(lines) + "\n"


def _build_helpful_lookup(package_payload: dict[str, Any]) -> dict[tuple[str, str], dict[str, Any]]:
    helpful_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for row in package_payload.get("helpful_harmful_communication", []):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("experiment_name") or ""),
            str(row.get("method_name") or ""),
        )
        if key == ("", ""):
            continue
        helpful_lookup[key] = row
    return helpful_lookup


def _build_acceptance_lookup(acceptance_payload: dict[str, Any]) -> dict[tuple[str, str, str], dict[str, Any]]:
    acceptance_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in acceptance_payload.get("all_rows", []):
        if not isinstance(row, dict):
            continue
        key = (
            str(row.get("family") or ""),
            str(row.get("experiment_name") or ""),
            str(row.get("primary_method_name") or ""),
        )
        if key == ("", "", ""):
            continue
        acceptance_lookup[key] = row
    return acceptance_lookup


def _normalize_landscape_row(
    row: dict[str, Any],
    helpful_lookup: dict[tuple[str, str], dict[str, Any]],
    acceptance_lookup: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[str, Any]:
    helpful = helpful_lookup.get(
        (
            str(row.get("experiment_name") or ""),
            str(row.get("primary_method_name") or ""),
        ),
        {},
    )
    acceptance = acceptance_lookup.get(
        (
            str(row.get("family") or ""),
            str(row.get("experiment_name") or ""),
            str(row.get("primary_method_name") or ""),
        ),
        {},
    )
    normalized = {
        "global_rank_by_score": None,
        "track_rank_by_score": None,
        "tier_track_rank_by_score": None,
        "family": row.get("family"),
        "experiment_name": row.get("experiment_name"),
        "evaluation_track": row.get("evaluation_track"),
        "evidence_tier": row.get("evidence_tier"),
        "primary_method_name": row.get("primary_method_name"),
        "faithful_score": row.get("faithful_score"),
        "delta_vs_best_no_comm": row.get("delta_vs_best_no_comm"),
        "delta_vs_full_comm": row.get("delta_vs_full_comm"),
        "delta_vs_full_context": row.get("delta_vs_full_context"),
        "total_tokens_mean": row.get("total_tokens_mean"),
        "communication_tokens_mean": row.get("communication_tokens_mean"),
        "calls_per_question_mean": row.get("calls_per_question_mean"),
        "stage_ceiling_gap": row.get("stage_ceiling_gap"),
        "helpful_rate": helpful.get("helpful_rate"),
        "harmful_rate": helpful.get("harmful_rate"),
        "acceptance_status": acceptance.get("status"),
        "acceptance_notes": acceptance.get("notes"),
        "run_dir": row.get("run_dir"),
    }
    return normalized


def _assign_rank_fields(rows: list[dict[str, Any]]) -> None:
    for rank, row in enumerate(_sorted_board(rows), start=1):
        row["global_rank_by_score"] = rank

    grouped_by_track: dict[str, list[dict[str, Any]]] = defaultdict(list)
    grouped_by_track_tier: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        track = str(row.get("evaluation_track") or "")
        tier = str(row.get("evidence_tier") or "")
        grouped_by_track[track].append(row)
        grouped_by_track_tier[(track, tier)].append(row)

    for track_rows in grouped_by_track.values():
        for rank, row in enumerate(_sorted_board(track_rows), start=1):
            row["track_rank_by_score"] = rank

    for tier_rows in grouped_by_track_tier.values():
        for rank, row in enumerate(_sorted_board(tier_rows), start=1):
            row["tier_track_rank_by_score"] = rank


def _build_family_rollup(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("family") or "")].append(row)

    rollup: list[dict[str, Any]] = []
    for family, family_rows in grouped.items():
        ordered_rows = _sorted_board(family_rows)
        tracks = _ordered_unique(row["evaluation_track"] for row in ordered_rows)
        evidence_tiers = _ordered_unique(row["evidence_tier"] for row in ordered_rows)
        acceptance_statuses = _ordered_unique(
            row["acceptance_status"]
            for row in ordered_rows
            if row.get("acceptance_status")
        )
        rollup.append(
            {
                "family": family,
                "row_count": len(ordered_rows),
                "tracks": ", ".join(tracks),
                "evidence_tiers": ", ".join(evidence_tiers),
                "acceptance_statuses": ", ".join(acceptance_statuses),
                "experiments": " | ".join(
                    (
                        f"{row['experiment_name']}"
                        f" ({row['evaluation_track']}/{row['evidence_tier']}, "
                        f"rank #{row['global_rank_by_score']}, score={_format_value(row.get('faithful_score'))})"
                    )
                    for row in ordered_rows
                ),
            }
        )
    return sorted(rollup, key=lambda row: str(row.get("family") or ""))


def _ordered_unique(values: Any) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    sort_lookup = {name: index for index, name in enumerate((*TRACK_ORDER, *TIER_ORDER))}
    for value in sorted({str(item) for item in values if item}, key=lambda item: (sort_lookup.get(item, 999), item)):
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _sorted_board(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=_board_sort_key)


def _board_sort_key(row: dict[str, Any]) -> tuple[Any, ...]:
    score = row.get("faithful_score")
    track = str(row.get("evaluation_track") or "")
    tier = str(row.get("evidence_tier") or "")
    return (
        score is None,
        -float(score or 0.0),
        _order_index(track, TRACK_ORDER),
        _order_index(tier, TIER_ORDER),
        str(row.get("family") or ""),
        str(row.get("experiment_name") or ""),
        str(row.get("primary_method_name") or ""),
    )


def _order_index(value: str, choices: tuple[str, ...]) -> int:
    try:
        return choices.index(value)
    except ValueError:
        return len(choices)


def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_format_value(row.get(header)) for header in headers) + " |")
    return lines


def _format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _resolve_state_path(state_path_or_root: str | Path) -> Path:
    path = Path(state_path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path
