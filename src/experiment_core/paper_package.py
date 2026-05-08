"""Render paper-facing reports and lightweight figures for faithful runs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
import math
from statistics import mean
from typing import Any

from experiment_core.matrix_specs import (
    EVIDENCE_DIAGNOSTIC,
    EVIDENCE_HEADLINE,
    EVIDENCE_REFERENCE,
    EVIDENCE_SUPPORTING,
    TRACK_SAME_CONTEXT,
    TRACK_SPLIT_CONTEXT,
    get_experiment_matrix_spec,
)
from experiment_core.paper_statistics import render_paper_statistics


PREDICTION_FILE_CANDIDATES = (
    "policy_predictions.jsonl",
    "final_predictions.jsonl",
    "predictions.jsonl",
)

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

TRACK_COLORS = {
    TRACK_SAME_CONTEXT: COLOR_BLUE,
    TRACK_SPLIT_CONTEXT: COLOR_ORANGE,
    "other": COLOR_GRAY,
}

FIGURE_LABEL_OVERRIDES = {
    "aggregation_auditing_ablation_v1": "SPARC agg-audit",
    "auditing_ablation_v1": "SPARC audit",
    "content_ablation_v1": "SPARC content",
    "cue_v1": "CUE",
    "dala_lite_same_context_v1": "DALA same",
    "dala_lite_split_context_v1": "DALA split",
    "free_mad_lite_v1": "Free-MAD-lite",
    "hotpotqa_split_main": "Hotpot split",
    "multi_agent_main": "Vanilla MAD",
    "robustness": "Robustness",
    "sid_lite_v1": "SID-lite",
    "sparc_v1_smoke": "SPARC",
    "trigger_early_exit_v1": "Trigger early-exit",
    "trigger_voc_v2": "Trigger VOC",
}


def render_paper_package(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
    figures_root: str | Path | None = None,
) -> dict[str, str]:
    """Render a paper package from an existing matrix run."""
    state_path = _resolve_state_path(state_path_or_root)
    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)

    analysis_path = root / "faithful_analysis.json"
    if not analysis_path.exists():
        from experiment_core.faithful_analysis import render_faithful_analysis

        render_faithful_analysis(state_path)
    statistics_path = root / "paper_statistics.json"
    if not statistics_path.exists():
        render_paper_statistics(state_path, output_root=root)

    state_payload = _load_json(state_path)
    analysis = _load_json(analysis_path)
    statistics = _load_json(statistics_path)
    package = build_paper_package_payload(state_payload, analysis, statistics)

    run_id = state_path.parent.name
    figure_dir = Path(figures_root) if figures_root is not None else Path("reports") / "figures" / run_id
    figure_dir.mkdir(parents=True, exist_ok=True)
    figure_paths = _render_figures(package, figure_dir)
    package["figure_paths"] = {name: path.as_posix() for name, path in figure_paths.items()}

    package_json = root / "paper_package.json"
    package_md = root / "paper_package.md"
    published = Path(published_path) if published_path is not None else Path("reports") / "summary" / f"{run_id}-paper_package.md"
    markdown = render_paper_package_markdown(package)
    package_json.write_text(json.dumps(package, ensure_ascii=False, indent=2), encoding="utf-8")
    package_md.write_text(markdown, encoding="utf-8")
    published.parent.mkdir(parents=True, exist_ok=True)
    published.write_text(markdown, encoding="utf-8")
    return {
        "package_json": package_json.as_posix(),
        "package_markdown": package_md.as_posix(),
        "published_path": published.as_posix(),
        "figures_root": figure_dir.as_posix(),
    }


def build_paper_package_payload(
    state_payload: dict[str, Any],
    analysis: dict[str, Any],
    statistics: dict[str, Any],
) -> dict[str, Any]:
    """Convert matrix artifacts into stable paper-facing sections."""
    rows = [row for row in analysis.get("combined_overall", []) if isinstance(row, dict)]
    sections = {
        "same_context_main_table": _rows_by_tier(rows, EVIDENCE_HEADLINE, TRACK_SAME_CONTEXT),
        "split_context_main_table": _rows_by_tier(rows, EVIDENCE_HEADLINE, TRACK_SPLIT_CONTEXT),
        "supporting_evidence_table": _rows_by_tier(rows, EVIDENCE_SUPPORTING, None),
        "diagnostic_evidence_table": _rows_by_tier(rows, EVIDENCE_DIAGNOSTIC, None),
        "reference_table": _rows_by_tier(rows, EVIDENCE_REFERENCE, None),
    }
    state_entries = [
        entry
        for entry in state_payload.get("semantic_entries", [])
        if entry.get("status") == "completed" and entry.get("run_dir")
    ]
    budget_references = _build_budget_matched_single_agent_references(state_entries, rows)
    helpful_harmful = _build_helpful_harmful_breakdown(state_entries)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "phase_name": state_payload.get("overrides", {}).get("phase_name"),
        "model_ref": state_payload.get("overrides", {}).get("model_ref"),
        "counts": state_payload.get("counts", {}),
        "sections": sections,
        "statistics": statistics,
        "budget_matched_single_agent_references": budget_references,
        "helpful_harmful_communication": helpful_harmful,
    }


def render_paper_package_markdown(package: dict[str, Any]) -> str:
    """Render the package payload as a compact paper-oriented Markdown report."""
    lines = [
        "# Paper Package",
        "",
        f"- generated_at: `{package.get('generated_at')}`",
        f"- phase_name: `{package.get('phase_name')}`",
        f"- model_ref: `{package.get('model_ref')}`",
        f"- counts: `{json.dumps(package.get('counts', {}), ensure_ascii=False)}`",
        "",
        "## Same-Context Main Table",
        "",
    ]
    lines.extend(_render_result_table(package["sections"]["same_context_main_table"]))
    lines.extend(["", "## Split-Context Main Table", ""])
    lines.extend(_render_result_table(package["sections"]["split_context_main_table"]))
    lines.extend(["", "## Supporting Evidence", ""])
    lines.extend(_render_result_table(package["sections"]["supporting_evidence_table"]))
    lines.extend(["", "## Diagnostic Evidence", ""])
    lines.extend(_render_result_table(package["sections"]["diagnostic_evidence_table"]))
    lines.extend(["", "## Equal-Budget Single-Agent References", ""])
    lines.extend(_render_budget_baseline_table(package.get("budget_matched_single_agent_references", [])))
    lines.extend(["", "## Fixed Statistical Comparisons", ""])
    lines.extend(_render_statistics_table(package.get("statistics", {}).get("comparisons", []), package.get("statistics", {})))
    lines.extend(["", "## Helpful / Harmful Communication", ""])
    lines.extend(_render_helpful_table(package.get("helpful_harmful_communication", [])))
    figure_paths = package.get("figure_paths", {})
    if figure_paths:
        lines.extend(["", "## Figures", ""])
        for name, path in sorted(figure_paths.items()):
            lines.append(f"- `{name}`: `{path}`")
    lines.extend(
        [
            "",
            "## Interpretation Guardrails",
            "",
            "- Headline tables are the only rows intended for main-text claims.",
            "- Supporting and diagnostic rows explain mechanisms, limits, and ablations.",
            "- Equal-budget single-agent rows are evaluation controls, not new method steps.",
            "- If confirmatory results weaken a headline claim, demote the claim rather than changing the method graph.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rows_by_tier(rows: list[dict[str, Any]], evidence_tier: str, track: str | None) -> list[dict[str, Any]]:
    selected = [
        row
        for row in rows
        if row.get("evidence_tier") == evidence_tier and (track is None or row.get("evaluation_track") == track)
    ]
    return sorted(selected, key=lambda row: (str(row.get("family")), str(row.get("experiment_name"))))


def _build_budget_matched_single_agent_references(
    entries: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    single_agent_rows = _collect_single_agent_summary_rows(entries)
    headline_rows = [row for row in overall_rows if row.get("evidence_tier") == EVIDENCE_HEADLINE]
    baselines: list[dict[str, Any]] = []
    for row in headline_rows:
        target_tokens = _as_float(row.get("total_tokens_mean"))
        target_calls = _as_float(row.get("calls_per_question_mean"))
        for baseline_name, method_name in (
            ("budget_matched_long_cot", "cot_1"),
            ("budget_matched_sc", "sc_5"),
        ):
            reference = _nearest_single_agent_row(single_agent_rows, method_name=method_name, target_tokens=target_tokens)
            baselines.append(
                _baseline_record(
                    source_row=row,
                    baseline_name=baseline_name,
                    reference=reference,
                    target_calls=target_calls,
                    target_tokens=target_tokens,
                )
            )
        if row.get("evaluation_track") == TRACK_SPLIT_CONTEXT and row.get("full_context_reference"):
            baselines.append(
                {
                    "experiment_name": row.get("experiment_name"),
                    "baseline_name": "budget_matched_full_context_single",
                    "matched_budget_source": row.get("full_context_reference"),
                    "matched_calls_per_q": None,
                    "matched_total_tokens_mean": None,
                    "score": row.get("full_context_score"),
                    "target_calls_per_q": target_calls,
                    "target_total_tokens_mean": target_tokens,
                    "budget_match_status": "same_experiment_full_context_reference",
                }
            )
    return baselines


def _baseline_record(
    *,
    source_row: dict[str, Any],
    baseline_name: str,
    reference: dict[str, Any] | None,
    target_calls: float,
    target_tokens: float,
) -> dict[str, Any]:
    if reference is None:
        return {
            "experiment_name": source_row.get("experiment_name"),
            "baseline_name": baseline_name,
            "matched_budget_source": None,
            "matched_calls_per_q": None,
            "matched_total_tokens_mean": None,
            "score": None,
            "target_calls_per_q": target_calls,
            "target_total_tokens_mean": target_tokens,
            "budget_match_status": "missing_reference_run",
        }
    matched_tokens = _as_float(reference.get("total_tokens_mean"))
    return {
        "experiment_name": source_row.get("experiment_name"),
        "baseline_name": baseline_name,
        "matched_budget_source": f"{reference.get('experiment_name')}/{reference.get('method_name')}/{reference.get('dataset')}",
        "matched_calls_per_q": _as_float(reference.get("calls_per_question_mean")),
        "matched_total_tokens_mean": matched_tokens,
        "score": _as_float(reference.get("accuracy_mean")),
        "target_calls_per_q": target_calls,
        "target_total_tokens_mean": target_tokens,
        "token_ratio_to_target": None if target_tokens <= 0 else round(matched_tokens / target_tokens, 6),
        "budget_match_status": "available_proxy_not_exact_budget",
    }


def _collect_single_agent_summary_rows(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        if entry.get("family") != "single_agent":
            continue
        run_dir = Path(str(entry["run_dir"]))
        payload = _safe_load_json(run_dir / "metrics.json")
        for row in payload.get("summary", []) if isinstance(payload, dict) else []:
            if isinstance(row, dict):
                enriched = dict(row)
                enriched["experiment_name"] = entry.get("experiment_name")
                rows.append(enriched)
    return rows + _synthesize_single_agent_overall_rows(rows)


def _nearest_single_agent_row(
    rows: list[dict[str, Any]],
    *,
    method_name: str,
    target_tokens: float,
) -> dict[str, Any] | None:
    candidates = [row for row in rows if row.get("method_name") == method_name and row.get("dataset") == "overall"]
    if not candidates:
        return None
    return min(
        candidates,
        key=lambda row: (
            _single_agent_reference_rank(str(row.get("experiment_name") or "")),
            abs(_as_float(row.get("total_tokens_mean")) - target_tokens),
        ),
    )


def _single_agent_reference_rank(experiment_name: str) -> int:
    if experiment_name == "main_baselines":
        return 0
    if experiment_name == "main_table_same_context":
        return 1
    return 2


def _synthesize_single_agent_overall_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    direct_keys = {
        (row.get("experiment_name"), row.get("method_name"))
        for row in rows
        if row.get("dataset") == "overall"
    }
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("dataset") == "overall":
            continue
        key = (row.get("experiment_name"), row.get("method_name"))
        if key in direct_keys:
            continue
        grouped.setdefault(key, []).append(row)

    synthesized: list[dict[str, Any]] = []
    for (experiment_name, method_name), items in grouped.items():
        if not items:
            continue
        synthesized.append(
            {
                "experiment_name": experiment_name,
                "method_name": method_name,
                "dataset": "overall",
                "accuracy_mean": round(mean(_as_float(item.get("accuracy_mean")) for item in items), 6),
                "total_tokens_mean": round(mean(_as_float(item.get("total_tokens_mean")) for item in items), 6),
                "calls_per_question_mean": round(mean(_as_float(item.get("calls_per_question_mean")) for item in items), 6),
            }
        )
    return synthesized


def _build_helpful_harmful_breakdown(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for entry in entries:
        spec = get_experiment_matrix_spec(str(entry["config_path"]))
        if spec.evidence_tier != EVIDENCE_HEADLINE:
            continue
        prediction_rows = [
            row
            for row in _load_prediction_rows(Path(str(entry["run_dir"])))
            if row.get("method_name") == spec.primary_method_name
        ]
        if not prediction_rows:
            continue
        helpful = 0
        harmful = 0
        neutral = 0
        for row in prediction_rows:
            if bool(row.get("corrected_by_method")):
                helpful += 1
                continue
            if bool(row.get("harmed_by_method")):
                harmful += 1
                continue
            stage_a = _as_optional_float(row.get("stage_a_score"))
            final = _as_optional_float(row.get("score"))
            if stage_a is None or final is None:
                neutral += 1
            elif final > stage_a:
                helpful += 1
            elif final < stage_a:
                harmful += 1
            else:
                neutral += 1
        rows.append(
            {
                "experiment_name": entry.get("experiment_name"),
                "method_name": spec.primary_method_name,
                "sample_method_rows": len(prediction_rows),
                "helpful": helpful,
                "harmful": harmful,
                "neutral": neutral,
                "helpful_rate": round(helpful / len(prediction_rows), 6),
                "harmful_rate": round(harmful / len(prediction_rows), 6),
            }
        )
    return rows


def _render_figures(package: dict[str, Any], figure_dir: Path) -> dict[str, Path]:
    sections = package.get("sections", {})
    figure_specs = {
        "budget_frontier_same_context": {
            "points": _frontier_points(sections.get("same_context_main_table", [])),
            "renderer": _render_budget_frontier_svg,
        },
        "budget_frontier_split_context": {
            "points": _frontier_points(sections.get("split_context_main_table", [])),
            "renderer": _render_budget_frontier_svg,
        },
        "trigger_utility": {
            "points": _trigger_utility_points(sections.get("same_context_main_table", [])),
            "renderer": _render_trigger_utility_svg,
        },
        "stage_ceiling_gap": {
            "points": _stage_gap_points(sections),
            "renderer": _render_stage_gap_svg,
        },
        "helpful_harmful_comm": {
            "points": _helpful_points(package.get("helpful_harmful_communication", [])),
            "renderer": _render_helpful_harmful_svg,
        },
    }
    paths: dict[str, Path] = {}
    for name, spec in figure_specs.items():
        points = spec["points"]
        path = figure_dir / f"{name}.svg"
        path.write_text(spec["renderer"](name, points), encoding="utf-8")
        paths[name] = path
        data_path = figure_dir / f"{name}.csv"
        data_path.write_text(_render_points_csv(points), encoding="utf-8")
        paths[f"{name}_data"] = data_path
    return paths


def _frontier_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        score = _as_optional_float(row.get("faithful_score"))
        if score is None:
            continue
        x = _as_optional_float(row.get("token_ratio_vs_full_comm"))
        if x is None:
            x = _as_optional_float(row.get("token_ratio_vs_best_no_comm"))
        x = 1.0 if x is None else x
        points.append(
            {
                "label": str(row.get("experiment_name")),
                "short_label": _short_figure_label(row.get("experiment_name")),
                "family": str(row.get("family") or ""),
                "track": str(row.get("evaluation_track") or ""),
                "method_name": str(row.get("primary_method_name") or ""),
                "x": x,
                "y": score,
                "value": score,
            }
        )
    return sorted(points, key=lambda point: (float(point["x"]), -float(point["y"]), str(point["label"])))


def _trigger_utility_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        experiment_name = str(row.get("experiment_name") or "")
        if "trigger" not in experiment_name:
            continue
        delta = _as_optional_float(row.get("delta_vs_best_no_comm"))
        if delta is None:
            continue
        token_ratio = _as_optional_float(row.get("token_ratio_vs_full_comm"))
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "x": 1.0 if token_ratio is None else token_ratio,
                "y": delta,
                "value": delta,
                "direction": "improved" if delta >= 0 else "regressed",
            }
        )
    return sorted(points, key=lambda point: (float(point["x"]), -float(point["y"]), str(point["label"])))


def _stage_gap_points(sections: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("same_context_main_table", "split_context_main_table", "supporting_evidence_table", "diagnostic_evidence_table"):
        rows.extend(sections.get(key, []))
    points: list[dict[str, Any]] = []
    for row in rows:
        gap = _as_optional_float(row.get("stage_ceiling_gap"))
        if gap is None:
            continue
        experiment_name = str(row.get("experiment_name") or "")
        track = str(row.get("evaluation_track") or "other")
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "track": track,
                "track_label": _track_display_label(track),
                "x": gap,
                "y": gap,
                "value": gap,
                "family": str(row.get("family") or ""),
            }
        )
    return sorted(points, key=lambda point: (float(point["value"]), str(point["track"]), str(point["label"])))


def _helpful_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        helpful_rate = _as_optional_float(row.get("helpful_rate"))
        harmful_rate = _as_optional_float(row.get("harmful_rate"))
        experiment_name = str(row.get("experiment_name") or "")
        points.append(
            {
                "label": experiment_name,
                "short_label": _short_figure_label(experiment_name),
                "method_name": str(row.get("method_name") or ""),
                "sample_method_rows": int(row.get("sample_method_rows") or 0),
                "helpful_rate": 0.0 if helpful_rate is None else helpful_rate,
                "harmful_rate": 0.0 if harmful_rate is None else harmful_rate,
                "net_gain": (0.0 if helpful_rate is None else helpful_rate) - (0.0 if harmful_rate is None else harmful_rate),
                "value": 0.0 if helpful_rate is None else helpful_rate,
            }
        )
    return sorted(points, key=lambda point: (-float(point["net_gain"]), str(point["label"])))


def _render_points_csv(points: list[dict[str, Any]]) -> str:
    headers = _points_csv_headers(points)
    lines = [",".join(headers)]
    for point in points:
        lines.append(
            ",".join(
                [
                    _csv_cell(str(point.get(header, "")))
                    for header in headers
                ]
            )
        )
    return "\n".join(lines) + "\n"


def _render_budget_frontier_svg(name: str, points: list[dict[str, Any]]) -> str:
    title = "Budget frontier"
    subtitle = _figure_subtitle(
        name=name,
        default="Faithful score versus token ratio relative to full communication.",
    )
    if not points:
        return _render_empty_svg(title, subtitle)

    width = 960
    height = 560
    left = 88
    right = 42
    top = 78
    bottom = 86
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["x"]) for point in points]
    x_min, x_max = _expand_domain(x_values, include_values=[1.0], pad_fraction=0.08)
    y_min, y_max = 0.0, 1.0
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    y_ticks = [0.0, 0.25, 0.5, 0.75, 1.0]
    x_min, x_max = x_ticks[0], x_ticks[-1]

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, subtitle))
    lines.extend(_svg_axes_and_grid(left, top, plot_width, plot_height, x_ticks, y_ticks, x_min, x_max, y_min, y_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label="Token ratio vs full communication (lower is cheaper)",
            y_label="Faithful score",
        )
    )

    reference_x = _scale_linear(1.0, x_min, x_max, left, left + plot_width)
    lines.append(
        f'<line x1="{reference_x:.2f}" y1="{top}" x2="{reference_x:.2f}" y2="{top + plot_height}" '
        f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
    )
    lines.append(
        f'<text x="{reference_x + 6:.2f}" y="{top + 16}" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">full-comm baseline</text>'
    )

    for index, point in enumerate(points):
        cx = _scale_linear(float(point["x"]), x_min, x_max, left, left + plot_width)
        cy = _scale_linear(float(point["y"]), y_min, y_max, top + plot_height, top)
        label_dx = 10 if index % 2 == 0 else -12
        anchor = "start" if label_dx > 0 else "end"
        label_y = cy - 8 if index % 3 != 1 else cy + 18
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="5.5" fill="{COLOR_BLUE}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{cx + label_dx:.2f}" y="{label_y:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" text-anchor="{anchor}" fill="{COLOR_TEXT}">{escape(str(point["short_label"]))}</text>'
        )
    lines.extend(_svg_note_block(height, "x=1.0 denotes full communication; all scores use the faithful analysis output."))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_trigger_utility_svg(name: str, points: list[dict[str, Any]]) -> str:
    title = "Trigger utility"
    subtitle = _figure_subtitle(
        name=name,
        default="Utility of trigger methods relative to the best no-communication baseline.",
    )
    if not points:
        return _render_empty_svg(title, subtitle)

    width = 960
    height = 560
    left = 88
    right = 42
    top = 78
    bottom = 86
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["x"]) for point in points]
    y_values = [float(point["y"]) for point in points]
    x_min, x_max = _expand_domain(x_values, include_values=[1.0], pad_fraction=0.08)
    y_min, y_max = _expand_domain(y_values, include_values=[0.0], pad_fraction=0.15)
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    y_ticks = _nice_ticks(y_min, y_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    y_min, y_max = y_ticks[0], y_ticks[-1]

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, subtitle))
    lines.extend(_svg_axes_and_grid(left, top, plot_width, plot_height, x_ticks, y_ticks, x_min, x_max, y_min, y_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label="Token ratio vs full communication (lower is cheaper)",
            y_label="Delta vs best no-communication baseline",
        )
    )

    reference_x = _scale_linear(1.0, x_min, x_max, left, left + plot_width)
    reference_y = _scale_linear(0.0, y_min, y_max, top + plot_height, top)
    lines.append(
        f'<line x1="{reference_x:.2f}" y1="{top}" x2="{reference_x:.2f}" y2="{top + plot_height}" '
        f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
    )
    lines.append(
        f'<line x1="{left}" y1="{reference_y:.2f}" x2="{left + plot_width}" y2="{reference_y:.2f}" '
        f'stroke="{COLOR_MUTED}" stroke-width="1.2" stroke-dasharray="6 4"/>'
    )

    for point in points:
        cx = _scale_linear(float(point["x"]), x_min, x_max, left, left + plot_width)
        cy = _scale_linear(float(point["y"]), y_min, y_max, top + plot_height, top)
        color = COLOR_GREEN if float(point["y"]) >= 0 else COLOR_RED
        label_y = cy - 10 if float(point["y"]) >= 0 else cy + 18
        lines.append(
            f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="6" fill="{color}" stroke="#ffffff" stroke-width="1.5"/>'
        )
        lines.append(
            f'<text x="{cx + 8:.2f}" y="{label_y:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(str(point["short_label"]))}</text>'
        )
    lines.extend(_svg_note_block(height, "Positive y indicates improvement over the best no-communication control."))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_stage_gap_svg(name: str, points: list[dict[str, Any]]) -> str:
    title = "Distance to stage ceiling"
    subtitle = _figure_subtitle(
        name=name,
        default="Lower values indicate less remaining room between faithful score and the stage ceiling.",
    )
    if not points:
        return _render_empty_svg(title, subtitle)

    width = 1040
    row_height = 28
    top = 86
    bottom = 72
    left = 290
    right = 36
    height = max(420, top + bottom + row_height * len(points))
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_values = [float(point["value"]) for point in points]
    x_min, x_max = 0.0, _expand_domain(x_values, include_values=[0.0], pad_fraction=0.10)[1]
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    band_step = plot_height / max(1, len(points))
    bar_height = max(12.0, band_step * 0.56)

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, subtitle))
    lines.extend(_svg_vertical_grid(left, top, plot_width, plot_height, x_ticks, x_min, x_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label="Gap to stage ceiling (absolute score; lower is better)",
            y_label=None,
        )
    )
    lines.extend(_svg_track_legend(width - right - 220, 44))

    for index, point in enumerate(points):
        value = float(point["value"])
        y = top + index * band_step + (band_step - bar_height) / 2
        x_end = _scale_linear(value, x_min, x_max, left, left + plot_width)
        color = TRACK_COLORS.get(str(point.get("track")), TRACK_COLORS["other"])
        label = f"{point['short_label']} [{point['track_label']}]"
        lines.append(
            f'<text x="{left - 12}" y="{y + bar_height / 2 + 4:.2f}" text-anchor="end" '
            f'font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left:.2f}" y="{y:.2f}" width="{max(x_end - left, 0.8):.2f}" height="{bar_height:.2f}" '
            f'fill="{color}" opacity="0.92"/>'
        )
        lines.append(
            f'<text x="{x_end + 8:.2f}" y="{y + bar_height / 2 + 4:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{value:.3f}</text>'
        )
    lines.extend(_svg_note_block(height, "Track colors separate same-context, split-context, and other supporting rows."))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_helpful_harmful_svg(name: str, points: list[dict[str, Any]]) -> str:
    title = "Helpful versus harmful communication"
    subtitle = _figure_subtitle(
        name=name,
        default="Rates are measured over samples that use communication within each experiment.",
    )
    if not points:
        return _render_empty_svg(title, subtitle)

    width = 1040
    group_height = 36
    top = 86
    bottom = 78
    left = 260
    right = 40
    height = max(420, top + bottom + group_height * len(points))
    plot_width = width - left - right
    plot_height = height - top - bottom

    max_rate = max(
        max(float(point["helpful_rate"]) for point in points),
        max(float(point["harmful_rate"]) for point in points),
        0.01,
    )
    x_min, x_max = 0.0, _expand_domain([max_rate], include_values=[0.0], pad_fraction=0.20)[1]
    x_ticks = _nice_ticks(x_min, x_max, target_ticks=5)
    x_min, x_max = x_ticks[0], x_ticks[-1]
    band_step = plot_height / max(1, len(points))
    bar_height = max(8.0, band_step * 0.26)

    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, subtitle))
    lines.extend(_svg_vertical_grid(left, top, plot_width, plot_height, x_ticks, x_min, x_max))
    lines.extend(
        _svg_axis_labels(
            left=left,
            top=top,
            plot_width=plot_width,
            plot_height=plot_height,
            x_label="Rate over communication-using samples",
            y_label=None,
        )
    )
    lines.extend(
        _svg_series_legend(
            x=width - right - 180,
            y=44,
            items=[("Helpful", COLOR_GREEN), ("Harmful", COLOR_RED)],
        )
    )

    for index, point in enumerate(points):
        helpful = float(point["helpful_rate"])
        harmful = float(point["harmful_rate"])
        group_y = top + index * band_step
        helpful_y = group_y + band_step * 0.18
        harmful_y = group_y + band_step * 0.56
        helpful_x = _scale_linear(helpful, x_min, x_max, left, left + plot_width)
        harmful_x = _scale_linear(harmful, x_min, x_max, left, left + plot_width)
        label = str(point["short_label"])
        lines.append(
            f'<text x="{left - 12}" y="{group_y + band_step * 0.50:.2f}" text-anchor="end" '
            f'font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
        lines.append(
            f'<rect x="{left:.2f}" y="{helpful_y:.2f}" width="{max(helpful_x - left, 0.8):.2f}" height="{bar_height:.2f}" '
            f'fill="{COLOR_GREEN}" opacity="0.92"/>'
        )
        lines.append(
            f'<rect x="{left:.2f}" y="{harmful_y:.2f}" width="{max(harmful_x - left, 0.8):.2f}" height="{bar_height:.2f}" '
            f'fill="{COLOR_RED}" opacity="0.88"/>'
        )
        lines.append(
            f'<text x="{helpful_x + 8:.2f}" y="{helpful_y + bar_height - 1:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{helpful:.3f}</text>'
        )
        lines.append(
            f'<text x="{harmful_x + 8:.2f}" y="{harmful_y + bar_height - 1:.2f}" font-size="11" '
            f'font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{harmful:.3f}</text>'
        )
    lines.extend(_svg_note_block(height, "Helpful and harmful rates are shown separately to avoid visual cancellation."))
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_result_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "experiment_name",
        "primary_method_name",
        "faithful_score",
        "delta_vs_best_no_comm",
        "delta_vs_full_comm",
        "delta_vs_full_context",
        "token_ratio_vs_full_comm",
        "calls_per_question_mean",
    ]
    return _render_table(rows, headers)


def _render_budget_baseline_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "experiment_name",
        "baseline_name",
        "matched_budget_source",
        "matched_calls_per_q",
        "matched_total_tokens_mean",
        "score",
        "budget_match_status",
    ]
    return _render_table(rows, headers)


def _render_statistics_table(comparisons: list[dict[str, Any]], statistics: dict[str, Any]) -> list[str]:
    rows: list[dict[str, Any]] = []
    bootstrap = statistics.get("bootstrap_ci", {})
    paired = statistics.get("paired_win_loss", {})
    mcnemar = statistics.get("mcnemar_tests", {})
    for comparison in comparisons:
        comparison_id = comparison.get("comparison_id")
        ci = bootstrap.get(comparison_id, {})
        pair = paired.get(comparison_id, {})
        test = mcnemar.get(comparison_id, {})
        rows.append(
            {
                "comparison_id": comparison_id,
                "status": comparison.get("status"),
                "paired_n": comparison.get("paired_n"),
                "mean_delta": comparison.get("mean_delta"),
                "delta_ci95": _format_ci(ci.get("delta_ci95")),
                "wins": pair.get("wins"),
                "losses": pair.get("losses"),
                "ties": pair.get("ties"),
                "mcnemar_p": test.get("p_value_chi_square_cc"),
            }
        )
    return _render_table(rows, ["comparison_id", "status", "paired_n", "mean_delta", "delta_ci95", "wins", "losses", "ties", "mcnemar_p"])


def _render_helpful_table(rows: list[dict[str, Any]]) -> list[str]:
    return _render_table(rows, ["experiment_name", "method_name", "sample_method_rows", "helpful", "harmful", "neutral", "helpful_rate", "harmful_rate"])


def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display(row.get(header)) for header in headers) + " |")
    return lines


def _format_ci(value: Any) -> str:
    if not isinstance(value, dict):
        return ""
    return f"[{_display(value.get('low'))}, {_display(value.get('high'))}]"


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    for filename in PREDICTION_FILE_CANDIDATES:
        path = run_dir / filename
        if path.exists():
            return _load_jsonl(path)
    return []


def _resolve_state_path(state_path_or_root: str | Path) -> Path:
    path = Path(state_path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return _load_json(path)
    except Exception:
        return {}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _csv_cell(value: str) -> str:
    if any(char in value for char in [",", "\"", "\n"]):
        return '"' + value.replace('"', '""') + '"'
    return value


def _points_csv_headers(points: list[dict[str, Any]]) -> list[str]:
    preferred = [
        "label",
        "short_label",
        "family",
        "track",
        "track_label",
        "method_name",
        "direction",
        "x",
        "y",
        "value",
        "helpful_rate",
        "harmful_rate",
        "net_gain",
        "sample_method_rows",
    ]
    seen = {key for point in points for key in point}
    headers = [key for key in preferred if key in seen]
    headers.extend(sorted(key for key in seen if key not in headers))
    return headers or ["label", "x", "y", "value"]


def _figure_subtitle(name: str, default: str) -> str:
    if name == "budget_frontier_same_context":
        return "Same-context headline methods only. Lower x is cheaper; higher y is better."
    if name == "budget_frontier_split_context":
        return "Split-context headline methods only. Lower x is cheaper; higher y is better."
    if name == "trigger_utility":
        return "Trigger methods in same-context settings. Positive y indicates improvement."
    if name == "stage_ceiling_gap":
        return "All headline and supporting rows sorted by gap to the stage ceiling."
    if name == "helpful_harmful_comm":
        return "Communication outcome breakdown for experiments with message exchange."
    return default


def _render_empty_svg(title: str, subtitle: str) -> str:
    width = 960
    height = 220
    lines = _svg_canvas(width, height)
    lines.extend(_svg_title_block(title, subtitle))
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


def _svg_title_block(title: str, subtitle: str) -> list[str]:
    return [
        f'<text x="48" y="34" font-size="20" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}" font-weight="600">{escape(title)}</text>',
        f'<text x="48" y="56" font-size="12" font-family="{FONT_FAMILY}" fill="{COLOR_MUTED}">{escape(subtitle)}</text>',
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


def _svg_track_legend(x: int, y: int) -> list[str]:
    items = [
        (_track_display_label(TRACK_SAME_CONTEXT), TRACK_COLORS[TRACK_SAME_CONTEXT]),
        (_track_display_label(TRACK_SPLIT_CONTEXT), TRACK_COLORS[TRACK_SPLIT_CONTEXT]),
        ("other", TRACK_COLORS["other"]),
    ]
    return _svg_series_legend(x, y, items)


def _svg_series_legend(x: int, y: int, items: list[tuple[str, str]]) -> list[str]:
    lines: list[str] = []
    for index, (label, color) in enumerate(items):
        y_offset = y + index * 18
        lines.append(f'<rect x="{x}" y="{y_offset - 9}" width="12" height="12" fill="{color}" opacity="0.92"/>')
        lines.append(
            f'<text x="{x + 18}" y="{y_offset + 1}" font-size="11" font-family="{FONT_FAMILY}" fill="{COLOR_TEXT}">{escape(label)}</text>'
        )
    return lines


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


def _short_figure_label(value: Any) -> str:
    label = str(value or "")
    if label in FIGURE_LABEL_OVERRIDES:
        return FIGURE_LABEL_OVERRIDES[label]
    cleaned = label
    for fragment in ("_v1", "_v2", "_smoke", "_main"):
        cleaned = cleaned.replace(fragment, "")
    cleaned = cleaned.replace("same_context", "same")
    cleaned = cleaned.replace("split_context", "split")
    cleaned = cleaned.replace("_", " ").strip()
    return cleaned[:24].strip() or label[:24]


def _track_display_label(track: str) -> str:
    if track == TRACK_SAME_CONTEXT:
        return "same"
    if track == TRACK_SPLIT_CONTEXT:
        return "split"
    return "other"
