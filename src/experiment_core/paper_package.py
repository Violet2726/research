"""Render paper-facing reports and lightweight figures for faithful runs."""

from __future__ import annotations

from datetime import datetime, timezone
from html import escape
from pathlib import Path
import json
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
        "budget_frontier_same_context": _frontier_points(sections.get("same_context_main_table", [])),
        "budget_frontier_split_context": _frontier_points(sections.get("split_context_main_table", [])),
        "trigger_utility": _trigger_utility_points(sections.get("same_context_main_table", [])),
        "stage_ceiling_gap": _stage_gap_points(sections),
        "helpful_harmful_comm": _helpful_points(package.get("helpful_harmful_communication", [])),
    }
    paths: dict[str, Path] = {}
    for name, points in figure_specs.items():
        path = figure_dir / f"{name}.svg"
        path.write_text(_render_svg_bar_or_scatter(name, points), encoding="utf-8")
        paths[name] = path
        data_path = figure_dir / f"{name}.csv"
        data_path.write_text(_render_points_csv(points), encoding="utf-8")
        paths[f"{name}_data"] = data_path
    return paths


def _frontier_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        x = row.get("token_ratio_vs_full_comm") or row.get("token_ratio_vs_best_no_comm") or 1.0
        points.append(
            {
                "label": str(row.get("experiment_name")),
                "x": _as_float(x),
                "y": _as_float(row.get("faithful_score")),
                "value": _as_float(row.get("faithful_score")),
            }
        )
    return points


def _trigger_utility_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "label": str(row.get("experiment_name")),
            "value": _as_float(row.get("delta_vs_best_no_comm")),
            "x": _as_float(row.get("token_ratio_vs_full_comm") or 1.0),
            "y": _as_float(row.get("delta_vs_best_no_comm")),
        }
        for row in rows
        if "trigger" in str(row.get("experiment_name"))
    ]


def _stage_gap_points(sections: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in ("same_context_main_table", "split_context_main_table", "supporting_evidence_table", "diagnostic_evidence_table"):
        rows.extend(sections.get(key, []))
    return [
        {
            "label": str(row.get("experiment_name")),
            "value": _as_float(row.get("stage_ceiling_gap")),
            "x": float(index),
            "y": _as_float(row.get("stage_ceiling_gap")),
        }
        for index, row in enumerate(rows)
    ]


def _helpful_points(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    points: list[dict[str, Any]] = []
    for row in rows:
        points.append({"label": f"{row.get('experiment_name')}:helpful", "value": _as_float(row.get("helpful_rate"))})
        points.append({"label": f"{row.get('experiment_name')}:harmful", "value": -_as_float(row.get("harmful_rate"))})
    return points


def _render_svg_bar_or_scatter(title: str, points: list[dict[str, Any]]) -> str:
    width = 900
    height = 360
    padding = 48
    if not points:
        return f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}"><text x="24" y="36">{escape(title)}: no data</text></svg>\n'
    values = [float(point.get("value", point.get("y", 0.0)) or 0.0) for point in points]
    min_value = min(0.0, min(values))
    max_value = max(0.0, max(values))
    span = max(max_value - min_value, 1e-9)
    bar_width = max(12, int((width - 2 * padding) / max(1, len(points)) * 0.62))
    zero_y = height - padding - ((0.0 - min_value) / span) * (height - 2 * padding)
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{padding}" y="28" font-size="18" font-family="Arial" fill="#111827">{escape(title)}</text>',
        f'<line x1="{padding}" y1="{zero_y:.2f}" x2="{width - padding}" y2="{zero_y:.2f}" stroke="#9ca3af" stroke-width="1"/>',
    ]
    step = (width - 2 * padding) / max(1, len(points))
    for index, point in enumerate(points):
        value = float(point.get("value", point.get("y", 0.0)) or 0.0)
        x = padding + index * step + (step - bar_width) / 2
        y = height - padding - ((max(value, 0.0) - min_value) / span) * (height - 2 * padding)
        zero = zero_y
        if value < 0:
            y = zero_y
            zero = height - padding - ((value - min_value) / span) * (height - 2 * padding)
        color = "#2563eb" if value >= 0 else "#dc2626"
        lines.append(f'<rect x="{x:.2f}" y="{min(y, zero):.2f}" width="{bar_width}" height="{abs(zero - y):.2f}" fill="{color}" opacity="0.86"/>')
        lines.append(f'<text x="{x:.2f}" y="{height - 16}" font-size="10" font-family="Arial" transform="rotate(-35 {x:.2f},{height - 16})">{escape(str(point.get("label", ""))[:30])}</text>')
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def _render_points_csv(points: list[dict[str, Any]]) -> str:
    lines = ["label,x,y,value"]
    for point in points:
        lines.append(
            ",".join(
                [
                    _csv_cell(str(point.get("label", ""))),
                    _csv_cell(str(point.get("x", ""))),
                    _csv_cell(str(point.get("y", ""))),
                    _csv_cell(str(point.get("value", ""))),
                ]
            )
        )
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
