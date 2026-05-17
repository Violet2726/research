"""面向论文复现支线矩阵的分析产物。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from statistics import mean
from typing import Any

from research_experiments.matrix.matrix_specs import (
    ANALYSIS_MODE_PRIMARY_SUMMARY,
    ANALYSIS_MODE_SCALING_SUMMARY,
    get_experiment_matrix_spec,
)
from research_experiments.reporting.report_views import MatrixStateEntryView, SummaryRowView, SummaryTableView, load_json_payload
from research_experiments.workspace.layout import default_reports_root


POLICY_METRIC_FAMILIES = {"cue", "selective_comm"}


def render_reproduction_analysis(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
) -> dict[str, str]:
    """读取 reproduction matrix 状态并输出 JSON/Markdown。"""

    state_path = _resolve_state_path(state_path_or_root)
    state_payload = load_json_payload(state_path)
    analysis = build_reproduction_analysis(state_payload)

    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "reproduction_analysis.json"
    markdown_path = root / "reproduction_analysis.md"
    published_output = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("reproduction_matrix")) / f"{state_path.parent.name}-reproduction.md"
    )

    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_reproduction_analysis_markdown(analysis)
    markdown_path.write_text(markdown, encoding="utf-8")
    published_output.parent.mkdir(parents=True, exist_ok=True)
    published_output.write_text(markdown, encoding="utf-8")
    return {
        "state_path": str(state_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "published_path": str(published_output),
    }


def build_reproduction_analysis(state_payload: dict[str, Any]) -> dict[str, Any]:
    """把平行论文复现矩阵状态整理成可读分析载荷。"""

    entries_payload: list[dict[str, Any]] = []
    state_entries = [MatrixStateEntryView.from_row(item) for item in state_payload.get("semantic_entries", [])]
    matrix_id = str(state_payload.get("matrix_id") or "reproduction")
    for entry in state_entries:
        if entry.status != "completed" or not entry.run_dir:
            continue
        spec = get_experiment_matrix_spec(entry.config_path, matrix_id)
        run_dir = Path(entry.run_dir)
        metrics = _load_metrics(run_dir, entry.family)
        summary = SummaryTableView.from_metrics_payload(metrics)
        if spec.analysis_mode == ANALYSIS_MODE_SCALING_SUMMARY:
            entries_payload.append(
                {
                    "family": entry.family,
                    "experiment_name": entry.experiment_name,
                    "config_path": entry.config_path,
                    "track_name": spec.track_name,
                    "entry_role": spec.entry_role,
                    "analysis_mode": spec.analysis_mode,
                    "comparison_scope": spec.comparison_scope,
                    "primary_method_name": spec.primary_method_name,
                    "primary_metric_field": spec.primary_metric_field,
                    "primary_metric_label": spec.primary_metric_label,
                    "run_dir": entry.run_dir,
                    "summary_rows": [row.raw for row in summary.rows],
                    "overall_row": None,
                    "dataset_rows": [],
                    "scaling_summary": load_json_payload(run_dir / "scaling_summary.json"),
                }
            )
            continue

        overall_row = _find_or_aggregate_method_row(summary.rows, spec.primary_method_name, "overall", spec.primary_metric_field)
        dataset_rows = [
            _normalize_summary_row(
                row,
                spec=spec,
                family=entry.family,
                experiment_name=entry.experiment_name,
                run_dir=entry.run_dir,
                metric_field=spec.primary_metric_field,
            )
            for row in summary.rows
            if row.method_name == spec.primary_method_name and row.dataset != "overall"
        ]
        entries_payload.append(
            {
                "family": entry.family,
                "experiment_name": entry.experiment_name,
                "config_path": entry.config_path,
                "track_name": spec.track_name,
                "entry_role": spec.entry_role,
                "analysis_mode": spec.analysis_mode,
                "comparison_scope": spec.comparison_scope,
                "primary_method_name": spec.primary_method_name,
                "primary_metric_field": spec.primary_metric_field,
                "primary_metric_label": spec.primary_metric_label,
                "run_dir": entry.run_dir,
                "summary_rows": [row.raw for row in summary.rows],
                "overall_row": None
                if overall_row is None
                else _normalize_summary_row(
                    overall_row,
                    spec=spec,
                    family=entry.family,
                    experiment_name=entry.experiment_name,
                    run_dir=entry.run_dir,
                    metric_field=spec.primary_metric_field,
                ),
                "dataset_rows": dataset_rows,
                "scaling_summary": {},
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "matrix_id": matrix_id,
        "matrix_kind": str(state_payload.get("matrix_kind") or "reproduction_matrix"),
        "phase_name": state_payload.get("overrides", {}).get("phase_name"),
        "model_ref": state_payload.get("overrides", {}).get("model_ref"),
        "counts": state_payload.get("counts", {}),
        "entries": entries_payload,
    }


def render_reproduction_analysis_markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Reproduction Analysis",
        "",
        f"- generated_at: `{payload.get('generated_at')}`",
        f"- matrix_id: `{payload.get('matrix_id')}`",
        f"- phase_name: `{payload.get('phase_name')}`",
        f"- model_ref: `{payload.get('model_ref')}`",
        "",
        "## Entries",
        "",
        "| family | experiment_name | track_name | entry_role | analysis_mode | primary_method_name | primary_metric_label | primary_metric_value | total_tokens_mean | run_dir |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for entry in payload.get("entries", []):
        overall = entry.get("overall_row") or {}
        lines.append(
            "| "
            + " | ".join(
                [
                    str(entry.get("family") or ""),
                    str(entry.get("experiment_name") or ""),
                    str(entry.get("track_name") or ""),
                    str(entry.get("entry_role") or ""),
                    str(entry.get("analysis_mode") or ""),
                    str(entry.get("primary_method_name") or ""),
                    str(entry.get("primary_metric_label") or ""),
                    _display(overall.get("primary_metric_value")),
                    _display(overall.get("total_tokens_mean")),
                    str(entry.get("run_dir") or ""),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines) + "\n"


def _normalize_summary_row(
    row: SummaryRowView,
    *,
    spec,
    family: str,
    experiment_name: str,
    run_dir: str,
    metric_field: str,
) -> dict[str, Any]:
    primary_metric_value = _metric_value_from_row(row, metric_field)
    return {
        "family": family,
        "experiment_name": experiment_name,
        "track_name": spec.track_name,
        "entry_role": spec.entry_role,
        "analysis_mode": spec.analysis_mode,
        "comparison_scope": spec.comparison_scope,
        "dataset": row.dataset,
        "primary_method_name": spec.primary_method_name,
        "primary_metric_field": metric_field,
        "primary_metric_label": spec.primary_metric_label,
        "primary_metric_value": primary_metric_value,
        "total_tokens_mean": _as_optional_float(row.raw.get("total_tokens_mean")),
        "communication_tokens_mean": _as_optional_float(row.raw.get("communication_tokens_mean")),
        "calls_per_question_mean": _as_optional_float(row.raw.get("calls_per_question_mean")),
        "run_dir": run_dir,
    }


def _find_or_aggregate_method_row(
    summary_rows: list[SummaryRowView],
    method_name: str,
    dataset: str,
    metric_field: str,
) -> SummaryRowView | None:
    direct = next((row for row in summary_rows if row.method_name == method_name and row.dataset == dataset), None)
    if direct is not None:
        return direct
    if dataset != "overall":
        return None
    method_rows = [row for row in summary_rows if row.method_name == method_name and row.dataset != "overall"]
    if not method_rows:
        return None
    metric_values = [_metric_value_from_row(row, metric_field) for row in method_rows]
    metric_values = [value for value in metric_values if value is not None]
    return SummaryRowView.from_row(
        {
            "dataset": "overall",
            "method_name": method_name,
            metric_field: round(mean(metric_values), 6) if metric_values else None,
            "total_tokens_mean": round(mean(_as_float(row.raw.get("total_tokens_mean")) for row in method_rows), 6),
            "communication_tokens_mean": round(mean(_as_float(row.raw.get("communication_tokens_mean")) for row in method_rows), 6),
            "calls_per_question_mean": round(mean(_as_float(row.raw.get("calls_per_question_mean")) for row in method_rows), 6),
        }
    )


def _metric_value_from_row(row: SummaryRowView, metric_field: str) -> float | None:
    return _as_optional_float(row.raw.get(metric_field))


def _load_metrics(run_dir: Path, family: str) -> dict[str, Any]:
    metric_name = "policy_metrics.json" if family in POLICY_METRIC_FAMILIES else "metrics.json"
    return load_json_payload(run_dir / metric_name)


def _resolve_state_path(path_or_root: str | Path) -> Path:
    path = Path(path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path


def _display(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


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
