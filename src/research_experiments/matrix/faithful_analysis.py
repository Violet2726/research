"""汇总 count20 矩阵状态并生成 faithful 对照分析。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from research_experiments.core.foundation.workspace import default_reports_root
from research_experiments.matrix.matrix_specs import get_experiment_matrix_spec
from research_experiments.reporting.report_views import (
    MatrixAnalysisRowView,
    MatrixAnalysisTableView,
    MatrixStateEntryView,
    SummaryRowView,
    SummaryTableView,
    load_json_payload,
    load_jsonl_rows,
)


POLICY_METRIC_FAMILIES = {"cue", "selective_comm"}
PREDICTION_FILE_CANDIDATES = (
    "policy_predictions.jsonl",
    "final_predictions.jsonl",
    "predictions.jsonl",
)
STAGE_SCORE_KEYS = (
    "score",
    "stage_a_score",
    "stage_b_score",
    "initial_vote_score",
    "communication_score",
    "audited_score",
)


def render_faithful_analysis(
    state_path_or_root: str | Path,
    *,
    reference_state_path_or_root: str | Path | None = None,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
) -> dict[str, str]:
    """读取矩阵状态并输出 faithful analysis 的 JSON/Markdown 文件。"""
    state_path = _resolve_state_path(state_path_or_root)
    state_payload = load_json_payload(state_path)
    reference_lookup = _build_reference_lookup(reference_state_path_or_root)
    analysis = build_faithful_analysis(state_payload, reference_lookup=reference_lookup)

    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "faithful_analysis.json"
    markdown_path = root / "faithful_analysis.md"
    published_output = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("faithful_matrix")) / f"{state_path.parent.name}-faithful.md"
    )

    import json

    json_path.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_faithful_analysis_markdown(analysis)
    markdown_path.write_text(markdown, encoding="utf-8")
    published_output.parent.mkdir(parents=True, exist_ok=True)
    published_output.write_text(markdown, encoding="utf-8")
    return {
        "state_path": str(state_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "published_path": str(published_output),
    }


def build_faithful_analysis(
    state_payload: dict[str, Any],
    *,
    reference_lookup: dict[tuple[str, str, str], float] | None = None,
) -> dict[str, Any]:
    """把跨实验运行状态整理成可比较的 faithful 指标行。"""
    rows: list[dict[str, Any]] = []
    experiments: list[dict[str, Any]] = []
    reference_lookup = reference_lookup or {}

    entries = [MatrixStateEntryView.from_row(entry) for entry in state_payload.get("semantic_entries", [])]
    for entry in entries:
        if entry.status != "completed" or not entry.run_dir:
            continue
        spec = get_experiment_matrix_spec(entry.config_path)
        run_dir = Path(entry.run_dir)
        summary = _load_summary_rows(run_dir, entry.family)
        if not summary.rows:
            continue

        primary_method = spec.primary_method_name
        best_no_comm_control = _pick_reference_method(summary.rows, spec.best_no_comm_candidates)
        full_comm_reference = _pick_reference_method(
            summary.rows,
            (spec.full_comm_reference,) if spec.full_comm_reference else (),
        )
        full_context_reference = _pick_reference_method(
            summary.rows,
            (spec.full_context_reference,) if spec.full_context_reference else (),
        )
        family_envelope = _pick_reference_method(
            summary.rows,
            tuple({row.method_name for row in summary.rows if row.method_name}),
        )
        prediction_rows = _load_prediction_rows(run_dir)

        experiments.append(
            {
                "family": entry.family,
                "experiment_name": entry.experiment_name,
                "config_path": entry.config_path,
                "evaluation_track": spec.evaluation_track,
                "evidence_tier": spec.evidence_tier,
                "primary_method_name": primary_method,
                "best_no_comm_control": best_no_comm_control,
                "full_comm_reference": full_comm_reference,
                "full_context_reference": full_context_reference,
                "family_envelope": family_envelope,
                "run_dir": entry.run_dir,
            }
        )

        primary_rows = [row for row in summary.rows if row.method_name == primary_method]
        for primary_row in primary_rows:
            dataset = primary_row.dataset
            no_comm_row = _find_method_row(summary.rows, best_no_comm_control, dataset) if best_no_comm_control else None
            full_comm_row = _find_method_row(summary.rows, full_comm_reference, dataset) if full_comm_reference else None
            full_context_row = _find_method_row(summary.rows, full_context_reference, dataset) if full_context_reference else None
            envelope_row = _find_method_row(summary.rows, family_envelope, dataset) if family_envelope else None
            stage_ceiling = _compute_stage_ceiling(prediction_rows, dataset=dataset, method_name=primary_method)
            faithful_score = _as_float(primary_row.accuracy_mean)
            reference_key = (entry.config_path, primary_method, dataset)
            reference_score = reference_lookup.get(reference_key)
            rows.append(
                MatrixAnalysisRowView.from_row(
                    {
                    "family": entry.family,
                    "experiment_name": entry.experiment_name,
                    "evaluation_track": spec.evaluation_track,
                    "evidence_tier": spec.evidence_tier,
                    "config_path": entry.config_path,
                    "dataset": dataset,
                    "primary_method_name": primary_method,
                    "faithful_score": faithful_score,
                    "best_no_comm_control": best_no_comm_control,
                    "best_no_comm_score": _as_float(no_comm_row.accuracy_mean) if no_comm_row else None,
                    "delta_vs_best_no_comm": _score_delta(primary_row, no_comm_row),
                    "full_comm_reference": full_comm_reference,
                    "full_comm_score": _as_float(full_comm_row.accuracy_mean) if full_comm_row else None,
                    "delta_vs_full_comm": _score_delta(primary_row, full_comm_row),
                    "full_context_reference": full_context_reference,
                    "full_context_score": _as_float(full_context_row.accuracy_mean) if full_context_row else None,
                    "delta_vs_full_context": _score_delta(primary_row, full_context_row),
                    "family_envelope": family_envelope,
                    "family_envelope_score": _as_float(envelope_row.accuracy_mean) if envelope_row else None,
                    "delta_vs_family_envelope": _score_delta(primary_row, envelope_row),
                    "stage_ceiling": stage_ceiling,
                    "stage_ceiling_gap": None if stage_ceiling is None else round(stage_ceiling - faithful_score, 6),
                    "token_ratio_vs_best_no_comm": _token_ratio(primary_row, no_comm_row),
                    "token_ratio_vs_full_comm": _token_ratio(primary_row, full_comm_row),
                    "communication_token_ratio_vs_full_comm": _communication_token_ratio(primary_row, full_comm_row),
                    "engineering_noise_gap": None if reference_score is None else round(faithful_score - reference_score, 6),
                    "calls_per_question_mean": _as_float(primary_row.calls_per_question_mean),
                    "total_tokens_mean": _as_float(primary_row.total_tokens_mean),
                    "communication_tokens_mean": _as_float(primary_row.communication_tokens_mean),
                    "run_dir": entry.run_dir,
                }
                ).to_dict()
            )
        if not any(row.dataset == "overall" for row in primary_rows):
            primary_overall = _find_or_aggregate_method_row(summary.rows, primary_method, "overall")
            if primary_overall is not None:
                no_comm_row = _find_or_aggregate_method_row(summary.rows, best_no_comm_control, "overall") if best_no_comm_control else None
                full_comm_row = _find_or_aggregate_method_row(summary.rows, full_comm_reference, "overall") if full_comm_reference else None
                full_context_row = _find_or_aggregate_method_row(summary.rows, full_context_reference, "overall") if full_context_reference else None
                envelope_row = _find_or_aggregate_method_row(summary.rows, family_envelope, "overall") if family_envelope else None
                stage_ceiling = _compute_stage_ceiling(prediction_rows, dataset="overall", method_name=primary_method)
                faithful_score = _as_float(primary_overall.accuracy_mean)
                reference_key = (entry.config_path, primary_method, "overall")
                reference_score = reference_lookup.get(reference_key)
                rows.append(
                    MatrixAnalysisRowView.from_row(
                        {
                        "family": entry.family,
                        "experiment_name": entry.experiment_name,
                        "evaluation_track": spec.evaluation_track,
                        "evidence_tier": spec.evidence_tier,
                        "config_path": entry.config_path,
                        "dataset": "overall",
                        "primary_method_name": primary_method,
                        "faithful_score": faithful_score,
                        "best_no_comm_control": best_no_comm_control,
                        "best_no_comm_score": _as_float(no_comm_row.accuracy_mean) if no_comm_row else None,
                        "delta_vs_best_no_comm": _score_delta(primary_overall, no_comm_row),
                        "full_comm_reference": full_comm_reference,
                        "full_comm_score": _as_float(full_comm_row.accuracy_mean) if full_comm_row else None,
                        "delta_vs_full_comm": _score_delta(primary_overall, full_comm_row),
                        "full_context_reference": full_context_reference,
                        "full_context_score": _as_float(full_context_row.accuracy_mean) if full_context_row else None,
                        "delta_vs_full_context": _score_delta(primary_overall, full_context_row),
                        "family_envelope": family_envelope,
                        "family_envelope_score": _as_float(envelope_row.accuracy_mean) if envelope_row else None,
                        "delta_vs_family_envelope": _score_delta(primary_overall, envelope_row),
                        "stage_ceiling": stage_ceiling,
                        "stage_ceiling_gap": None if stage_ceiling is None else round(stage_ceiling - faithful_score, 6),
                        "token_ratio_vs_best_no_comm": _token_ratio(primary_overall, no_comm_row),
                        "token_ratio_vs_full_comm": _token_ratio(primary_overall, full_comm_row),
                        "communication_token_ratio_vs_full_comm": _communication_token_ratio(primary_overall, full_comm_row),
                        "engineering_noise_gap": None if reference_score is None else round(faithful_score - reference_score, 6),
                        "calls_per_question_mean": _as_float(primary_overall.calls_per_question_mean),
                        "total_tokens_mean": _as_float(primary_overall.total_tokens_mean),
                        "communication_tokens_mean": _as_float(primary_overall.communication_tokens_mean),
                        "run_dir": entry.run_dir,
                    }
                    ).to_dict()
                )

    table = MatrixAnalysisTableView.from_rows(rows)
    same_context_rows = [row.to_dict() for row in table.rows if row.evaluation_track == "same_context"]
    split_context_rows = [row.to_dict() for row in table.rows if row.evaluation_track == "split_context"]
    overall_rows = [row.to_dict() for row in table.overall_rows()]
    same_context_overall = [row for row in overall_rows if row["evaluation_track"] == "same_context"]
    split_context_overall = [row for row in overall_rows if row["evaluation_track"] == "split_context"]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rows": rows,
        "experiments": experiments,
        "same_context_overall": _sort_rows(same_context_overall),
        "split_context_overall": _sort_rows(split_context_overall),
        "combined_overall": _sort_rows(overall_rows),
        "same_context_rows": _sort_rows([row for row in same_context_rows if row["dataset"] != "overall"]),
        "split_context_rows": _sort_rows([row for row in split_context_rows if row["dataset"] != "overall"]),
    }


def render_faithful_analysis_markdown(analysis: dict[str, Any]) -> str:
    """把 faithful analysis 汇总渲染成人类审阅用 Markdown。"""
    lines = [
        "# Faithful Analysis",
        "",
        f"- generated_at: `{analysis.get('generated_at')}`",
        f"- experiment_count: `{len(analysis.get('experiments', []))}`",
        "",
        "## same_context_overall",
        "",
    ]
    lines.extend(_render_table(analysis.get("same_context_overall", [])))
    lines.extend(["", "## split_context_overall", ""])
    lines.extend(_render_table(analysis.get("split_context_overall", [])))
    lines.extend(["", "## same_context_rows", ""])
    lines.extend(_render_table(analysis.get("same_context_rows", [])))
    lines.extend(["", "## split_context_rows", ""])
    lines.extend(_render_table(analysis.get("split_context_rows", [])))
    return "\n".join(lines) + "\n"


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "family",
        "experiment_name",
        "evidence_tier",
        "dataset",
        "primary_method_name",
        "faithful_score",
        "best_no_comm_control",
        "delta_vs_best_no_comm",
        "full_comm_reference",
        "delta_vs_full_comm",
        "family_envelope",
        "delta_vs_family_envelope",
        "stage_ceiling",
        "stage_ceiling_gap",
        "engineering_noise_gap",
    ]
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(_display_value(row.get(header)) for header in headers) + " |")
    return lines


def _display_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _build_reference_lookup(reference_state_path_or_root: str | Path | None) -> dict[tuple[str, str, str], float]:
    if reference_state_path_or_root is None:
        return {}
    reference_state = load_json_payload(_resolve_state_path(reference_state_path_or_root))
    lookup: dict[tuple[str, str, str], float] = {}
    for entry in [MatrixStateEntryView.from_row(item) for item in reference_state.get("semantic_entries", [])]:
        if entry.status != "completed" or not entry.run_dir:
            continue
        summary_rows = _load_summary_rows(Path(entry.run_dir), entry.family)
        for row in summary_rows.rows:
            method_name = row.method_name
            dataset = row.dataset
            if not method_name or not dataset:
                continue
            lookup[(entry.config_path, str(method_name), str(dataset))] = _as_float(row.accuracy_mean)
    return lookup


def _resolve_state_path(state_path_or_root: str | Path) -> Path:
    path = Path(state_path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path


def _load_summary_rows(run_dir: Path, family: str) -> SummaryTableView:
    metric_name = "policy_metrics.json" if family in POLICY_METRIC_FAMILIES else "metrics.json"
    payload = load_json_payload(run_dir / metric_name)
    return SummaryTableView.from_metrics_payload(payload)


def _load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    for filename in PREDICTION_FILE_CANDIDATES:
        target = run_dir / filename
        if target.exists():
            return load_jsonl_rows(target)
    return []


def _pick_reference_method(summary_rows: list[SummaryRowView], candidates: tuple[str, ...]) -> str | None:
    if not candidates:
        return None
    ranked: list[tuple[float, float, str]] = []
    for method_name in candidates:
        method_rows = [row for row in summary_rows if row.method_name == method_name]
        if not method_rows:
            continue
        overall_row = next((row for row in method_rows if row.dataset == "overall"), None)
        if overall_row is not None:
            ranked.append(
                (
                    _as_float(overall_row.accuracy_mean),
                    -_as_float(overall_row.total_tokens_mean),
                    method_name,
                )
            )
            continue
        ranked.append(
            (
                mean(_as_float(row.accuracy_mean) for row in method_rows),
                -mean(_as_float(row.total_tokens_mean) for row in method_rows),
                method_name,
            )
        )
    if not ranked:
        return None
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))
    return ranked[0][2]


def _find_method_row(summary_rows: list[SummaryRowView], method_name: str | None, dataset: str) -> SummaryRowView | None:
    if method_name is None:
        return None
    return next(
        (
            row
            for row in summary_rows
            if row.method_name == method_name and row.dataset == dataset
        ),
        None,
    )


def _find_or_aggregate_method_row(
    summary_rows: list[SummaryRowView],
    method_name: str | None,
    dataset: str,
) -> SummaryRowView | None:
    if method_name is None:
        return None
    direct = _find_method_row(summary_rows, method_name, dataset)
    if direct is not None:
        return direct
    if dataset != "overall":
        return None
    method_rows = [
        row
        for row in summary_rows
        if row.method_name == method_name and row.dataset != "overall"
    ]
    if not method_rows:
        return None
    return SummaryRowView.from_row({
        "dataset": "overall",
        "method_name": method_name,
        "accuracy_mean": round(mean(_as_float(row.accuracy_mean) for row in method_rows), 6),
        "total_tokens_mean": round(mean(_as_float(row.total_tokens_mean) for row in method_rows), 6),
        "communication_tokens_mean": round(mean(_as_float(row.communication_tokens_mean) for row in method_rows), 6),
        "calls_per_question_mean": round(mean(_as_float(row.calls_per_question_mean) for row in method_rows), 6),
    })


def _compute_stage_ceiling(
    prediction_rows: list[dict[str, Any]],
    *,
    dataset: str,
    method_name: str,
) -> float | None:
    if dataset == "overall":
        filtered = [row for row in prediction_rows if row.get("method_name") == method_name]
    else:
        filtered = [
            row
            for row in prediction_rows
            if row.get("dataset") == dataset and row.get("method_name") == method_name
        ]
    if not filtered:
        return None
    maxima: list[float] = []
    for row in filtered:
        scores = [
            _as_optional_float(row.get(key))
            for key in STAGE_SCORE_KEYS
        ]
        valid_scores = [score for score in scores if score is not None]
        if not valid_scores:
            continue
        maxima.append(max(valid_scores))
    if not maxima:
        return None
    return round(mean(maxima), 6)


def _score_delta(primary_row: SummaryRowView, reference_row: SummaryRowView | None) -> float | None:
    if reference_row is None:
        return None
    return round(_as_float(primary_row.accuracy_mean) - _as_float(reference_row.accuracy_mean), 6)


def _token_ratio(primary_row: SummaryRowView, reference_row: SummaryRowView | None) -> float | None:
    if reference_row is None:
        return None
    denominator = _as_float(reference_row.total_tokens_mean)
    if denominator <= 0:
        return None
    return round(_as_float(primary_row.total_tokens_mean) / denominator, 6)


def _communication_token_ratio(primary_row: SummaryRowView, reference_row: SummaryRowView | None) -> float | None:
    if reference_row is None:
        return None
    denominator = _as_float(reference_row.communication_tokens_mean)
    if denominator <= 0:
        return None
    return round(_as_float(primary_row.communication_tokens_mean) / denominator, 6)


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (row["family"], row["experiment_name"], row["dataset"]))


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_float(value: Any) -> float:
    return float(value or 0.0)



