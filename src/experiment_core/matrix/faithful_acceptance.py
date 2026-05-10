"""把 faithful analysis 结果压缩为 acceptance 结论。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from typing import Any

from experiment_core.matrix.matrix_specs import get_experiment_matrix_spec
from experiment_core.foundation.workspace import default_reports_root

SAME_CONTEXT_NON_INFERIORITY_FLOOR = -0.02
SAME_CONTEXT_FULL_COMM_TOKEN_RATIO_CEILING = 0.85
SPLIT_CONTEXT_FULL_COMM_TOKEN_RATIO_CEILING = 1.0


def render_acceptance_summary(
    analysis_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    published_path: str | Path | None = None,
) -> dict[str, str]:
    """基于分析结果生成 acceptance 的 JSON 与 Markdown 产物。"""
    analysis_path = _resolve_analysis_path(analysis_path_or_root)
    analysis_payload = _load_json(analysis_path)
    summary = build_acceptance_summary(analysis_payload)

    root = Path(output_root) if output_root is not None else analysis_path.parent
    root.mkdir(parents=True, exist_ok=True)
    json_path = root / "acceptance_summary.json"
    markdown_path = root / "acceptance_summary.md"
    published_output = (
        Path(published_path)
        if published_path is not None
        else Path(default_reports_root("faithful_matrix")) / f"{analysis_path.parent.name}-acceptance.md"
    )

    json_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    markdown = render_acceptance_markdown(summary)
    markdown_path.write_text(markdown, encoding="utf-8")
    published_output.parent.mkdir(parents=True, exist_ok=True)
    published_output.write_text(markdown, encoding="utf-8")
    return {
        "analysis_path": str(analysis_path),
        "json_path": str(json_path),
        "markdown_path": str(markdown_path),
        "published_path": str(published_output),
    }


def build_acceptance_summary(analysis_payload: dict[str, Any]) -> dict[str, Any]:
    """对 overall 行逐项打门槛并汇总 accepted / negative control 家族。"""
    overall_rows = [row for row in analysis_payload.get("rows", []) if row.get("dataset") == "overall"]
    evaluated = [_evaluate_overall_row(row) for row in overall_rows]
    accepted_same_context = [row for row in evaluated if row["status"] == "accepted_same_context"]
    accepted_split_context = [row for row in evaluated if row["status"] == "accepted_split_context"]
    negative_control = [row for row in evaluated if row["status"] == "negative_control_family"]
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "counts": {
            "evaluated": len(evaluated),
            "accepted_same_context": len(accepted_same_context),
            "accepted_split_context": len(accepted_split_context),
            "negative_control_family": len(negative_control),
        },
        "accepted_same_context": _sort_rows(accepted_same_context),
        "accepted_split_context": _sort_rows(accepted_split_context),
        "negative_control_family": _sort_rows(negative_control),
        "all_rows": _sort_rows(evaluated),
    }


def render_acceptance_markdown(summary: dict[str, Any]) -> str:
    """把 acceptance 汇总渲染成人可读 Markdown。"""
    lines = [
        "# Faithful Acceptance Summary",
        "",
        f"- generated_at: `{summary.get('generated_at')}`",
        f"- counts: `{json.dumps(summary.get('counts', {}), ensure_ascii=False)}`",
        "",
        "## accepted_same_context",
        "",
    ]
    lines.extend(_render_table(summary.get("accepted_same_context", [])))
    lines.extend(["", "## accepted_split_context", ""])
    lines.extend(_render_table(summary.get("accepted_split_context", [])))
    lines.extend(["", "## negative_control_family", ""])
    lines.extend(_render_table(summary.get("negative_control_family", [])))
    return "\n".join(lines) + "\n"


def _evaluate_overall_row(row: dict[str, Any]) -> dict[str, Any]:
    evaluated = dict(row)
    evaluation_track = str(row.get("evaluation_track") or "")
    reasons: list[str] = []

    if evaluation_track == "same_context":
        score_ok = _as_float(row.get("delta_vs_best_no_comm")) >= SAME_CONTEXT_NON_INFERIORITY_FLOOR
        token_ok = True
        token_gate_basis = _token_gate_basis(row)
        if row.get("full_comm_reference") and token_gate_basis != "none":
            ratio_key = "communication_token_ratio_vs_full_comm" if token_gate_basis == "communication" else "token_ratio_vs_full_comm"
            token_ratio = row.get(ratio_key)
            token_ok = token_ratio is not None and _as_float(token_ratio) <= SAME_CONTEXT_FULL_COMM_TOKEN_RATIO_CEILING
        if not score_ok:
            reasons.append("best_no_comm_non_inferiority_failed")
        if row.get("full_comm_reference") and token_gate_basis != "none" and not token_ok:
            reasons.append("full_comm_token_savings_failed")
        evaluated["status"] = "accepted_same_context" if score_ok and token_ok else "negative_control_family"
        evaluated["score_gate_passed"] = score_ok
        evaluated["token_gate_passed"] = token_ok
        evaluated["token_gate_basis"] = token_gate_basis
        evaluated["notes"] = ", ".join(reasons) if reasons else "passed_same_context_gate"
        return evaluated

    better_than_no_comm = _as_float(row.get("delta_vs_best_no_comm")) > 0.0
    best_no_comm_score = row.get("best_no_comm_score")
    full_context_score = row.get("full_context_score")
    faithful_score = _as_float(row.get("faithful_score"))
    recovered_gap: float | None = None
    required_gap: float | None = None
    half_gap_ok = True
    if best_no_comm_score is not None and full_context_score is not None:
        recovered_gap = round(faithful_score - _as_float(best_no_comm_score), 6)
        total_gap = _as_float(full_context_score) - _as_float(best_no_comm_score)
        required_gap = round(max(total_gap, 0.0) * 0.5, 6)
        half_gap_ok = recovered_gap >= required_gap
    token_ok = True
    token_gate_basis = _token_gate_basis(row)
    if row.get("full_comm_reference") and token_gate_basis != "none":
        ratio_key = "communication_token_ratio_vs_full_comm" if token_gate_basis == "communication" else "token_ratio_vs_full_comm"
        token_ratio = row.get(ratio_key)
        token_ok = token_ratio is not None and _as_float(token_ratio) <= SPLIT_CONTEXT_FULL_COMM_TOKEN_RATIO_CEILING
    if not better_than_no_comm:
        reasons.append("split_no_comm_improvement_failed")
    if row.get("full_context_reference") and not half_gap_ok:
        reasons.append("full_context_gap_recovery_failed")
    if row.get("full_comm_reference") and token_gate_basis != "none" and not token_ok:
        reasons.append("full_comm_token_cap_failed")
    evaluated["status"] = "accepted_split_context" if better_than_no_comm and half_gap_ok and token_ok else "negative_control_family"
    evaluated["score_gate_passed"] = better_than_no_comm
    evaluated["half_gap_gate_passed"] = half_gap_ok
    evaluated["token_gate_passed"] = token_ok
    evaluated["token_gate_basis"] = token_gate_basis
    evaluated["recovered_gap"] = recovered_gap
    evaluated["required_gap"] = required_gap
    evaluated["notes"] = ", ".join(reasons) if reasons else "passed_split_context_gate"
    return evaluated


def _render_table(rows: list[dict[str, Any]]) -> list[str]:
    headers = [
        "family",
        "experiment_name",
        "evaluation_track",
        "primary_method_name",
        "faithful_score",
        "delta_vs_best_no_comm",
        "delta_vs_full_comm",
        "delta_vs_full_context",
        "token_ratio_vs_full_comm",
        "stage_ceiling_gap",
        "engineering_noise_gap",
        "status",
        "notes",
    ]
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines


def _resolve_analysis_path(path_or_root: str | Path) -> Path:
    path = Path(path_or_root)
    if path.is_dir():
        candidate = path / "faithful_analysis.json"
        if candidate.exists():
            return candidate
        candidate = path / "state.json"
        if candidate.exists():
            return candidate.parent / "faithful_analysis.json"
    return path


def _sort_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            str(row.get("evaluation_track") or ""),
            str(row.get("family") or ""),
            str(row.get("experiment_name") or ""),
        ),
    )


def _load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _as_float(value: Any) -> float:
    if value is None:
        return 0.0
    return float(value)


def _token_gate_basis(row: dict[str, Any]) -> str:
    config_path = row.get("config_path")
    if not config_path:
        return "none"
    return get_experiment_matrix_spec(str(config_path)).token_gate_basis

