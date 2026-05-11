"""Free-MAD-lite 指标与摘要生成。"""

from __future__ import annotations

from pathlib import Path
import csv
from typing import Any

from experiment_core.foundation.family_helpers import safe_mean
from free_mad_lite.logic import METHOD_ORDER
from free_mad_lite.prompting import anti_conformity_prompt_hash


def build_metrics_payload(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    summary: list[dict[str, Any]] = []
    datasets = sorted({row["dataset"] for row in prediction_rows})
    for dataset in [*datasets, "overall"]:
        rows_for_dataset = prediction_rows if dataset == "overall" else [row for row in prediction_rows if row["dataset"] == dataset]
        for method in METHOD_ORDER:
            rows = [row for row in rows_for_dataset if row["method_name"] == method]
            if not rows:
                continue
            accuracy = safe_mean(float(row["score"]) for row in rows)
            total_tokens = safe_mean(float(row["total_tokens_per_question"]) for row in rows)
            summary.append(
                {
                    "dataset": dataset,
                    "model_name": rows[0]["model_name"],
                    "method_name": method,
                    "method_kind": rows[0]["method_kind"],
                    "question_count": len(rows),
                    "accuracy_mean": round(accuracy, 6),
                    "prompt_tokens_mean": round(safe_mean(float(row["prompt_tokens_per_question"]) for row in rows), 6),
                    "completion_tokens_mean": round(safe_mean(float(row["completion_tokens_per_question"]) for row in rows), 6),
                    "total_tokens_mean": round(total_tokens, 6),
                    "communication_tokens_mean": round(safe_mean(float(row["communication_tokens_per_question"]) for row in rows), 6),
                    "latency_ms_mean": round(safe_mean(float(row["latency_ms_per_question"]) for row in rows), 6),
                    "calls_per_question_mean": round(safe_mean(float(row["calls_per_question"]) for row in rows), 6),
                    "acc_per_1k_tokens": round((accuracy / total_tokens * 1000) if total_tokens else 0.0, 6),
                    "judge_fallback_rate": round(safe_mean(1.0 if row.get("judge_fallback_used") else 0.0 for row in rows), 6),
                    "changed_answer_rate": round(safe_mean(float(row.get("changed_answer_rate") or 0.0) for row in rows), 6),
                    "corrected_count": sum(1 for row in rows if row.get("corrected_by_method")),
                    "harmed_count": sum(1 for row in rows if row.get("harmed_by_method")),
                    "minority_rescue_count": sum(1 for row in rows if row.get("minority_rescue")),
                }
            )
    return {"summary": summary}


def build_diagnostics_payload(prediction_rows: list[dict[str, Any]], score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    trajectory_rows = [row for row in prediction_rows if row.get("method_name") == "free_mad_lite_llm_trajectory"]
    return {
        "judge_fallback_rate": round(safe_mean(1.0 if row.get("judge_fallback_used") else 0.0 for row in trajectory_rows), 6),
        "judge_fallback_count": sum(1 for row in score_rows if row.get("judge_fallback_used")),
        "trajectory_score_rows": len(score_rows),
        "anti_conformity_prompt_hash": anti_conformity_prompt_hash(),
    }


def write_paper_summary(path: Path, metrics_payload: dict[str, Any]) -> None:
    fieldnames = [
        "dataset",
        "model_name",
        "method_name",
        "accuracy_mean",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "acc_per_1k_tokens",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_payload.get("summary", []):
            writer.writerow({key: row.get(key) for key in fieldnames})
