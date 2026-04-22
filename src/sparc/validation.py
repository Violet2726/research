"""SPARC 运行结果校验。"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any


def validate_run(run_dir: str | Path, compare_run_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    experiment_kind = manifest.get("experiment_kind", "")
    report_name = {
        "content_ablation": "content_ablation_report.md",
        "auditing_ablation": "auditing_ablation_report.md",
        "sparc_v1": "sparc_v1_report.md",
    }.get(experiment_kind, "report.md")
    required = [
        "manifest.json",
        "stage_a_turns.jsonl",
        "message_packets.jsonl",
        "belief_updates.jsonl",
        "audit_turns.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "diagnostics.json",
        "progress.json",
        "paper_summary.csv",
        report_name,
    ]
    missing = [name for name in required if not (root / name).exists()]
    stage_a_rows = _load_jsonl(root / "stage_a_turns.jsonl")
    belief_rows = _load_jsonl(root / "belief_updates.jsonl")
    audit_rows = _load_jsonl(root / "audit_turns.jsonl")
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl")
    request_failures = sum(1 for row in stage_a_rows + belief_rows + audit_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in stage_a_rows + belief_rows + audit_rows if row.get("output_status") == "schema_fail")
    shared_hash_check = _validate_shared_stage_a_hashes(prediction_rows)
    skipped_audit_check = _validate_skipped_audit_tokens(prediction_rows)
    local_audit_scope_check = _validate_local_audit_scope(audit_rows)
    compare_check = _validate_compare_run(prediction_rows, compare_run_dir)
    return {
        "run_dir": str(root),
        "passed": all(
            [
                not missing,
                request_failures == 0,
                schema_failures == 0,
                shared_hash_check["passed"],
                skipped_audit_check["passed"],
                local_audit_scope_check["passed"],
                compare_check["passed"],
                bool(prediction_rows),
            ]
        ),
        "missing_files": missing,
        "checks": {
            "request_failures_total": request_failures,
            "schema_failures_total": schema_failures,
            "shared_stage_a_hash_check": shared_hash_check,
            "skipped_audit_zero_token_check": skipped_audit_check,
            "local_audit_scope_check": local_audit_scope_check,
            "compare_run_sample_ids_check": compare_check,
        },
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _validate_shared_stage_a_hashes(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].append(row)
    for (dataset, sample_id), rows in grouped.items():
        hashes = {row.get("stage_a_trace_hash") for row in rows if row.get("stage_a_trace_hash")}
        if len(hashes) > 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "hashes": sorted(hashes)})
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_skipped_audit_tokens(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    for row in prediction_rows:
        if row.get("audit_status") in {"skipped_consensus", "early_exit", "not_applicable"} and float(row.get("audit_tokens_per_question") or 0.0) != 0.0:
            mismatches.append(
                {
                    "dataset": row.get("dataset"),
                    "sample_id": row.get("sample_id"),
                    "method_name": row.get("method_name"),
                    "audit_status": row.get("audit_status"),
                    "audit_tokens_per_question": row.get("audit_tokens_per_question"),
                }
            )
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _validate_local_audit_scope(audit_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "method_name": row.get("method_name"),
        }
        for row in audit_rows
        if row.get("method_name") == "local_auditing" and bool(row.get("input_includes_full_debate"))
    ]
    return {"passed": len(violations) == 0, "violation_count": len(violations), "violations": violations[:20]}


def _validate_compare_run(
    prediction_rows: list[dict[str, Any]],
    compare_run_dir: str | Path | None,
) -> dict[str, Any]:
    if compare_run_dir is None:
        return {"passed": True, "enabled": False}
    compare_rows = _load_jsonl(Path(compare_run_dir) / "final_predictions.jsonl")
    current_ids = sorted({(row.get("dataset"), row.get("sample_id")) for row in prediction_rows})
    compare_ids = sorted({(row.get("dataset"), row.get("sample_id")) for row in compare_rows})
    return {
        "passed": current_ids == compare_ids,
        "enabled": True,
        "current_count": len(current_ids),
        "compare_count": len(compare_ids),
    }


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
