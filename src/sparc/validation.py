"""SPARC 运行结果校验。

该校验器重点验证共享 Stage A 哈希、跳过审计时的 token 归零、
局部审计只访问局部信息，以及不同实验形态下的配对设计是否成立。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any


DEFAULT_AUDITING_METHODS = ["majority_vote", "single_judge", "final_round_vote", "local_auditing"]


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
    auditing_paired_check = _validate_auditing_paired_design(manifest, prediction_rows)
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
                auditing_paired_check["passed"],
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
            "auditing_ablation_paired_design_check": auditing_paired_check,
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


def _validate_auditing_paired_design(
    manifest: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    if manifest.get("experiment_kind") != "auditing_ablation":
        return {"passed": True, "enabled": False}

    expected_methods = [
        str(method)
        for method in manifest.get("aggregation_methods", [])
    ] or DEFAULT_AUDITING_METHODS
    rows_by_method = {
        method: [row for row in prediction_rows if row.get("method_name") == method]
        for method in expected_methods
    }
    expected_count = _expected_prediction_count(manifest)
    sample_sets = {
        method: {(row.get("dataset"), row.get("sample_id")) for row in rows}
        for method, rows in rows_by_method.items()
    }
    observed_union = set().union(*sample_sets.values()) if sample_sets else set()
    reference_method = expected_methods[0] if expected_methods else None
    reference_set = sample_sets.get(reference_method, set()) if reference_method else set()

    missing_methods = [method for method, rows in rows_by_method.items() if not rows]
    count_mismatches = [
        {
            "method_name": method,
            "observed_count": len(rows),
            "expected_count": expected_count or len(observed_union),
        }
        for method, rows in rows_by_method.items()
        if len(rows) != (expected_count or len(observed_union))
    ]
    sample_set_mismatches = [
        {
            "method_name": method,
            "missing_from_method": sorted(reference_set - sample_set)[:20],
            "extra_in_method": sorted(sample_set - reference_set)[:20],
        }
        for method, sample_set in sample_sets.items()
        if sample_set != reference_set
    ]
    stage_b_mismatches = _stage_b_trace_mismatches(prediction_rows, expected_methods)
    unexpected_methods = sorted(
        {
            str(row.get("method_name"))
            for row in prediction_rows
            if row.get("method_name") not in expected_methods
        }
    )
    passed = not any(
        [
            missing_methods,
            count_mismatches,
            sample_set_mismatches,
            stage_b_mismatches,
            unexpected_methods,
        ]
    )
    return {
        "passed": passed,
        "enabled": True,
        "expected_methods": expected_methods,
        "expected_count_per_method": expected_count or len(observed_union),
        "observed_count_per_method": {
            method: len(rows)
            for method, rows in rows_by_method.items()
        },
        "sample_count": len(observed_union),
        "missing_methods": missing_methods,
        "unexpected_methods": unexpected_methods,
        "count_mismatches": count_mismatches,
        "sample_set_mismatches": sample_set_mismatches,
        "stage_b_trace_mismatches": stage_b_mismatches[:20],
    }


def _stage_b_trace_mismatches(
    prediction_rows: list[dict[str, Any]],
    expected_methods: list[str],
) -> list[dict[str, Any]]:
    stage_b_methods = [method for method in expected_methods if method != "majority_vote"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in prediction_rows:
        if row.get("method_name") in stage_b_methods:
            grouped[(str(row.get("dataset")), str(row.get("sample_id")))].append(row)

    mismatches: list[dict[str, Any]] = []
    for (dataset, sample_id), rows in grouped.items():
        present_methods = {str(row.get("method_name")) for row in rows}
        hashes = {
            row.get("stage_b_trace_hash_used")
            for row in rows
            if row.get("stage_b_trace_hash_used")
        }
        if present_methods != set(stage_b_methods) or len(hashes) != 1:
            mismatches.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "present_methods": sorted(present_methods),
                    "expected_methods": stage_b_methods,
                    "stage_b_trace_hashes": sorted(str(item) for item in hashes),
                }
            )
    return mismatches


def _expected_prediction_count(manifest: dict[str, Any]) -> int | None:
    phase_metadata = manifest.get("phase_metadata") or {}
    split_suffix = str(phase_metadata.get("split_suffix") or "")
    if split_suffix == "smoke20_seed42":
        return _sum_benchmark_size(manifest, "smoke_size")
    if split_suffix == "pilot100_seed42":
        return _sum_benchmark_size(manifest, "pilot_size")
    if split_suffix in {"dev300_seed42", "dev_full_229_seed42"}:
        return _sum_benchmark_size(manifest, "main_size")
    return None


def _sum_benchmark_size(manifest: dict[str, Any], key: str) -> int | None:
    benchmarks = manifest.get("benchmarks") or []
    if not benchmarks:
        return None
    sizes = []
    for benchmark in benchmarks:
        value = benchmark.get(key)
        if value is None:
            return None
        sizes.append(int(value))
    return sum(sizes)


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
