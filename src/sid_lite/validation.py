"""SID-lite 运行产物验证。

该校验器重点验证共享前缀、公平对照与机制约束：
例如 early-exit 是否真的零通信、共享 Stage A 哈希是否一致、置信度失效时是否 fail-open。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any


REQUIRED_FILES = [
    "manifest.json",
    "stage_a_turns.jsonl",
    "message_packets.jsonl",
    "belief_updates.jsonl",
    "final_predictions.jsonl",
    "metrics.json",
    "diagnostics.json",
    "progress.json",
    "sid_lite_report.md",
]


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """验证 SID-lite run 是否满足 smoke 实验契约。"""
    root = Path(run_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    manifest = _load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    stage_a_rows = _load_jsonl(root / "stage_a_turns.jsonl")
    packet_rows = _load_jsonl(root / "message_packets.jsonl")
    belief_rows = _load_jsonl(root / "belief_updates.jsonl")
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl")

    request_failures = sum(1 for row in stage_a_rows + belief_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in stage_a_rows + belief_rows if row.get("output_status") == "schema_fail")
    paired_check = _paired_design_check(manifest, prediction_rows)
    stage_hash_check = _shared_stage_hash_check(prediction_rows)
    early_exit_check = _early_exit_zero_comm_check(prediction_rows)
    packet_cap_check = _packet_cap_check(packet_rows)
    fail_open_check = _invalid_confidence_fail_open_check(prediction_rows)
    passed = (
        not missing
        and request_failures == 0
        and schema_failures == 0
        and paired_check["passed"]
        and stage_hash_check["passed"]
        and early_exit_check["passed"]
        and packet_cap_check["passed"]
        and fail_open_check["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "checks": {
            "request_failures_total": request_failures,
            "schema_failures_total": schema_failures,
            "paired_design_check": paired_check,
            "shared_stage_a_hash_check": stage_hash_check,
            "early_exit_zero_comm_check": early_exit_check,
            "packet_cap_check": packet_cap_check,
            "invalid_confidence_fail_open_check": fail_open_check,
        },
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _paired_design_check(manifest: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(manifest.get("methods") or ["mv_3", "always_full", "compression_only", "sid_lite"])
    by_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        by_sample[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    missing = [
        {"dataset": dataset, "sample_id": sample_id, "missing_methods": sorted(set(methods) - observed)}
        for (dataset, sample_id), observed in sorted(by_sample.items())
        if set(methods) - observed
    ]
    counts = Counter(row.get("method_name") for row in prediction_rows)
    expected_count = len(by_sample)
    count_mismatches = [
        {"method_name": method, "expected": expected_count, "observed": counts.get(method, 0)}
        for method in methods
        if counts.get(method, 0) != expected_count
    ]
    return {
        "passed": not missing and not count_mismatches and expected_count > 0,
        "sample_count": expected_count,
        "missing_methods": missing[:20],
        "count_mismatches": count_mismatches,
    }


def _shared_stage_hash_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("stage_a_trace_hash")))
    for (dataset, sample_id), hashes in sorted(grouped.items()):
        hashes.discard("")
        hashes.discard("None")
        if len(hashes) != 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "hashes": sorted(hashes)})
    return {"passed": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _early_exit_zero_comm_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in prediction_rows
        if row.get("method_name") == "sid_lite"
        and row.get("early_exit")
        and float(row.get("communication_tokens_per_question") or 0.0) != 0.0
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _packet_cap_check(packet_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in packet_rows
        if int(row.get("approx_packet_tokens") or 0) > int(row.get("token_cap") or 0)
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _invalid_confidence_fail_open_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    violations = [
        row
        for row in prediction_rows
        if row.get("method_name") == "sid_lite"
        and row.get("any_invalid_confidence")
        and row.get("early_exit")
    ]
    return {"passed": not violations, "violation_count": len(violations), "violations": _compact_rows(violations)}


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "method_name": row.get("method_name"),
            "agent_id": row.get("agent_id"),
        }
        for row in rows[:20]
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
