"""CATCH 运行的非阻断完整性摘要。

Validation is deliberately descriptive.  Scientific quality, missing requests,
and partial samples are reported as warnings and never used to authorize another
phase or turn a completed experiment into a process failure.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CORE_PATHS = (
    "manifest.json",
    "progress.json",
    "turns/agent_turns.jsonl",
    "turns/router_decisions.jsonl",
    "views/predictions.jsonl",
    "views/metrics.json",
    "views/run_summary.json",
    "report.md",
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Summarize readability, coverage, and recoverable execution problems."""

    root = Path(run_dir)
    artifact_errors: list[str] = []
    warnings: list[str] = []
    payloads: dict[str, Any] = {}
    rows: dict[str, list[dict[str, Any]]] = {
        "turns": [],
        "routers": [],
        "predictions": [],
    }

    for relative in CORE_PATHS:
        path = root / relative
        if not path.exists():
            artifact_errors.append(f"missing:{relative}")

    for relative, name in (
        ("manifest.json", "manifest"),
        ("progress.json", "progress"),
        ("views/metrics.json", "metrics"),
        ("views/run_summary.json", "summary"),
    ):
        path = root / relative
        if not path.exists():
            continue
        try:
            payloads[name] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError):
            artifact_errors.append(f"invalid:{relative}")

    for relative, name in (
        ("turns/agent_turns.jsonl", "turns"),
        ("turns/router_decisions.jsonl", "routers"),
        ("views/predictions.jsonl", "predictions"),
    ):
        path = root / relative
        if not path.exists():
            continue
        try:
            rows[name] = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if not all(isinstance(row, dict) for row in rows[name]):
                raise TypeError("JSONL row must be an object")
        except (OSError, json.JSONDecodeError, TypeError):
            artifact_errors.append(f"invalid:{relative}")

    manifest = payloads.get("manifest") or {}
    progress = payloads.get("progress") or {}
    turns = rows["turns"]
    routers = rows["routers"]
    predictions = rows["predictions"]
    sample_errors = list((payloads.get("summary") or {}).get("sample_errors") or [])
    dataset_errors = list((payloads.get("summary") or {}).get("dataset_errors") or [])
    request_failures = [row for row in turns if row.get("request_error")]
    parse_failures = [row for row in turns if row.get("protocol_parse_status") == "failed"]

    if request_failures:
        warnings.append(f"request_failures:{len(request_failures)}")
    if parse_failures:
        warnings.append(f"parse_failures:{len(parse_failures)}")
    if sample_errors:
        warnings.append(f"sample_errors:{len(sample_errors)}")
    if dataset_errors:
        warnings.append(f"dataset_errors:{len(dataset_errors)}")
    warnings.extend(str(item) for item in manifest.get("execution_warnings") or [])

    configured_limit = int(manifest.get("max_network_attempts") or 0)
    actual_attempts = sum(int(row.get("network_attempt_count") or 0) for row in turns)
    if configured_limit and actual_attempts > configured_limit:
        warnings.append(f"network_attempt_soft_limit_exceeded:{actual_attempts - configured_limit}")

    planned = int(
        (payloads.get("summary") or {}).get("planned_sample_count")
        or manifest.get("planned_sample_count")
        or manifest.get("sample_count")
        or 0
    )
    summary_execution = (payloads.get("summary") or {}).get("execution") or {}
    completed_ids = {
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in routers
        if row.get("sample_id")
    }
    status = str(manifest.get("run_status") or progress.get("status") or "unknown")
    completed_sample_count = int(
        summary_execution.get("attempted_sample_count")
        if summary_execution.get("attempted_sample_count") is not None
        else len(completed_ids)
    )
    artifact_valid = not artifact_errors
    return {
        "passed": artifact_valid,
        "artifact_valid": artifact_valid,
        "run_status": status,
        "scientific_gate_applicable": False,
        "scientific_gate_passed": None,
        "performance_gate_passed": None,
        "artifact_errors": sorted(set(artifact_errors)),
        "artifact_violations": sorted(set(artifact_errors)),
        "warnings": sorted(set(warnings)),
        "scientific_violations": [],
        "validator_exception": None,
        "archive_integrity": {"applicable": False, "passed": None},
        "termination_reason": str(
            manifest.get("termination_reason") or progress.get("termination_reason") or ""
        ),
        "counts": {
            "turns": len(turns),
            "routers": len(routers),
            "predictions": len(predictions),
            "request_failures": len(request_failures),
            "parse_failures": len(parse_failures),
            "sample_errors": len(sample_errors),
            "dataset_errors": len(dataset_errors),
            "logical_calls": len(turns),
            "cached_logical_calls": sum(bool(row.get("cache_hit")) for row in turns),
            "physical_network_attempts": actual_attempts,
            "retry_attempts": sum(
                max(0, int(row.get("network_attempt_count") or 0) - 1)
                for row in turns
                if int(row.get("network_attempt_count") or 0) > 0
            ),
            "actual_total_tokens": sum(float(row.get("actual_total_tokens") or 0) for row in turns),
            "planned_samples": planned,
            "completed_samples": completed_sample_count,
            "incomplete_samples": max(0, planned - completed_sample_count),
        },
    }
