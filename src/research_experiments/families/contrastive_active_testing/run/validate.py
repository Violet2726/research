"""CATCH 运行产物与科学质量的分层校验。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    root = Path(run_dir)
    required = [
        root / "manifest.json",
        root / "views" / "metrics.json",
        root / "views" / "predictions.jsonl",
        root / "views" / "run_summary.json",
        root / "turns" / "agent_turns.jsonl",
        root / "turns" / "router_decisions.jsonl",
        root / "diagnostics" / "gate.json",
    ]
    artifact_violations = [f"missing:{path.relative_to(root).as_posix()}" for path in required if not path.exists()]
    scientific_violations: list[str] = []
    manifest: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    if (root / "manifest.json").exists():
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact_violations.append("invalid:manifest.json")
    for path, target, label in (
        (root / "turns" / "agent_turns.jsonl", turns, "agent_turns"),
        (root / "turns" / "preflight_turns.jsonl", turns, "preflight_turns"),
        (root / "views" / "predictions.jsonl", predictions, "predictions"),
    ):
        if not path.exists():
            continue
        try:
            target.extend(json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line)
        except (OSError, json.JSONDecodeError):
            artifact_violations.append(f"invalid:{label}")

    expected_namespace = str(manifest.get("cache_namespace") or "")
    predecessor_namespace = str(manifest.get("baseline_read_cache_namespace") or "")
    allowed = {
        "catch-dev-v1",
        "catch-heldout-v1",
        "catch-confirm-v1",
        "catch-dev-v2",
        "catch-heldout-v2",
        "catch-confirm-v2",
    }
    if expected_namespace not in allowed:
        artifact_violations.append("unexpected_cache_namespace")
    if manifest.get("request_source") not in {
        "fresh_catch_confirmation_cache",
        "role_aware_catch_v2_cache",
    }:
        artifact_violations.append("missing_request_source_declaration")
    permitted_turn_namespaces = {expected_namespace, predecessor_namespace} - {""}
    invalid_turns = [
        row
        for row in turns
        if row.get("cache_namespace") not in permitted_turn_namespaces
        or row.get("request_source") not in {
            "catch_confirmation_cache",
            "network",
            "active_cache",
            "predecessor_cache",
        }
        or not isinstance(row.get("payload"), dict)
        or not row.get("cache_key")
        or "raw_finish_reason" not in row
        or "network_attempt_count" not in row
    ]
    if invalid_turns:
        artifact_violations.append("turn_audit_contract_failed")
    if any(row.get("request_error") for row in turns):
        scientific_violations.append("terminal_request_failure")
    if turns and any(
        row.get("usage_source") != "reported"
        or row.get("actual_total_tokens") is None
        or row.get("reasoning_tokens") is None
        for row in turns
    ):
        scientific_violations.append("missing_actual_or_reasoning_tokens")
    if predictions and any(int(row.get("logical_calls_per_question") or 0) not in {5, 6, 8} for row in predictions):
        scientific_violations.append("logical_call_alignment_failed")
    if manifest.get("run_status") == "failed":
        scientific_violations.append(str(manifest.get("termination_reason") or "run_failed"))
    if (
        manifest.get("method_version") == "catch_v2"
        and manifest.get("phase_name") == "development"
        and manifest.get("run_status") == "completed"
    ):
        preflight_path = root / "diagnostics" / "preflight.json"
        if not preflight_path.exists():
            artifact_violations.append("missing:diagnostics/preflight.json")
        else:
            try:
                if not json.loads(preflight_path.read_text(encoding="utf-8")).get("passed"):
                    scientific_violations.append("structural_preflight_failed")
            except (OSError, json.JSONDecodeError):
                artifact_violations.append("invalid:diagnostics/preflight.json")
    gate_passed = None
    gate_payload: dict[str, Any] = {}
    gate_path = root / "diagnostics" / "gate.json"
    if gate_path.exists():
        try:
            gate_payload = json.loads(gate_path.read_text(encoding="utf-8"))
            gate_passed = bool(gate_payload.get("passed"))
        except (OSError, json.JSONDecodeError):
            artifact_violations.append("invalid:gate.json")
    return {
        "passed": not artifact_violations and not scientific_violations,
        "family_name": "contrastive_active_testing",
        "artifact_violations": sorted(set(artifact_violations)),
        "scientific_violations": sorted(set(scientific_violations)),
        "validator_exception": None,
        "counts": {
            "turns": len(turns),
            "predictions": len(predictions),
            "request_failures": sum(bool(row.get("request_error")) for row in turns),
            "planned_samples": gate_payload.get("planned_sample_count"),
            "completed_samples": gate_payload.get("completed_sample_count"),
            "incomplete_samples": gate_payload.get("incomplete_sample_count"),
        },
        "performance_gate_passed": gate_passed,
    }
