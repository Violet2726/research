"""CATCH 运行产物与科学质量的分层校验。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from research_experiments.families.contrastive_active_testing.artifact_replay import (
    audit_v3_artifact_recomputation,
)


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
    structural_preflight_passed = False
    manifest: dict[str, Any] = {}
    turns: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    routers: list[dict[str, Any]] = []
    if (root / "manifest.json").exists():
        try:
            manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            artifact_violations.append("invalid:manifest.json")
    for path, target, label in (
        (root / "turns" / "agent_turns.jsonl", turns, "agent_turns"),
        (root / "turns" / "preflight_turns.jsonl", turns, "preflight_turns"),
        (root / "turns" / "router_decisions.jsonl", routers, "router_decisions"),
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
        "catch-dev-v3",
        "catch-heldout-v3",
        "catch-confirm-v3",
    }
    if expected_namespace not in allowed:
        artifact_violations.append("unexpected_cache_namespace")
    if manifest.get("request_source") not in {
        "fresh_catch_confirmation_cache",
        "role_aware_catch_v2_cache",
        "role_aware_versioned_catch_cache",
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
    artifact_recomputation = None
    if manifest.get("method_version") == "catch_v3" and predictions:
        if not routers or any(row.get("protocol_version") != "catch_v3" for row in routers):
            artifact_recomputation = {
                "passed": False,
                "audited_sample_count": 0,
                "violations": [{"reason": "missing_or_non_v3_router_contract"}],
            }
        else:
            artifact_recomputation = audit_v3_artifact_recomputation(
                turns=turns,
                routers=routers,
                predictions=predictions,
                seed=int(manifest.get("global_seed") or 42),
            )
        if not artifact_recomputation.get("passed"):
            artifact_violations.append("v3_artifact_recomputation_failed")
    if (
        manifest.get("method_version") == "catch_v3"
        and manifest.get("phase_name") == "development"
        and manifest.get("run_mode") == "full"
    ):
        archived_audit_path = root / "diagnostics" / "preflight_human_audit.json"
        expected_audit_sha = str(
            (((manifest.get("preflight_dependency") or {}).get("human_audit") or {}).get("sha256"))
            or ""
        )
        if not archived_audit_path.exists():
            artifact_violations.append("missing:diagnostics/preflight_human_audit.json")
        elif (
            not expected_audit_sha
            or _sha256_file(archived_audit_path) != expected_audit_sha
        ):
            artifact_violations.append("preflight_human_audit_archive_hash_mismatch")
    if manifest.get("run_status") == "failed":
        scientific_violations.append(str(manifest.get("termination_reason") or "run_failed"))
    if (
        manifest.get("method_version") in {"catch_v2", "catch_v3"}
        and manifest.get("phase_name") == "development"
        and manifest.get("run_status") == "completed"
        and (
            manifest.get("method_version") == "catch_v2"
            or manifest.get("run_mode") == "structural_preflight"
        )
    ):
        preflight_path = root / "diagnostics" / "preflight.json"
        if not preflight_path.exists():
            artifact_violations.append("missing:diagnostics/preflight.json")
        else:
            try:
                structural_preflight_passed = bool(
                    json.loads(preflight_path.read_text(encoding="utf-8")).get("passed")
                )
                if not structural_preflight_passed:
                    scientific_violations.append("structural_preflight_failed")
            except (OSError, json.JSONDecodeError):
                artifact_violations.append("invalid:diagnostics/preflight.json")
    if manifest.get("method_version") == "catch_v3" and manifest.get("run_mode") == "structural_preflight":
        replay_path = root / "diagnostics" / "canonicalization_replay.json"
        if not replay_path.exists():
            artifact_violations.append("missing:diagnostics/canonicalization_replay.json")
        else:
            try:
                replay = json.loads(replay_path.read_text(encoding="utf-8"))
                if replay.get("network_requests") != 0 or not replay.get("passed"):
                    scientific_violations.append("canonicalization_replay_headroom_failed")
            except (OSError, json.JSONDecodeError):
                artifact_violations.append("invalid:diagnostics/canonicalization_replay.json")
        audit_sample_path = root / "diagnostics" / "preflight_human_audit_sample.json"
        if not audit_sample_path.exists():
            artifact_violations.append("missing:diagnostics/preflight_human_audit_sample.json")
        else:
            try:
                audit_sample = json.loads(audit_sample_path.read_text(encoding="utf-8"))
                items = audit_sample.get("items")
                hashes = {
                    str(item.get("coordinate_sha256") or "")
                    for item in items or []
                    if isinstance(item, dict) and item.get("coordinate_sha256")
                }
                if (
                    audit_sample.get("audit_version")
                    != "catch_v3_icv_blind_coordinate_audit_v1"
                    or audit_sample.get("source_preflight_run_id") != manifest.get("run_id")
                    or audit_sample.get("source_config_sha256")
                    != manifest.get("frozen_config_sha256")
                    or not isinstance(items, list)
                    or len(hashes) != len(items or [])
                    or (structural_preflight_passed and len(items) != 40)
                ):
                    artifact_violations.append("preflight_human_audit_sample_contract_failed")
            except (OSError, json.JSONDecodeError):
                artifact_violations.append("invalid:diagnostics/preflight_human_audit_sample.json")
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
        "artifact_recomputation": artifact_recomputation,
    }


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
