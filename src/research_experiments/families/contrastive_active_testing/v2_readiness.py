"""生成 CATCH-Cert v2 的非阻断式科研就绪度诊断。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def write_cert_v2_readiness_assessment(
    run_dir: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    validation = json.loads((root / "run_validation.json").read_text(encoding="utf-8"))
    routers = _read_jsonl(root / "turns" / "router_decisions.jsonl")
    predictions = _read_jsonl(root / "views" / "predictions.jsonl")
    audit = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    if manifest.get("protocol_version") != "catch_cert_v2" or manifest.get("phase_name") != "development":
        raise ValueError("Readiness can be assessed only from a CATCH-Cert v2 development run.")

    triggered = [row for row in routers if row.get("triggered")]
    cert_rows = [row for row in predictions if row.get("method_name") == "catch_cert_v2"]
    sc_rows = {str(row.get("sample_id")): row for row in predictions if row.get("method_name") == "sc_5"}
    eligible_recoverable = [
        row
        for row in triggered
        if row.get("target_oracle_correct")
        and row.get("gold_candidate_key") != row.get("anchor_key")
        and row.get("gold_candidate_key") in set(row.get("eligible_challengers") or [])
    ]
    recoverable = [
        row
        for row in triggered
        if row.get("target_oracle_correct") and row.get("gold_candidate_key") != row.get("anchor_key")
    ]
    expected_verifier_rows = sum(
        int(panel.get("expected_test_count") or 0) for row in triggered for panel in row.get("verifier_panels") or []
    )
    valid_verifier_rows = sum(
        int(panel.get("valid_test_count") or 0) for row in triggered for panel in row.get("verifier_panels") or []
    )
    corrected = sum(bool(row.get("corrected_by_debate")) for row in cert_rows)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in cert_rows)
    headroom = sum(
        bool(row.get("target_oracle_correct")) and float(row.get("initial_vote_score") or 0) < 1.0 for row in cert_rows
    )
    sc_correct_risk = sum(float(row.get("initial_vote_score") or 0) == 1.0 for row in cert_rows)
    total_cert_tokens = sum(float(row.get("total_tokens_per_question") or 0) for row in cert_rows)
    total_sc_tokens = sum(float(row.get("total_tokens_per_question") or 0) for row in sc_rows.values())
    cert_correct = sum(float(row.get("score") or 0) for row in cert_rows)
    sc_correct = sum(float(row.get("score") or 0) for row in sc_rows.values())
    cert_efficiency = _ratio(cert_correct * 1_000, total_cert_tokens)
    sc_efficiency = _ratio(sc_correct * 1_000, total_sc_tokens)
    turns = _read_jsonl(root / "turns" / "agent_turns.jsonl")
    designer_turns = [row for row in turns if row.get("role") == "certificate_designer_v2"]
    seq_stage_turns = [row for row in turns if row.get("dataset") == "seqbench" and row.get("role") == "stage_a_solver"]
    false_passes = sum(bool(row.get("override_accepted")) and float(row.get("score") or 0) < 1.0 for row in cert_rows)
    conditions = {
        "artifact_valid": validation.get("artifact_valid") is True,
        "development_completed": str(manifest.get("run_status") or "").startswith("completed"),
        "two_reviewer_audit_complete": _audit_complete(audit),
        "seqbench_executor_golden_tests_passed": audit.get("seqbench_executor_golden_tests_passed") is True,
        "seqbench_stage_request_success_at_least_99_percent": not seq_stage_turns
        or _ratio(sum(not row.get("request_error") for row in seq_stage_turns), len(seq_stage_turns)) >= 0.99,
        "seqbench_stage_format_validity_at_least_99_percent": not seq_stage_turns
        or _ratio(sum(row.get("protocol_parse_status") == "ok" for row in seq_stage_turns), len(seq_stage_turns))
        >= 0.99,
        "answer_node_semantic_coverage_100_percent": all(
            len(row.get("candidate_answer_nodes") or {}) == int(row.get("candidate_count") or 0) for row in triggered
        ),
        "designer_schema_validity_at_least_98_percent": _ratio(
            sum(row.get("protocol_parse_status") == "ok" for row in designer_turns), len(designer_turns)
        )
        >= 0.98,
        "verifier_complete_result_rate_at_least_95_percent": _ratio(valid_verifier_rows, expected_verifier_rows)
        >= 0.95,
        "valid_certificate_case_coverage_at_least_80_percent": _ratio(
            sum(bool(row.get("eligible_challengers")) for row in triggered), len(triggered)
        )
        >= 0.80,
        "correct_challenger_certificate_recall_at_least_60_percent": _ratio(len(eligible_recoverable), len(recoverable))
        >= 0.60,
        "mandatory_obligation_coverage_at_least_80_percent": _ratio(
            sum(float(row.get("obligation_coverage") or 0) for row in triggered), len(triggered)
        )
        >= 0.80,
        "known_protocol_false_pass_count_zero": false_passes == 0,
        "net_headroom_utilization_at_least_15_percent": _ratio(corrected - harmed, headroom) >= 0.15,
        "sc_correct_harm_rate_at_most_5_percent": _ratio(harmed, sc_correct_risk) <= 0.05,
        "wrong_to_correct_exceeds_correct_to_wrong": corrected > harmed,
        "correct_per_1000_tokens_at_least_80_percent_of_sc5": sc_efficiency == 0
        or cert_efficiency >= 0.80 * sc_efficiency,
    }
    unmet_conditions = [name for name, met in conditions.items() if not met]
    all_recommended_conditions_met = not unmet_conditions
    payload = {
        "schema_version": "catch_cert_v2_readiness_assessment_v2",
        "protocol_version": "catch_cert_v2",
        "source_run_id": manifest.get("run_id"),
        "source_config_sha256": manifest.get("frozen_config_sha256"),
        "enforcement": "advisory_only",
        "blocks_execution": False,
        "conditions": conditions,
        "all_recommended_conditions_met": all_recommended_conditions_met,
        "unmet_conditions": unmet_conditions,
        "recommended_interpretation": (
            "confirmation_candidate" if all_recommended_conditions_met else "exploratory_diagnostic_evidence"
        ),
        "evidence": {
            "triggered_count": len(triggered),
            "recoverable_count": len(recoverable),
            "correct_challenger_eligible_count": len(eligible_recoverable),
            "corrected": corrected,
            "harmed": harmed,
            "headroom": headroom,
            "false_passes": false_passes,
            "cert_correct_per_1000_tokens": cert_efficiency,
            "sc5_correct_per_1000_tokens": sc_efficiency,
        },
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def freeze_cert_v2_readiness(
    run_dir: str | Path,
    audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    """兼容旧命令名称；只生成诊断，不冻结或阻止后续实验。"""

    return write_cert_v2_readiness_assessment(run_dir, audit_path, output_path)


def _audit_complete(audit: dict[str, Any]) -> bool:
    cases = audit.get("cases")
    if not isinstance(cases, list) or len(cases) != 120:
        return False
    for case in cases:
        first = case.get("reviewer_1") or {}
        second = case.get("reviewer_2") or {}
        if any(
            review.get("certificate_necessary") is None or review.get("certificate_sufficient") is None
            for review in (first, second)
        ):
            return False
        if first != second:
            adjudicated = case.get("adjudicated") or {}
            if adjudicated.get("certificate_necessary") is None or adjudicated.get("certificate_sufficient") is None:
                return False
    return True


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _ratio(numerator: float, denominator: float) -> float:
    return float(numerator) / float(denominator) if denominator else 0.0
