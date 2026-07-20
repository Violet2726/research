from __future__ import annotations

from research_experiments.families.contrastive_active_testing.v1_failure_audit import (
    ERROR_LABELS,
    build_v1_mechanism_audit,
    render_v1_mechanism_audit_markdown,
)


def _router(sample_id: str, *, anchor: str, gold: str | None, oracle: bool, target: bool) -> dict[str, object]:
    return {
        "protocol_version": "catch_cert_v1",
        "triggered": True,
        "dataset": "bbeh",
        "sample_id": sample_id,
        "task": "dyck_languages",
        "anchor_key": anchor,
        "gold_candidate_key": gold,
        "candidate_oracle_correct": oracle,
        "target_oracle_correct": target,
        "candidate_public_to_answer_class_key": {"H0": anchor, "H1": gold or "other"},
        "public_pairs": [{"pair_id": "P0", "left_candidate": "H0", "right_candidate": "H1"}],
        "claim_graphs": {
            "H0": {"nodes": [{"normalized_value": f"answer {anchor}"}]},
            "H1": {"nodes": [{"normalized_value": f"answer {gold or 'other'}"}]},
        },
        "eligible_challengers": [gold] if gold and gold != anchor else [],
        "audit_source_question": "Find the first step that is wrong.",
        "certificate_tests": [],
        "certificates": [],
        "verifier_panels": [],
        "decision": {"resolver": "no_certificate", "override_accepted": False},
    }


def test_v1_audit_builds_development_only_double_annotation_queue() -> None:
    routers = [
        _router("recoverable", anchor="31", gold="11", oracle=True, target=True),
        _router("risk", anchor="11", gold="11", oracle=True, target=True),
        _router("unrecoverable", anchor="31", gold=None, oracle=False, target=False),
    ]
    predictions = [
        {"sample_id": row["sample_id"], "method_name": "catch_cert", "score": int(row["sample_id"] == "risk")}
        for row in routers
    ]
    payload = build_v1_mechanism_audit(routers=routers, predictions=predictions)
    assert payload["selection_is_gold_after_run_development_only"] is True
    assert set(payload["error_label_vocabulary"]) == set(ERROR_LABELS)
    assert payload["actual_counts"] == {
        "recoverable_wrong": 1,
        "sc_correct_risk": 1,
        "candidate_oracle_failure": 1,
    }
    assert all(case["reviewer_1"]["certificate_sufficient"] is None for case in payload["cases"])
    assert "双人标注" in render_v1_mechanism_audit_markdown(payload)
