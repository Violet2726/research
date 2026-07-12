import json

import pytest

from research_experiments.families.risk_controlled_trace_mad.run.sample import _parse_audit, _parse_selector


def test_selector_rejects_anchor_and_novel_labels() -> None:
    mapping = {"A": "1", "B": "2"}
    assert (
        _parse_selector(json.dumps({"challenger_label": "B", "decisive_difference": "check"}), mapping, "1")[
            "challenger_answer"
        ]
        == "2"
    )
    with pytest.raises(ValueError):
        _parse_selector(json.dumps({"challenger_label": "A", "decisive_difference": "check"}), mapping, "1")
    with pytest.raises(ValueError):
        _parse_selector(json.dumps({"challenger_label": "C", "decisive_difference": "check"}), mapping, "1")


def test_audit_maps_random_labels_back_to_existing_answers() -> None:
    raw = json.dumps(
        {
            "preferred_label": "B",
            "decisive_claim": "2 + 3 = 5",
            "evidence": [
                {
                    "target_label": "B",
                    "claim_kind": "support",
                    "test_type": "arithmetic",
                    "payload": {"left": "2+3", "right": "5", "relation": "eq"},
                }
            ],
        }
    )
    result = _parse_audit(raw, {"A": "4", "B": "5"}, "Compute 2 + 3.")
    assert result["preferred_answer"] == "5"
    assert result["evidence_results"][0]["status"] == "pass"
