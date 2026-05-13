"""Cover `comm_necessary.runner` output-contract stabilization helpers."""

from __future__ import annotations

from research_experiments.families.comm_necessary.run.sample import _apply_belief_answer_fallback


def test_apply_belief_answer_fallback_keeps_previous_grounded_answer() -> None:
    row = {
        "output_status": "ok",
        "prediction": "",
        "normalized_answer": "",
        "final_answer_raw": "",
        "supporting_facts": [],
        "changed_answer": True,
        "validated_output": {
            "changed_answer": True,
            "final_answer": None,
            "supporting_facts": [],
        },
    }
    previous_row = {
        "normalized_answer": "chinese coffee",
        "final_answer_raw": "Chinese Coffee",
        "supporting_facts": [["Ira Lewis", 2]],
    }

    _apply_belief_answer_fallback(row, previous_row)

    assert row["changed_answer"] is False
    assert row["prediction"] == "chinese coffee"
    assert row["normalized_answer"] == "chinese coffee"
    assert row["final_answer_raw"] == "Chinese Coffee"
    assert row["supporting_facts"] == [["Ira Lewis", 2]]
    assert row["belief_fallback"] == "kept_previous_answer_after_empty_belief_output"
    assert row["validated_output"]["final_answer"] == "Chinese Coffee"


def test_apply_belief_answer_fallback_preserves_abstention_without_prior_answer() -> None:
    row = {
        "output_status": "ok",
        "prediction": "",
        "normalized_answer": "",
        "final_answer_raw": "",
        "supporting_facts": [],
        "changed_answer": True,
        "validated_output": {
            "changed_answer": True,
            "final_answer": None,
            "supporting_facts": [],
        },
    }
    previous_row = {
        "normalized_answer": "",
        "final_answer_raw": "",
        "supporting_facts": [],
    }

    _apply_belief_answer_fallback(row, previous_row)

    assert row["changed_answer"] is False
    assert row["prediction"] == ""
    assert row["normalized_answer"] == ""
    assert row["belief_fallback"] == "abstained_without_prior_answer"
