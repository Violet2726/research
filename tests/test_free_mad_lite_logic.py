from __future__ import annotations

from free_mad_lite.logic import build_trajectory_decision, deterministic_trajectory_fallback


def test_free_mad_lite_trajectory_fallback_uses_anti_majority() -> None:
    initial_rows = [
        {"agent_id": 1, "normalized_answer": "a"},
        {"agent_id": 2, "normalized_answer": "b"},
        {"agent_id": 3, "normalized_answer": "b"},
    ]
    anti_rows = [
        {"agent_id": 1, "normalized_answer": "c"},
        {"agent_id": 2, "normalized_answer": "c"},
        {"agent_id": 3, "normalized_answer": "b"},
    ]
    decision = deterministic_trajectory_fallback(initial_rows, anti_rows)
    assert decision.fallback_used is True
    assert decision.final_answer == "c"
    assert decision.selected_agent_id == 1


def test_free_mad_lite_valid_judge_output_avoids_fallback() -> None:
    judge_row = {
        "output_status": "ok",
        "validated_output": {
            "final_answer": "yes",
            "selected_agent_id": 2,
            "rationale": "best evidence",
        },
    }
    decision = build_trajectory_decision(judge_row, [], [])
    assert decision.fallback_used is False
    assert decision.final_answer == "yes"
    assert decision.selected_agent_id == 2
