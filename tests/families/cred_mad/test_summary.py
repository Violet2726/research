from __future__ import annotations

from research_experiments.families.cred_mad.run.sample import _summarize_prediction_rows


def test_summary_preserves_zero_initial_vote_score_for_debate_gain() -> None:
    rows = [
        {
            "method_type": "mad",
            "score": 1.0,
            "initial_vote_score": 0.0,
            "total_tokens_per_question": 100.0,
            "prompt_tokens_per_question": 40.0,
            "completion_tokens_per_question": 60.0,
            "debate_total_tokens_per_question": 30.0,
            "latency_ms_per_question": 1.0,
            "calls_per_question": 3,
            "protocol_failures_per_question": 0,
            "reason_missing_turns_per_question": 0,
            "debate_rounds": 1,
            "agent_count": 5,
            "triggered": True,
            "corrected_by_debate": True,
            "harmed_by_debate": False,
            "vote_flipped": True,
            "initial_consensus": False,
            "final_consensus": True,
        }
    ]

    summary = _summarize_prediction_rows(
        rows,
        dataset="math500",
        model_name="model",
        method_name="cred_refute_queue_v1",
        aggregate_kind="dataset",
    )

    assert summary["accuracy_mean"] == 1.0
    assert summary["initial_vote_accuracy_mean"] == 0.0
    assert summary["debate_gain_over_initial_vote"] == 1.0
