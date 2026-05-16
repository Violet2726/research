from __future__ import annotations

from research_experiments.families.econ.algorithms import (
    ACTION_ORDER,
    apply_belief_answer_safeguard,
    build_belief_state,
)
from research_experiments.families.econ.config import load_experiment_config, load_protocol_config
from research_experiments.families.econ.run.sample import _build_metrics


def test_load_econ_experiment_config_reads_methods_and_protocol() -> None:
    experiment = load_experiment_config("configs/families/econ/experiments/econ_same_context_main.toml")
    protocol = load_protocol_config(experiment.protocol)

    assert experiment.name == "econ_same_context_main"
    assert [method.name for method in experiment.methods] == [
        "single_agent_cot",
        "vote_mv3",
        "econ_full_comm_r1",
        "econ_bne_main",
    ]
    assert protocol.agent_count == 3
    assert protocol.peer_packet_token_cap == 120


def test_build_belief_state_scores_finite_action_space() -> None:
    protocol = load_protocol_config("configs/families/econ/protocols/paper_main.toml")
    state = build_belief_state(
        [
            {
                "agent_id": 1,
                "normalized_answer": "5",
                "confidence_value": 0.55,
                "confidence_valid": True,
                "claim_span": "The subtotal is five.",
                "key_evidence": "I counted five rows.",
                "reasoning_trace": "Count rows.",
                "final_answer": "5",
                "keyword_clues": ["count", "rows"],
            },
            {
                "agent_id": 2,
                "normalized_answer": "4",
                "confidence_value": 0.85,
                "confidence_valid": True,
                "claim_span": "The subtotal is four.",
                "key_evidence": "One row is not valid.",
                "reasoning_trace": "Exclude the placeholder row.",
                "final_answer": "4",
                "keyword_clues": ["exclude", "placeholder"],
            },
            {
                "agent_id": 3,
                "normalized_answer": "4",
                "confidence_value": 0.72,
                "confidence_valid": True,
                "claim_span": "The answer is four.",
                "key_evidence": "Only four rows satisfy the condition.",
                "reasoning_trace": "Check the condition row by row.",
                "final_answer": "4",
                "keyword_clues": ["condition", "rows"],
            },
        ],
        protocol=protocol,
    )

    assert state["selected_action"] in {"adopt_vote", "query_best_peer", "query_two_peers"}
    assert [row["action"] for row in state["action_scores"]] == list(ACTION_ORDER)
    assert state["expected_gain"] > 0.0


def test_apply_belief_answer_safeguard_keeps_previous_answer_without_peer_packets() -> None:
    updated = apply_belief_answer_safeguard(
        stage_a_row={
            "agent_id": 1,
            "normalized_answer": "4",
            "confidence_value": 0.8,
            "confidence_valid": True,
            "claim_span": "The answer is four.",
            "key_evidence": "Only four rows qualify.",
            "reasoning_trace": "Count qualified rows.",
            "keyword_clues": ["count"],
        },
        belief_row={
            "output_status": "ok",
            "validated_output": {
                "changed_answer": True,
                "new_answer": "5",
                "confidence_delta": -0.2,
                "reason_for_change": "peer says five",
                "remaining_disagreement": "",
            },
        },
        selected_peer_packets=[],
    )

    assert updated["normalized_answer"] == "4"
    assert updated["changed_answer"] is False


def test_build_metrics_reports_gain_and_action_mix() -> None:
    methods = load_experiment_config("configs/families/econ/experiments/econ_same_context_main.toml").methods
    metrics = _build_metrics(
        [
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "vote_mv3",
                "score": 0.0,
                "prompt_tokens_per_question": 60.0,
                "completion_tokens_per_question": 30.0,
                "total_tokens_per_question": 90.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "selected_action": "adopt_vote",
                "belief_score": None,
                "expected_gain": None,
                "communication_cost": 0.0,
                "changed_after_coordination": False,
                "correction_flag": False,
                "degradation_flag": False,
                "selected_peer_count_mean": 0.0,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "econ_full_comm_r1",
                "score": 1.0,
                "prompt_tokens_per_question": 80.0,
                "completion_tokens_per_question": 40.0,
                "total_tokens_per_question": 120.0,
                "communication_tokens_per_question": 30.0,
                "latency_ms_per_question": 20.0,
                "calls_per_question": 6,
                "selected_action": "query_all_peers",
                "belief_score": None,
                "expected_gain": None,
                "communication_cost": None,
                "changed_after_coordination": True,
                "correction_flag": True,
                "degradation_flag": False,
                "selected_peer_count_mean": 2.0,
            },
            {
                "dataset": "gsm8k",
                "model_name": "xiaomimimo/mimo-v2.5",
                "method_name": "econ_bne_main",
                "score": 1.0,
                "prompt_tokens_per_question": 70.0,
                "completion_tokens_per_question": 35.0,
                "total_tokens_per_question": 105.0,
                "communication_tokens_per_question": 15.0,
                "latency_ms_per_question": 15.0,
                "calls_per_question": 4,
                "selected_action": "query_best_peer",
                "belief_score": 0.42,
                "expected_gain": 0.35,
                "communication_cost": 0.18,
                "changed_after_coordination": True,
                "correction_flag": True,
                "degradation_flag": False,
                "selected_peer_count_mean": 1.0,
            },
        ],
        methods,
        model_name="xiaomimimo/mimo-v2.5",
    )

    bne_row = next(row for row in metrics["summary"] if row["dataset"] == "gsm8k" and row["method_name"] == "econ_bne_main")
    overall_vote = next(row for row in metrics["summary"] if row["dataset"] == "overall" and row["method_name"] == "vote_mv3")

    assert bne_row["gain_over_vote_mv3"] == 1.0
    assert bne_row["query_best_peer_rate"] == 1.0
    assert overall_vote["accuracy_mean"] == 0.0

