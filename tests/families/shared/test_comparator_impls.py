from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.family_runtime.comparator_impls import (
    build_shared_vanilla_mad_prediction,
    build_stage_a_mv3_prediction,
    run_shared_vanilla_mad_rounds,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONTROLLED_PROMPT_VERSION,
    build_debate_messages,
    build_initial_messages,
    prompt_version_uses_json_response_format,
)


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="gsm8k",
        sample_id="gsm8k-1",
        question="What is 2 + 2?",
        reference_answer="4",
        prompt_context="",
        metadata={},
    )


def test_build_stage_a_mv3_prediction_uses_stage_a_vote_and_zero_comm_cost() -> None:
    turn_rows = [
        {"prompt_tokens": 10.0, "completion_tokens": 2.0, "total_tokens": 12.0, "latency_ms": 100.0},
        {"prompt_tokens": 11.0, "completion_tokens": 3.0, "total_tokens": 14.0, "latency_ms": 110.0},
        {"prompt_tokens": 12.0, "completion_tokens": 4.0, "total_tokens": 16.0, "latency_ms": 120.0},
    ]
    row = build_stage_a_mv3_prediction(
        run_id="run",
        dataset="gsm8k",
        split_name="count20",
        sample=_sample(),
        question_preview="What is 2 + 2?",
        model_name="demo",
        stage_a_turns=turn_rows,
        stage_a_vote="4",
        stage_a_score=1.0,
        stage_a_trace_hash="trace",
        vote_counts={"4": 3},
        method_kind="baseline",
    )
    assert row["method_name"] == "mv_3"
    assert row["prediction"] == "4"
    assert row["communication_tokens_per_question"] == 0.0
    assert row["calls_per_question"] == 3.0


def test_run_shared_vanilla_mad_rounds_and_prediction_row() -> None:
    sample = _sample()
    responses = {
        (0, 1): "4",
        (0, 2): "4",
        (0, 3): "5",
        (1, 1): "4",
        (1, 2): "4",
        (1, 3): "4",
    }

    def execute_turn(**kwargs):
        round_index = kwargs["round_index"]
        agent_id = kwargs["agent_id"]
        answer = responses[(round_index, agent_id)]
        return {
            "agent_id": agent_id,
            "round_index": round_index,
            "role": kwargs["role"],
            "prompt_tokens": 10.0,
            "completion_tokens": 2.0,
            "total_tokens": 12.0,
            "latency_ms": 100.0,
            "normalized_answer": answer,
            "validated_output": {"final_answer": answer, "reasoning": "ok"},
        }

    result = run_shared_vanilla_mad_rounds(
        sample=sample,
        run_id="run",
        dataset="gsm8k",
        split_name="count20",
        method_name="mad_3a_r1",
        agent_count=3,
        debate_rounds=1,
        initial_temperature=0.7,
        debate_temperature=0.7,
        top_p=1.0,
        global_seed=42,
        prompt_version=CONTROLLED_PROMPT_VERSION,
        execute_turn=execute_turn,
        build_debate_row=lambda sender, recipient_id, round_index: {
            "sender": sender["agent_id"],
            "recipient": recipient_id,
            "round_index": round_index,
        },
    )
    row = build_shared_vanilla_mad_prediction(
        run_id="run",
        dataset="gsm8k",
        split_name="count20",
        sample=sample,
        method_name="mad_3a_r1",
        method_type="mad",
        model_name="demo",
        result=result,
    )
    assert row["initial_vote_prediction"] == "4"
    assert row["final_vote_prediction"] == "4"
    assert row["debate_rounds"] == 1
    assert row["corrected_by_debate"] is False


def test_run_shared_vanilla_mad_rounds_rejects_unsupported_prompt_version() -> None:
    sample = _sample()

    def execute_turn(**kwargs):
        return {
            "agent_id": kwargs["agent_id"],
            "round_index": kwargs["round_index"],
            "role": kwargs["role"],
            "prompt_tokens": 1.0,
            "completion_tokens": 1.0,
            "total_tokens": 2.0,
            "latency_ms": 1.0,
            "normalized_answer": "4",
            "validated_output": {"final_answer": "4", "reasoning": "ok"},
        }

    try:
        run_shared_vanilla_mad_rounds(
            sample=sample,
            run_id="run",
            dataset="gsm8k",
            split_name="count20",
            method_name="mad_3a_r1",
            agent_count=3,
            debate_rounds=1,
            initial_temperature=0.7,
            debate_temperature=0.7,
            top_p=1.0,
            global_seed=42,
            prompt_version="imad_controlled_json",
            execute_turn=execute_turn,
            build_debate_row=lambda sender, recipient_id, round_index: {},
        )
    except ValueError as exc:
        assert CONTROLLED_PROMPT_VERSION in str(exc)
    else:
        raise AssertionError("Expected shared vanilla MAD core to reject non-controlled prompt version.")


def test_consistent_json_prompt_builder_requires_anchor_fields() -> None:
    sample = DatasetSample(
        dataset="gpqa_diamond",
        sample_id="gpqa-1",
        question="Which option is correct?",
        reference_answer="A|||Alpha",
        prompt_context="Options:\nA. Alpha\nB. Beta\nC. Gamma\nD. Delta",
        metadata={},
    )
    initial_messages = build_initial_messages(sample, agent_id=1, prompt_version=CONTROLLED_PROMPT_VERSION)
    debate_messages = build_debate_messages(
        sample,
        agent_id=1,
        round_index=1,
        previous_reasoning="Option A best fits the evidence.",
        previous_answer="A",
        peer_messages=[
            {
                "agent": "agent_2",
                "answer": "B",
                "reasoning": "Different rationale.",
            }
        ],
        prompt_version=CONTROLLED_PROMPT_VERSION,
    )

    assert "keys reasoning and final_answer" in initial_messages[0]["content"]
    assert "single best option" in initial_messages[1]["content"]
    assert "Peer feedback:" in debate_messages[1]["content"]
    assert "agent_2 previous final_answer: B" in debate_messages[1]["content"]


def test_prompt_version_response_format_is_always_enabled_for_consistent_json() -> None:
    assert prompt_version_uses_json_response_format(CONTROLLED_PROMPT_VERSION) is True
