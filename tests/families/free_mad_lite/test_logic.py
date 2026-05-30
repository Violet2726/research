"""覆盖 Free-MAD-lite 逻辑与聚合行为的测试。"""

from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.free_mad_lite.algorithms import (
    build_trajectory_decision,
    deterministic_trajectory_fallback,
)
from research_experiments.families.free_mad_lite.prompts import (
    build_debate_messages,
    build_initial_messages,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    CONTROLLED_PROMPT_VERSION,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    build_debate_messages as build_standard_mad_debate_messages,
)
from research_experiments.family_runtime.vanilla_mad_prompting import (
    build_initial_messages as build_standard_mad_initial_messages,
)


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="gsm8k",
        sample_id="test-001",
        question="What is 2 + 2?",
        reference_answer="4",
        prompt_context="",
        metadata={},
    )


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


def test_free_mad_lite_stage_a_prompt_matches_standard_vanilla_mad() -> None:
    sample = _sample()
    assert build_initial_messages(sample, 1) == build_standard_mad_initial_messages(
        sample,
        1,
        prompt_version=CONTROLLED_PROMPT_VERSION,
    )


def test_free_mad_lite_vanilla_debate_prompt_matches_standard_vanilla_mad() -> None:
    sample = _sample()
    peer_messages = [{"agent": "agent_2", "answer": "4", "reasoning": "ok"}]
    assert build_debate_messages(
        sample,
        1,
        mode="vanilla",
        previous_answer="5",
        previous_reasoning="guess",
        peer_messages=peer_messages,
    ) == build_standard_mad_debate_messages(
        sample=sample,
        agent_id=1,
        round_index=1,
        previous_reasoning="guess",
        previous_answer="5",
        peer_messages=peer_messages,
        prompt_version=CONTROLLED_PROMPT_VERSION,
    )

