from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.dmad.config import load_roster_config
from research_experiments.families.dmad.prompts import (
    build_initial_messages,
    build_reasoning_stage_messages,
    build_solution_selector_messages,
)


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="math500",
        sample_id="math500-0",
        question="What is 2 + 2?",
        reference_answer="4",
        prompt_context="",
        metadata={},
    )


def _choice_sample() -> DatasetSample:
    return DatasetSample(
        dataset="gpqa_diamond",
        sample_id="gpqa-0",
        question="Which option is correct?\nA. one\nB. two\nC. three\nD. four",
        reference_answer="B",
        prompt_context="",
        metadata={},
    )


def test_persona_diverse_roster_changes_persona_but_not_strategy() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/persona_diverse_3agent.toml")

    assert roster.diversity_mode == "persona_diverse"
    assert len({agent.persona_name for agent in roster.agents}) == 3
    assert {agent.strategy_name for agent in roster.agents} == {"cot"}

    messages = build_initial_messages(_sample(), roster.agents[0], prompt_version="dmad_v1_json")
    assert "Persona: affirmative_debater" in messages[1]["content"]
    assert "Reasoning method: CoT" in messages[1]["content"]
    assert "Step-Back Prompting" not in messages[1]["content"]


def test_strategy_diverse_roster_injects_three_distinct_strategies() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/strategy_diverse_3agent.toml")
    strategy_names = {agent.strategy_name for agent in roster.agents}

    assert roster.diversity_mode == "strategy_diverse"
    assert strategy_names == {
        "cot",
        "sbp",
        "pot_l2m",
    }

    contents = [
        build_initial_messages(_sample(), agent, prompt_version="dmad_v1_json")[1]["content"]
        for agent in roster.agents
    ]
    assert any("Reasoning method: CoT" in content for content in contents)
    assert any("Reasoning method: SBP" in content for content in contents)
    assert any("Reasoning method: PoT" in content for content in contents)


def test_strategy_diverse_prompt_switches_to_l2m_on_non_math_choice_tasks() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/strategy_diverse_3agent.toml")
    content = build_initial_messages(_choice_sample(), roster.agents[2], prompt_version="dmad_v1_json")[1]["content"]

    assert "Reasoning method: L2M" in content
    assert "Least-to-Most prompting" in content
    assert "Reasoning method: PoT" not in content


def test_dmad_reasoning_stage_prompt_keeps_original_strategy_and_shows_peer_history() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/strategy_diverse_3agent.toml")
    prior_rounds = [
        [
            {
                "round_index": "1",
                "agent_id": "1",
                "persona_name": roster.agents[0].persona_name,
                "strategy_name": roster.agents[0].strategy_name,
                "answer": "3",
                "reasoning": "My previous process.",
            },
            {
                "round_index": "1",
                "agent_id": "2",
                "persona_name": roster.agents[1].persona_name,
                "strategy_name": roster.agents[1].strategy_name,
                "answer": "4",
                "reasoning": "Work backward from the target.",
            },
        ]
    ]
    messages = build_reasoning_stage_messages(
        _sample(),
        roster.agents[0],
        round_index=2,
        prior_rounds=prior_rounds,
        prompt_version="dmad_v1_json",
    )
    content = messages[1]["content"]

    assert "agent_2: method=sbp" in content
    assert "Keep using your assigned reasoning method." in content
    assert "Reasoning method: CoT" in content


def test_solution_selector_prompt_lists_candidate_solutions() -> None:
    messages = build_solution_selector_messages(
        _choice_sample(),
        [
            {
                "agent_id": 1,
                "persona_name": "affirmative_debater",
                "strategy_name": "cot",
                "reasoning": "Best option: B because the count is two.",
                "answer": "B",
            },
            {
                "agent_id": 2,
                "persona_name": "negative_debater",
                "strategy_name": "cot",
                "reasoning": "Best option: C because I miscounted.",
                "answer": "C",
            },
        ],
        prompt_version="dmad_v1_json",
    )
    content = messages[1]["content"]

    assert "Candidate 1:" in content
    assert "Candidate 2:" in content
    assert "\"selected_agent_id\", \"final_answer\", and \"rationale\"" in content
