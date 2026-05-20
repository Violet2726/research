from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.dmad.config import load_roster_config
from research_experiments.families.dmad.prompts import (
    build_answer_stage_messages,
    build_initial_messages,
    build_mrp_method_selection_messages,
    build_reasoning_stage_messages,
    build_self_contrast_revision_messages,
)


def _math_sample() -> DatasetSample:
    return DatasetSample(
        dataset="competition_math",
        sample_id="competition_math/test/algebra/0",
        question="What is 2 + 2?",
        reference_answer="4",
        prompt_context="",
        metadata={"subject": "algebra"},
    )


def _choice_sample() -> DatasetSample:
    return DatasetSample(
        dataset="mmlu_abstract_algebra",
        sample_id="mmlu_abstract_algebra-0",
        question="Which option is correct?\nA. one\nB. two\nC. three\nD. four",
        reference_answer="B|||two",
        prompt_context="Options:\nA. one\nB. two\nC. three\nD. four",
        metadata={"subject": "abstract_algebra"},
    )


def _counting_sample() -> DatasetSample:
    return DatasetSample(
        dataset="competition_math",
        sample_id="competition_math/test/counting_and_probability/191.json",
        question="How many valid arrangements are there?",
        reference_answer="48",
        prompt_context="",
        metadata={"subject": "counting_and_probability"},
    )


def _gpqa_sample() -> DatasetSample:
    return DatasetSample(
        dataset="gpqa_diamond",
        sample_id="gpqa-0",
        question="Which option is correct?\nA. one\nB. two\nC. three\nD. four",
        reference_answer="C|||three",
        prompt_context="Options:\nA. one\nB. two\nC. three\nD. four",
        metadata={"subject": "physics", "answer_letter": "C", "answer_text": "three"},
    )


def test_persona_discriminative_roster_uses_roles_and_cot() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/mad_persona_discriminative_3agent.toml")

    assert roster.diversity_mode == "persona_discriminative"
    assert {agent.strategy_name for agent in roster.agents} == {"cot"}

    content = build_initial_messages(_math_sample(), roster.agents[0], prompt_version="dmad_v1_json")[1]["content"]
    assert "Persona: affirmative_debater" in content
    assert "Reasoning method: CoT" in content


def test_strategy_diverse_roster_injects_cot_sbp_pot() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/dmad_cot_sbp_pot_3agent.toml")
    contents = [build_initial_messages(_math_sample(), agent, prompt_version="dmad_v1_json")[1]["content"] for agent in roster.agents]

    assert roster.diversity_mode == "strategy_diverse"
    assert {agent.strategy_name for agent in roster.agents} == {"cot", "sbp", "pot"}
    assert any("Reasoning method: CoT" in content for content in contents)
    assert any("Reasoning method: SBP" in content for content in contents)
    assert any("Reasoning method: PoT" in content for content in contents)
    assert any('"python_program"' in content for content in contents)
    assert any('variable named "ans"' in content for content in contents)


def test_l2m_roster_keeps_non_math_appendix_prompting_family() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/dmad_cot_sbp_l2m_3agent.toml")
    content = build_initial_messages(_choice_sample(), roster.agents[2], prompt_version="dmad_v1_json")[1]["content"]

    assert "Reasoning method: L2M" in content
    assert "Least-to-Most prompting" in content
    assert "Reasoning method: PoT" not in content


def test_strategy_diverse_roster_switches_pot_to_l2m_on_gpqa() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/dmad_cot_sbp_pot_3agent.toml")
    content = build_initial_messages(_gpqa_sample(), roster.agents[2], prompt_version="dmad_v1_json")[1]["content"]

    assert "Reasoning method: L2M" in content
    assert "Least-to-Most prompting" in content
    assert "Reasoning method: PoT" not in content


def test_reasoning_stage_prompt_shows_peer_history() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/dmad_cot_sbp_pot_3agent.toml")
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
                "reasoning": "Work backward from the governing principle.",
                "execution_result": "",
            },
        ]
    ]
    content = build_reasoning_stage_messages(
        _math_sample(),
        roster.agents[0],
        round_index=2,
        prior_rounds=prior_rounds,
        prompt_version="dmad_v1_json",
    )[1]["content"]

    assert "agent_2: method=sbp" in content
    assert "Keep using your assigned reasoning method." in content
    assert "Reasoning method: CoT" in content


def test_counting_probability_prompts_require_explicit_verification() -> None:
    roster = load_roster_config("configs/families/dmad/rosters/dmad_cot_sbp_pot_3agent.toml")

    reasoning_content = build_reasoning_stage_messages(
        _counting_sample(),
        roster.agents[0],
        round_index=2,
        prior_rounds=[],
        prompt_version="dmad_v1_json",
    )[1]["content"]
    answer_content = build_answer_stage_messages(
        _counting_sample(),
        roster.agents[2],
        round_index=2,
        solving_process="ans = 48",
        execution_result="48",
        execution_status="ok",
        prior_rounds=[],
        prompt_version="dmad_v1_json",
    )[1]["content"]

    assert "complement count" in reasoning_content
    assert "case split" in reasoning_content
    assert 'Program execution result from variable "ans": 48' in answer_content


def test_mrp_selection_prompt_lists_candidate_methods() -> None:
    content = build_mrp_method_selection_messages(
        _math_sample(),
        ["cot", "sbp", "pot"],
        prompt_version="dmad_v1_json",
    )[1]["content"]

    assert "Candidate reasoning methods:" in content
    assert "- cot:" in content
    assert '"selected_method" and "reasoning"' in content


def test_self_contrast_revision_prompt_mentions_checklist_and_candidates() -> None:
    content = build_self_contrast_revision_messages(
        _math_sample(),
        [
            {"strategy_name": "cot", "reasoning": "Compute directly.", "answer": "4"},
            {"strategy_name": "sbp", "reasoning": "Use addition principle.", "answer": "4"},
            {"strategy_name": "pot", "reasoning": "ans = 2 + 2", "answer": "4"},
        ],
        checklist="1. Verify arithmetic.\n2. Reject unsupported leaps.",
        prompt_version="dmad_v1_json",
    )[1]["content"]

    assert "Contrastive checklist:" in content
    assert "Candidate 1:" in content
    assert '"final_answer" and "reasoning"' in content
