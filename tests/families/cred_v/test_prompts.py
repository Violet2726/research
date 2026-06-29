from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_v.prompts import build_mc_blind_pairwise_duel_messages, build_stage_a_messages


def test_stage_a_anchor_prompt_matches_sc_contract_without_role_lens() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="mmlu-1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        agent_id=1,
        agent_role="constraint_skeptic",
        prompt_mode="sc5_anchor_free_text_v1",
        output_protocol="free_text_answer_v1",
    )

    combined = "\n".join(message["content"] for message in messages)
    assert "Assigned lens: constraint_skeptic" not in combined
    assert "REASONING:" in combined
    assert "FINAL_ANSWER:" in combined
    assert "JSON answer object" not in combined
    assert "answer card" not in combined


def test_stage_a_role_guided_prompt_is_legacy_ablation_only() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="mmlu-1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        agent_id=1,
        agent_role="constraint_skeptic",
        prompt_mode="reasoning_first_roles_v1",
        output_protocol="free_text_answer_v1",
    )

    combined = "\n".join(message["content"] for message in messages)
    assert "Assigned lens: constraint_skeptic" in combined
    assert "REASONING:" in combined
    assert "FINAL_ANSWER:" in combined


def test_pairwise_duel_prompt_hides_leader_and_original_letters() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="mmlu-1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_mc_blind_pairwise_duel_messages(sample, leader_answer="A", challenger_answer="B", variant_index=1)
    combined = "\n".join(message["content"] for message in messages)

    assert "Candidate X:" in combined
    assert "Candidate Y:" in combined
    assert "leader" not in combined.lower()
    assert "challenger" not in combined.lower()
    assert "selected_side" in combined
    assert "slot; check; choose X" in combined
    assert "Field guide:" not in combined
