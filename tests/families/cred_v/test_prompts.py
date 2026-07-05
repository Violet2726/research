from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.cred_v.prompts import (
    build_hotpot_certificate_proposal_messages,
    build_isp_shadow_messages,
    build_math_certificate_proposal_messages,
    build_mc_blind_pairwise_duel_messages,
    build_mc_shadow_evidence_select_messages,
    build_stage_a_messages,
)


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


def test_shadow_evidence_prompt_records_view_and_uses_blind_sides() -> None:
    sample = DatasetSample(
        dataset="gpqa_diamond",
        sample_id="gpqa-1",
        question="Which option is best?",
        reference_answer="A",
        prompt_context="",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_mc_shadow_evidence_select_messages(
        sample,
        leader_answer="A",
        challenger_answer="B",
        variant_index=1,
        evidence_view="constraint_elimination_shadow",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "constraint_elimination_shadow" in combined
    assert "Candidate X:" in combined
    assert "Candidate Y:" in combined
    assert "selected_side" in combined
    assert "leader" not in combined.lower()
    assert "challenger" not in combined.lower()


def test_math_certificate_prompt_specifies_positive_dsl_contract() -> None:
    sample = DatasetSample("math500", "m1", "Evaluate 1/2 + 1/3.", "5/6", "", {})

    messages = build_math_certificate_proposal_messages(
        sample,
        leader_answer="1/2",
        stage_rows=[],
        dsl_version="math_cert_v1",
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "math_cert_v1" in combined
    assert "certificate_type" in combined
    assert "problem_expression" in combined
    assert '"problem_expression": "1/2 + 1/3"' in combined
    assert "problem_constants" in combined
    assert "expression_evaluation" in combined
    assert "Return one compact JSON" in combined
    assert "one formula-level verification trace" in combined


def test_hotpot_certificate_prompt_requests_locatable_span_certificate() -> None:
    sample = DatasetSample(
        "hotpotqa",
        "h1",
        "Who led the expedition?",
        "Captain John Underhill",
        "[Expedition] The expedition was led by Captain John Underhill.",
        {},
    )

    messages = build_hotpot_certificate_proposal_messages(
        sample,
        leader_answer="John Underhill",
        stage_rows=[],
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "context_span_completion" in combined
    assert "source_title" in combined
    assert "source_sentence_index" in combined
    assert "evidence_span" in combined
    assert "missing_tokens" in combined


def test_isp_shadow_prompt_collects_second_order_distribution_without_changing_answer_contract() -> None:
    sample = DatasetSample("strategyqa", "s1", "Can X happen?", "yes", "", {})

    messages = build_isp_shadow_messages(
        sample,
        own_answer="yes",
        candidate_answers=("yes", "no"),
        agent_index=1,
    )
    combined = "\n".join(message["content"] for message in messages)

    assert "peer_distribution" in combined
    assert "yes" in combined
    assert "no" in combined
    assert "shadow" in combined.lower()
