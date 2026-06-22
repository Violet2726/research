from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from testsupport.filesystem import write_json, write_registered_family_manifest

from research_experiments.core.controls.control_prompts import FREE_TEXT_V1_PROMPT_VERSION, build_cot_messages
from research_experiments.core.controls.no_comm_controls import run_unified_control_sample
from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.adaptive_sparse_mad.algorithms import (
    aggregate_anchor_protected,
    aggregate_constraint_aware_stage_a,
    aggregate_evidence_grounded_stage_a,
    aggregate_family_slot_grounded_stage_a,
)
from research_experiments.families.adaptive_sparse_mad.config import (
    COT_MAD_GLOBAL_SYNC_METHOD,
    AdaptiveSparseMadProtocolConfig,
    load_control_catalog,
    load_experiment_config,
    load_protocol_config,
)
from research_experiments.families.adaptive_sparse_mad.prompts import (
    COT_GLOBAL_SYNC_PROMPT_VERSION,
    FREE_TEXT_DEBATE_PROMPT_VERSION,
    STAGE_A_V2_PROMPT_VERSION,
    STAGE_A_V4_PROMPT_VERSION,
    build_adaptive_addon_messages,
    build_global_sync_certificate_messages,
    build_global_sync_audit_messages,
    build_meta_router_head_messages,
    build_sparse_debate_messages,
    build_stage_a_messages,
    parse_adaptive_sparse_mad_free_text_output,
    parse_meta_router_head_output,
)
from research_experiments.families.adaptive_sparse_mad.run.sample import (
    _answers_share_family,
    _apply_stage_a_answer_slot_safeguard,
    _apply_stage_a_consistency_safeguard,
    _build_adaptive_gate_decision,
    _build_global_sync_candidate_board,
    _build_global_sync_gate_decision,
    _default_meta_router_payload,
    _execute_control_turn,
    _execute_turn,
    _resolve_global_sync_audit_outcome,
    _resolve_v7_all_three_wrong_override,
    _resolve_v7_single_step_override,
    _select_adaptive_addon_solver_sequence,
    _should_accept_counterfactual_override,
    _should_safe_retry_stage_a_result,
    _run_sample,
    _validate_control_output,
    _validate_stage_a_output,
    build_policy_diagnostics,
    build_router_eval_payload,
    build_stage_a_error_bucket_payload,
    build_stage_a_resolver_breakdown_payload,
    build_stage_a_solver_contribution_payload,
    refresh_prediction_rows_for_run,
    refresh_stage_a_prediction_rows,
    summarize_run,
)
from research_experiments.families.adaptive_sparse_mad.run.report import render_report
from research_experiments.family_runtime.output_protocols import FREE_TEXT_ANSWER_PROTOCOL_V1


def test_stage_a_v2_direct_solver_reuses_unified_cot_prompt() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        solver_mode="solver_cot",
        agent_id=1,
        prompt_version=STAGE_A_V2_PROMPT_VERSION,
    )

    assert messages == build_cot_messages(sample, 1, None)


def test_stage_a_v2_structured_solver_returns_schema_prompt() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        solver_mode="solver_skeptic",
        agent_id=3,
        prompt_version=STAGE_A_V2_PROMPT_VERSION,
    )

    assert messages[0]["role"] == "system"
    assert "answer_type" in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert '"key_constraints":"short constraints"' in messages[1]["content"]


def test_stage_a_v4_direct_solver_returns_evidence_schema_prompt() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )

    messages = build_stage_a_messages(
        sample,
        solver_mode="solver_cot",
        agent_id=1,
        prompt_version=STAGE_A_V4_PROMPT_VERSION,
    )

    assert messages[0]["role"] == "system"
    assert "claim_span" in messages[0]["content"]
    assert '"confidence_raw":0.0' in messages[1]["content"]


def test_stage_a_free_text_prompt_uses_required_tags_without_json_contract() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )

    messages = build_stage_a_messages(
        sample,
        solver_mode="solver_skeptic",
        agent_id=3,
        prompt_version=FREE_TEXT_DEBATE_PROMPT_VERSION,
    )

    user_content = messages[1]["content"]
    assert "REASONING:" in user_content
    assert "FINAL_ANSWER:" in user_content
    assert "CONFIDENCE:" in user_content
    assert "Return exactly one JSON object" not in user_content
    assert "strict JSON" not in messages[0]["content"]


def test_stage_a_global_sync_prompt_reuses_sc5_aligned_free_text_prompt() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )

    messages = build_stage_a_messages(
        sample,
        solver_mode="solver_cot",
        agent_id=1,
        prompt_version=COT_GLOBAL_SYNC_PROMPT_VERSION,
    )

    assert messages == build_cot_messages(sample, 1, FREE_TEXT_V1_PROMPT_VERSION)


def test_parse_adaptive_sparse_mad_free_text_output_validates_required_tags() -> None:
    payload = parse_adaptive_sparse_mad_free_text_output(
        "\n".join(
            [
                "REASONING: Option C matches the evidence.",
                "FINAL_ANSWER: Option C",
                "CONFIDENCE: 0.82",
                "ANSWER_TYPE: multiple_choice",
                "KEY_CONSTRAINTS: single option letter",
                "KEY_EVIDENCE: C is directly supported",
                "FAILURE_RISK: none",
            ]
        ),
        dataset="gpqa_diamond",
    )

    assert payload["final_answer"] == "C"
    assert payload["confidence_raw"] == 0.82
    assert payload["answer_type"] == "multiple_choice"


def test_parse_adaptive_sparse_mad_free_text_output_rejects_missing_required_fields() -> None:
    with pytest.raises(ValueError, match="KEY_EVIDENCE"):
        parse_adaptive_sparse_mad_free_text_output(
            "\n".join(
                [
                    "REASONING: Compute directly.",
                    "FINAL_ANSWER: 42",
                    "CONFIDENCE: 0.8",
                    "ANSWER_TYPE: number",
                    "KEY_CONSTRAINTS: numeric answer",
                    "FAILURE_RISK: none",
                ]
            ),
            dataset="gsm8k",
        )


def test_parse_adaptive_sparse_mad_free_text_output_rejects_malformed_confidence() -> None:
    with pytest.raises(ValueError, match="CONFIDENCE"):
        parse_adaptive_sparse_mad_free_text_output(
            "\n".join(
                [
                    "REASONING: Compute directly.",
                    "FINAL_ANSWER: 42",
                    "CONFIDENCE: very high",
                    "ANSWER_TYPE: number",
                    "KEY_CONSTRAINTS: numeric answer",
                    "KEY_EVIDENCE: 40 + 2",
                    "FAILURE_RISK: arithmetic slip",
                ]
            ),
            dataset="gsm8k",
        )


def test_parse_adaptive_sparse_mad_free_text_output_preserves_plain_math_answer() -> None:
    payload = parse_adaptive_sparse_mad_free_text_output(
        "\n".join(
            [
                "REASONING: Simplifying leaves x + 1.",
                "FINAL_ANSWER: x + 1",
                "CONFIDENCE: 75%",
                "ANSWER_TYPE: expression",
                "KEY_CONSTRAINTS: plain ASCII expression",
                "KEY_EVIDENCE: terms combine to x + 1",
                "FAILURE_RISK: algebra slip",
            ]
        ),
        dataset="math500",
    )

    assert payload["final_answer"] == "x + 1"
    assert payload["confidence_raw"] == 0.75


def test_parse_meta_router_head_output_normalizes_aliases_and_sequences() -> None:
    payload = parse_meta_router_head_output(
        """
        {
          "selected_candidate": "cot",
          "error_mode": "pseudo_majority",
          "should_trigger": true,
          "recommended_solver_sequence": ["solver_verify"],
          "router_confidence": "82%",
          "reasoning_short": "  majority looks fragile but recoverable  "
        }
        """
    )

    assert payload == {
        "selected_candidate": "solver_cot",
        "error_mode": "pseudo_majority",
        "should_trigger": True,
        "recommended_solver_sequence": ["solver_verify"],
        "router_confidence": 0.82,
        "reasoning_short": "majority looks fragile but recoverable",
    }


def test_build_meta_router_head_messages_includes_strict_json_contract() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )

    messages = build_meta_router_head_messages(
        sample,
        stage_a_rows=[
            {
                "solver_mode": "solver_cot",
                "normalized_answer": "french",
                "confidence_value": 0.9,
                "reasoning": "Quebec implies French.",
                "key_evidence": "French is spoken in Quebec.",
            },
            {
                "solver_mode": "solver_l2m",
                "normalized_answer": "english",
                "confidence_value": 0.5,
                "reasoning": "Nearby regions use English.",
                "key_evidence": "English appears elsewhere.",
            },
        ],
    )

    assert messages[0]["role"] == "system"
    assert "Return exactly one JSON object." in messages[0]["content"]
    assert messages[1]["role"] == "user"
    assert "selected_candidate" in messages[1]["content"]
    assert "recommended_solver_sequence" in messages[1]["content"]
    assert "no_confident_candidate" in messages[1]["content"]


def test_build_sparse_debate_messages_includes_peer_evidence_prior_answer_and_gate_reasons() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    own_row = {
        "agent_id": 1,
        "solver_mode": "solver_cot",
        "normalized_answer": "english",
        "reasoning": "I focused on the wrong country.",
        "key_evidence": "English appears nearby.",
        "confidence_value": 0.4,
    }
    peer_rows = [
        {
            "agent_id": 2,
            "solver_mode": "solver_evidence",
            "normalized_answer": "french",
            "reasoning": "Quebec evidence supports French.",
            "key_evidence": "French is spoken",
            "confidence_value": 0.8,
        }
    ]

    messages = build_sparse_debate_messages(
        sample,
        agent_id=1,
        round_index=1,
        own_row=own_row,
        peer_rows=peer_rows,
        gate_decision={"trigger_reasons": ["answer_disagreement", "evidence_conflict"]},
        leading_answer="french",
    )

    content = messages[1]["content"]
    assert "Your prior answer packet" in content
    assert "Peer answers and evidence" in content
    assert "French is spoken" in content
    assert "answer_disagreement" in content
    assert "REVISION_NOTE:" in content


def test_build_global_sync_certificate_messages_uses_budgeted_board_without_raw_reasoning() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken. " * 200,
        metadata={},
    )
    own_row = {
        "agent_id": 1,
        "solver_mode": "solver_cot",
        "normalized_answer": "english",
        "reasoning": "RAW_OWN_REASONING_SHOULD_NOT_APPEAR " * 30,
        "key_evidence": "English appears nearby.",
        "confidence_value": 0.44,
    }
    stage_a_rows = [
        own_row,
        {
            "agent_id": 2,
            "solver_mode": "solver_cot",
            "normalized_answer": "french",
            "reasoning": "RAW_PEER_REASONING_SHOULD_NOT_APPEAR " * 30,
            "key_evidence": "French is spoken in Quebec.",
            "confidence_value": 0.84,
        },
    ]
    candidate_board = [
        {
            "family_key": "open:french",
            "family_id": "F1",
            "representative_answer": "french",
            "vote_count": 4,
            "stage_a_support": 4,
            "agent_ids": [2, 3, 4, 5],
            "avg_confidence": 0.82,
            "evidence_digest": "French is spoken in Quebec.",
            "risk_signals": [],
        },
        {
            "family_key": "open:english",
            "family_id": "F2",
            "representative_answer": "english",
            "vote_count": 1,
            "stage_a_support": 1,
            "agent_ids": [1],
            "avg_confidence": 0.44,
            "evidence_digest": "English appears nearby.",
            "risk_signals": ["minority_only"],
        },
    ]

    messages = build_global_sync_certificate_messages(
        sample,
        agent_id=1,
        own_row=own_row,
        candidate_board=candidate_board,
        gate_decision={"trigger_reasons": ["answer_disagreement"], "vote_pattern": "4-1"},
        stage_a_majority_answer="french",
        own_prior_max_chars=32,
    )

    content = messages[1]["content"]
    assert "Compressed Stage A candidate board" in content
    assert "vote_count=4" in content
    assert "PREFERRED_FAMILY:" in content
    assert "MAJORITY_ERROR:" in content
    assert "ERROR_TYPE:" in content
    assert "CERT_EVIDENCE:" in content
    assert "REVISION_NOTE:" in content
    assert "RAW_OWN_REASONING_SHOULD_NOT_APPEAR" not in content
    assert "RAW_PEER_REASONING_SHOULD_NOT_APPEAR" not in content
    assert "Stage A peer packets" not in content
    assert "where French is spoken. The town is in Quebec" not in content
    assert "Return exactly one JSON object" not in content


def test_build_global_sync_candidate_board_limits_to_top_two_families_and_budget() -> None:
    rows = []
    for agent_id, answer in enumerate(["A", "A", "A", "B", "C"], start=1):
        rows.append(
            {
                "agent_id": agent_id,
                "solver_mode": "solver_cot",
                "normalized_answer": answer,
                "confidence_value": 0.8,
                "key_evidence": f"evidence for {answer} " * 80,
                "reasoning": f"long raw reasoning for {answer} " * 100,
                "output_status": "ok",
            }
        )

    board = _build_global_sync_candidate_board(
        rows,
        dataset="mmlu_pro",
        question="Which option is best?",
        max_board_chars=900,
        family_evidence_max_chars=80,
    )

    assert len(board) == 2
    assert [row["representative_answer"] for row in board] == ["A", "B"]
    assert all("evidence_snippets" not in row for row in board)
    assert all(len(str(row["evidence_digest"])) <= 80 for row in board)
    assert len(str(board)) <= 1200


def test_build_adaptive_addon_messages_includes_stage_a_candidate_summary() -> None:
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "claim_span": "A",
            "key_evidence": "alpha matches the definition",
            "answer_type": "option_letter",
            "key_constraints": "single option letter",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "B",
            "confidence_value": 0.5,
            "claim_span": "B",
            "key_evidence": "beta has the required property",
            "answer_type": "option_letter",
            "key_constraints": "single option letter",
        },
    ]

    messages = build_adaptive_addon_messages(
        sample,
        solver_mode="solver_option_elim",
        agent_id=4,
        stage_a_rows=stage_a_rows,
    )

    assert "Stage A candidate summary" in messages[1]["content"]
    assert "solver_cot" in messages[1]["content"]
    assert "selected_candidate" in messages[1]["content"]


def test_build_counterfactual_addon_messages_mentions_leading_candidate_family() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which gaming console were both games released on?",
        reference_answer="PlayStation 4",
        prompt_context="Both games were released for PlayStation 3 and PlayStation 4.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "playstation 3",
            "confidence_value": 0.61,
            "claim_span": "PlayStation 3",
            "key_evidence": "released for PlayStation 3",
            "answer_type": "gaming console",
            "key_constraints": "single console name",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "playstation 3",
            "confidence_value": 0.58,
            "claim_span": "PlayStation 3",
            "key_evidence": "released for PlayStation 3",
            "answer_type": "gaming console",
            "key_constraints": "single console name",
        },
    ]

    messages = build_adaptive_addon_messages(
        sample,
        solver_mode="solver_counterfactual",
        agent_id=4,
        stage_a_rows=stage_a_rows,
    )

    assert "Current leading candidate family" in messages[1]["content"]
    assert "playstation 3" in messages[1]["content"].lower()


def test_build_disconfirm_addon_messages_mentions_leading_candidate_family() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which gaming console were both games released on?",
        reference_answer="PlayStation 4",
        prompt_context="Both games were released for PlayStation 3 and PlayStation 4.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "playstation 3",
            "confidence_value": 0.61,
            "claim_span": "PlayStation 3",
            "key_evidence": "released for PlayStation 3",
            "answer_type": "gaming console",
            "key_constraints": "single console name",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "playstation 3",
            "confidence_value": 0.58,
            "claim_span": "PlayStation 3",
            "key_evidence": "released for PlayStation 3",
            "answer_type": "gaming console",
            "key_constraints": "single console name",
        },
    ]

    messages = build_adaptive_addon_messages(
        sample,
        solver_mode="solver_disconfirm",
        agent_id=4,
        stage_a_rows=stage_a_rows,
    )

    assert "Current leading candidate family" in messages[1]["content"]
    assert "must not be a trivial restatement" in messages[1]["content"]

def test_should_safe_retry_stage_a_result_for_recovered_output() -> None:
    result = SimpleNamespace(
        output_status="ok",
        response_payload={"assistant_text": '{"reasoning":"cut off"}'},
        validated_output={
            "final_answer": "42",
            "stage_a_recovery_fallback": "answer_core_recovery_fallback",
        },
    )

    assert _should_safe_retry_stage_a_result(result) is True


def test_should_safe_retry_stage_a_result_allows_clean_output() -> None:
    result = SimpleNamespace(
        output_status="ok",
        response_payload={"assistant_text": '{"final_answer":"42","reasoning":"done"}'},
        validated_output={"final_answer": "42"},
    )

    assert _should_safe_retry_stage_a_result(result) is False


def test_anchor_protected_aggregate_prefers_anchor_on_three_way_split() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.9},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "C", "confidence_value": 0.9},
    ]

    answer, support = aggregate_anchor_protected(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "B": 0.9, "C": 0.9}


def test_anchor_protected_aggregate_allows_two_to_one_override() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.95},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.5},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "B", "confidence_value": 0.5},
    ]

    answer, support = aggregate_anchor_protected(rows)

    assert answer == "B"
    assert support == {"A": 0.95, "B": 1.0}


def test_anchor_protected_prefers_clean_anchor_over_degraded_non_anchor_majority() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.5, "stage_a_safe_retry_used": True},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "B", "confidence_value": 0.5},
    ]

    answer, support = aggregate_anchor_protected(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "B": 1.0}


def test_anchor_protected_keeps_non_anchor_majority_when_anchor_is_also_degraded() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5, "stage_a_safe_retry_used": True},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.5, "stage_a_safe_retry_used": True},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "B", "confidence_value": 0.5},
    ]

    answer, _support = aggregate_anchor_protected(rows)

    assert answer == "B"


def test_anchor_protected_aggregate_ignores_unknown_majority() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "unknown", "confidence_value": 0.5},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "unknown", "confidence_value": 0.5},
    ]

    answer, support = aggregate_anchor_protected(rows)

    assert answer == "A"
    assert support == {"A": 0.5}


def test_constraint_aware_stage_a_defaults_to_anchor_vote_for_clean_split() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.9},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "C", "confidence_value": 0.9},
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "B": 0.9, "C": 0.9}
    assert resolver == "constraint_aware_anchor_vote"


def test_constraint_aware_stage_a_prefers_clean_cot_minority_in_two_to_one_split() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.5},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "B", "confidence_value": 0.5},
        {"agent_id": 3, "solver_mode": "solver_skeptic", "normalized_answer": "B", "confidence_value": 0.5},
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "B": 1.0}
    assert resolver == "constraint_aware_clean_anchor_minority_override"


def test_constraint_aware_stage_a_keeps_clean_expression_majority_over_cot_minority() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "((1)/(sinxcosx))", "confidence_value": 0.5},
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "cot x",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "expression",
                "key_constraints": "simplify trigonometric expression, no explanation in final answer",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "cot x",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "mathematical expression",
                "key_constraints": "simplify trigonometric expression, no extra terms, final answer as mathematical expression",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "cot x"
    assert support == {"((1)/(sinxcosx))": 0.5, "cot x": 1.0}
    assert resolver == "constraint_aware_clean_expression_majority_keep"


def test_constraint_aware_stage_a_keeps_clean_slot_majority_over_cot_minority() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "q inspired by charles frasersmith",
            "confidence_value": 0.5,
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "q",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "named type (job title)",
                "key_constraints": "short exact span from context; prefer shortest span",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "q",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "character",
                "key_constraints": "fictional head of British Secret Service division",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "q"
    assert support == {"q": 1.0, "q inspired by charles frasersmith": 0.5}
    assert resolver == "constraint_aware_clean_slot_majority_keep"


def test_constraint_aware_stage_a_prefers_typed_symmetry_minority() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "reasoning": "IR 1750 cm-1 points to a cyclopentanone precursor.",
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "reasoning": "Ring size stays 5 because the intermediate keeps the same carbon count.",
            "validated_output": {
                "answer_type": "symmetry group",
                "key_constraints": "generic symmetry group label",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "B",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "molecular symmetry group",
                "key_constraints": "molecular symmetry group must match final product",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "B"
    assert support == {"A": 1.0, "B": 0.5}
    assert resolver == "constraint_aware_typed_minority_override"


def test_constraint_aware_stage_a_prefers_option_minority_over_integer_majority() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "B", "confidence_value": 0.5},
        {
            "agent_id": 2,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "B",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "integer",
                "key_constraints": "count distinct items",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_l2m",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "multiple_choice",
                "key_constraints": "final answer must be a visible option letter",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "B": 1.0}
    assert resolver == "constraint_aware_typed_minority_override"


def test_constraint_aware_stage_a_prefers_numeric_minority_over_expression_majority() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "42105", "confidence_value": 0.5},
        {
            "agent_id": 2,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "42105",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "expression",
                "key_constraints": "free-form expression",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_l2m",
            "normalized_answer": "4210_(5)",
            "confidence_value": 0.5,
            "validated_output": {
                "answer_type": "number",
                "key_constraints": "canonical base-5 numeral",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "4210_(5)"
    assert support == {"42105": 1.0, "4210_(5)": 0.5}
    assert resolver == "constraint_aware_typed_minority_override"


def test_constraint_aware_stage_a_prefers_clean_skeptic_minority_for_conceptual_two_to_one_split() -> None:
    rows = [
        {"agent_id": 1, "solver_mode": "solver_cot", "normalized_answer": "E", "confidence_value": 0.5, "reasoning": "A trade surplus occurs when exports exceed imports.", "validated_output": {}},
        {"agent_id": 2, "solver_mode": "solver_l2m", "normalized_answer": "E", "confidence_value": 0.5, "reasoning": "A trade surplus means exports exceed imports.", "validated_output": {}},
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "reasoning": "Low domestic income reduces import demand, increasing surplus.",
            "validated_output": {
                "answer_type": "multiple-choice",
                "key_constraints": "option letter only, legal answer type is multiple-choice",
            },
        },
    ]

    answer, support, resolver = aggregate_constraint_aware_stage_a(rows)

    assert answer == "A"
    assert support == {"A": 0.5, "E": 1.0}
    assert resolver == "constraint_aware_clean_skeptic_minority_override"


def test_evidence_grounded_stage_a_prefers_evidenced_minority_over_degraded_majority() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "claim_span": "A",
            "key_evidence": "",
            "stage_a_safe_retry_used": True,
            "validated_output": {},
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "A",
            "confidence_value": 0.5,
            "claim_span": "A",
            "key_evidence": "",
            "validated_output": {},
        },
        {
            "agent_id": 4,
            "solver_mode": "solver_option_elim",
            "normalized_answer": "B",
            "confidence_value": 0.5,
            "claim_span": "B",
            "key_evidence": "Option B is the only choice consistent with the constraint.",
            "validated_output": {
                "answer_type": "multiple_choice",
                "key_constraints": "single option letter",
            },
        },
    ]

    answer, support, resolver = aggregate_evidence_grounded_stage_a(rows, anchor_answer="A")

    assert answer == "B"
    assert support["B"] > support["A"]
    assert resolver == "evidence_grounded_score_vote"


def test_evidence_grounded_stage_a_prefers_slot_complete_longer_answer() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "1840 students",
            "confidence_value": 0.5,
            "claim_span": "1,840 students",
            "key_evidence": "there were 1,840 students enrolled",
            "validated_output": {},
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "1840",
            "confidence_value": 0.5,
            "claim_span": "1,840",
            "key_evidence": "there were 1,840 students enrolled",
            "validated_output": {
                "answer_type": "number",
                "key_constraints": "exact enrollment count",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "1840",
            "confidence_value": 0.5,
            "claim_span": "1,840",
            "key_evidence": "there were 1,840 students enrolled",
            "validated_output": {
                "answer_type": "number",
                "key_constraints": "exact enrollment count",
            },
        },
    ]

    answer, _support, resolver = aggregate_evidence_grounded_stage_a(
        rows,
        question="How many students were enrolled in the school?",
    )

    assert answer == "1840 students"
    assert resolver == "evidence_grounded_score_vote"


def test_evidence_grounded_stage_a_prefers_year_prefixed_named_event() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "1991 perfect storm",
            "confidence_value": 0.5,
            "claim_span": "the 1991 Perfect Storm",
            "key_evidence": "the 1991 Perfect Storm developed off Atlantic Canada on October 29",
            "validated_output": {},
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "perfect storm",
            "confidence_value": 0.5,
            "claim_span": "the Perfect Storm",
            "key_evidence": "the Perfect Storm developed off Atlantic Canada on October 29",
            "validated_output": {
                "answer_type": "named storm",
                "key_constraints": "exact type words included",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "perfect storm",
            "confidence_value": 0.5,
            "claim_span": "The Perfect Storm",
            "key_evidence": "Robert Case inspired the naming of the Perfect Storm",
            "validated_output": {
                "answer_type": "named storm",
                "key_constraints": "exact type words included",
            },
        },
    ]

    answer, _support, resolver = aggregate_evidence_grounded_stage_a(
        rows,
        question="Which initial area of low pressure developed off Atlantic Canada on October 29 was inspired by Robert Case?",
    )

    assert answer == "1991 perfect storm"
    assert resolver == "evidence_grounded_score_vote"


def test_evidence_grounded_stage_a_prefers_short_boolean_over_explanatory_boolean() -> None:
    rows = [
        {
            "agent_id": 1,
            "solver_mode": "solver_cot",
            "normalized_answer": "no only vladimir danilevich is from russia",
            "confidence_value": 0.5,
            "claim_span": "No, only Vladimir Danilevich is from Russia.",
            "key_evidence": "Smith is American while Danilevich is Russian.",
            "validated_output": {},
        },
        {
            "agent_id": 2,
            "solver_mode": "solver_l2m",
            "normalized_answer": "no",
            "confidence_value": 0.5,
            "claim_span": "no",
            "key_evidence": "not both are from Russia",
            "validated_output": {
                "answer_type": "boolean",
                "key_constraints": "short yes/no answer",
            },
        },
        {
            "agent_id": 3,
            "solver_mode": "solver_skeptic",
            "normalized_answer": "no",
            "confidence_value": 0.5,
            "claim_span": "No",
            "key_evidence": "Smith is from Oregon, so not both are Russian",
            "validated_output": {
                "answer_type": "yes/no judgment",
                "key_constraints": "short yes/no answer",
            },
        },
    ]

    answer, _support, resolver = aggregate_evidence_grounded_stage_a(
        rows,
        question="Are both Harry Everett Smith and Vladimir Danilevich from Russia?",
    )

    assert answer == "no"
    assert resolver == "evidence_grounded_score_vote"


def test_stage_a_consistency_safeguard_recovers_numeric_answer_from_reasoning() -> None:
    payload = {
        "final_answer": "14",
        "reasoning": "Total is 60, discount is 18, final price is 42, so 50 minus 42 leaves 8.",
        "confidence_raw": 0.95,
        "claim_span": "14",
        "key_evidence": "50 minus 42 leaves 8.",
        "uncertain_point": None,
    }

    repaired = _apply_stage_a_consistency_safeguard(payload, dataset="gsm8k")

    assert repaired["final_answer"] == "8"
    assert repaired["consistency_fallback"] == "recovered_answer_from_reasoning"


def test_validate_stage_a_output_uses_raw_text_tail_for_truncated_numeric_reasoning() -> None:
    raw_text = (
        '{\n  "final_answer": "17.50",\n'
        '  "reasoning": "Total original price is 60. Discounted price is 42. '
        'Joe has 50 so he has 8 left.'
    )

    repaired = _validate_stage_a_output(raw_text, dataset="gsm8k")

    assert repaired["final_answer"] == "8"


def test_validate_stage_a_output_marks_unrecoverable_truncation_unknown() -> None:
    raw_text = '{"reasoning":"The model starts a long explanation but never returns a final answer before truncation.'

    repaired = _validate_stage_a_output(raw_text, dataset="hotpotqa")

    assert repaired["final_answer"] == "unknown"
    assert repaired["confidence_raw"] == 0.0
    assert repaired["stage_a_recovery_fallback"] == "unknown_after_unrecoverable_stage_a_output"


def test_validate_stage_a_output_trusts_valid_json_final_answer() -> None:
    raw_text = (
        '{"final_answer":"83","reasoning":"Capacity is 5000 - 3755 = 1245. '
        'Boxes = 1245 / 15 = 83. Check total weight is 5000.","confidence_raw":1.0}'
    )

    repaired = _validate_stage_a_output(raw_text, dataset="gsm8k")

    assert repaired["final_answer"] == "83"


def test_validate_stage_a_output_keeps_uncertainty_type_when_present() -> None:
    raw_text = (
        '{"final_answer":"B","reasoning":"Option B matches the definition.","confidence_raw":72,'
        '"uncertainty_type":"evidence_selection","answer_type":"option_letter",'
        '"key_constraints":"single option letter","failure_risk":"confusing nearby option text"}'
    )

    repaired = _validate_stage_a_output(raw_text, dataset="gpqa_diamond")

    assert repaired["final_answer"] == "B"
    assert repaired["confidence_raw"] == 72
    assert repaired["uncertainty_type"] == "evidence_selection"
    assert repaired["answer_type"] == "option_letter"


def test_validate_control_output_marks_unrecoverable_truncation_unknown() -> None:
    raw_text = (
        '{\n  "reasoning": "The question asks for an entity, but the response is truncated before '
        "it emits a final_answer field."
    )

    repaired = _validate_control_output(raw_text, dataset="hotpotqa")

    assert repaired["final_answer"] == "unknown"
    assert repaired["control_recovery_fallback"] == "unknown_after_unrecoverable_control_output"


def test_validate_stage_a_output_free_text_mode_does_not_decode_json_first() -> None:
    raw_text = (
        '{"final_answer":"83","reasoning":"Capacity is 5000 - 3755 = 1245. '
        'Boxes = 1245 / 15 = 83."}'
    )

    repaired = _validate_stage_a_output(
        raw_text,
        dataset="gsm8k",
        provider_reasoning_text="The final answer is 83.",
        response_format_mode="free_text",
    )

    assert repaired["final_answer"] == "83"
    assert repaired["stage_a_recovery_fallback"] == "answer_recovered_from_unstructured_stage_a_output"
    assert "free_text_parse_error" in repaired


def test_execute_turn_free_text_mode_disables_response_format(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[bool] = []

    def fake_execute_cached_turn(**kwargs):
        captured.append(bool(kwargs["use_response_format"]))
        return SimpleNamespace(
            output_status="ok",
            validated_output={"final_answer": "42", "reasoning": "short reasoning", "confidence_raw": 0.8},
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            response_payload={"latency_ms": 9, "assistant_text": "ok", "provider_reasoning_text": ""},
            cache_hit=False,
            request_error=None,
            prompt_hash="prompt-hash",
        )

    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample.execute_cached_turn",
        fake_execute_cached_turn,
    )

    sample = DatasetSample(
        dataset="gsm8k",
        sample_id="demo",
        question="What is 40 + 2?",
        reference_answer="42",
        prompt_context="",
        metadata={},
    )
    row = _execute_turn(
        run_id="run1",
        dataset="gsm8k",
        split_name="count20_seed42",
        sample=sample,
        stage_name="stage_a",
        method_name="solver_cot",
        role="initial",
        round_index=0,
        agent_id=1,
        messages=[{"role": "system", "content": "demo"}, {"role": "user", "content": "demo"}],
        backbone=SimpleNamespace(name="demo"),
        provider=SimpleNamespace(),
        cache=SimpleNamespace(),
        throttle=SimpleNamespace(),
        temperature=0.7,
        top_p=1.0,
        seed=42,
        output_mode="stage_a",
        prompt_version=FREE_TEXT_DEBATE_PROMPT_VERSION,
        response_format_mode="free_text",
    )

    assert captured == [False]
    assert row["prediction"] == "42"


def test_execute_turn_passes_certificate_max_tokens_only_when_requested(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int | None] = []

    def fake_execute_cached_turn(**kwargs):
        captured.append(kwargs.get("max_tokens"))
        return SimpleNamespace(
            output_status="ok",
            validated_output={
                "final_answer": "B",
                "reasoning": "certificate",
                "confidence_raw": 0.8,
                "selected_candidate": "F2",
                "majority_error": "majority conflicts with option text",
                "error_type": "evidence_conflict",
                "cert_evidence": "Option B states the required condition.",
            },
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            response_payload={"latency_ms": 9, "assistant_text": "ok", "provider_reasoning_text": ""},
            cache_hit=False,
            request_error=None,
            prompt_hash="prompt-hash",
        )

    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample.execute_cached_turn",
        fake_execute_cached_turn,
    )

    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="B|||beta",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )

    row = _execute_turn(
        run_id="run1",
        dataset="mmlu_pro",
        split_name="count20_seed42",
        sample=sample,
        stage_name="global_sync_certificate",
        method_name="global_sync_certificate",
        role="certificate_revision",
        round_index=1,
        agent_id=1,
        messages=[{"role": "system", "content": "demo"}, {"role": "user", "content": "demo"}],
        backbone=SimpleNamespace(name="demo"),
        provider=SimpleNamespace(),
        cache=SimpleNamespace(),
        throttle=SimpleNamespace(),
        temperature=0.7,
        top_p=1.0,
        seed=42,
        output_mode="stage_a",
        prompt_version=COT_GLOBAL_SYNC_PROMPT_VERSION,
        response_format_mode="free_text",
        max_tokens=256,
    )

    assert captured == [256]
    assert row["majority_error"] == "majority conflicts with option text"
    assert row["budget_cap_tokens"] == 256
    assert row["budget_cap_retry"] is False


def test_execute_turn_retries_without_budget_cap_when_provider_rejects_max_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[int | None] = []

    def fake_execute_cached_turn(**kwargs):
        captured.append(kwargs.get("max_tokens"))
        if kwargs.get("max_tokens") is not None:
            return SimpleNamespace(
                output_status="request_fail",
                validated_output={},
                usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                response_payload={
                    "latency_ms": 1,
                    "assistant_text": "",
                    "provider_reasoning_text": "",
                    "request_error": "unsupported parameter: max_tokens",
                },
                cache_hit=False,
                request_error="unsupported parameter: max_tokens",
                prompt_hash="prompt-hash",
            )
        return SimpleNamespace(
            output_status="ok",
            validated_output={
                "final_answer": "B",
                "reasoning": "certificate",
                "confidence_raw": 0.8,
                "selected_candidate": "F2",
                "majority_error": "majority conflicts with option text",
                "error_type": "evidence_conflict",
                "cert_evidence": "Option B states the required condition.",
            },
            usage={"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
            response_payload={"latency_ms": 9, "assistant_text": "ok", "provider_reasoning_text": ""},
            cache_hit=False,
            request_error=None,
            prompt_hash="prompt-hash-2",
        )

    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample.execute_cached_turn",
        fake_execute_cached_turn,
    )

    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is best?",
        reference_answer="B|||beta",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )

    row = _execute_turn(
        run_id="run1",
        dataset="mmlu_pro",
        split_name="count20_seed42",
        sample=sample,
        stage_name="global_sync_certificate",
        method_name="global_sync_certificate",
        role="certificate_revision",
        round_index=1,
        agent_id=1,
        messages=[{"role": "system", "content": "demo"}, {"role": "user", "content": "demo"}],
        backbone=SimpleNamespace(name="demo"),
        provider=SimpleNamespace(),
        cache=SimpleNamespace(),
        throttle=SimpleNamespace(),
        temperature=0.7,
        top_p=1.0,
        seed=42,
        output_mode="stage_a",
        prompt_version=COT_GLOBAL_SYNC_PROMPT_VERSION,
        response_format_mode="free_text",
        max_tokens=256,
    )

    assert captured == [256, None]
    assert row["prediction"] == "B"
    assert row["budget_cap_retry"] is True


def test_execute_control_turn_uses_output_protocol_runner(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, object] = {}

    def fake_execute_output_protocol_turn(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            payload={"messages": kwargs["messages"]},
            prompt_hash="prompt-hash",
            cache_key="cache-key",
            cache_hit=False,
            response_payload={"latency_ms": 13, "assistant_text": "demo", "provider_reasoning_text": "", "request_started_at": "ts"},
            request_error=None,
            request_status="ok",
            output_status="ok",
            validated_output={"final_answer": "42", "reasoning": "short reasoning"},
            usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
            output_protocol=kwargs["output_protocol"],
            protocol_parse_status="ok",
            protocol_parse_error=None,
            reason_present=True,
            request_count=1,
            cache_request_count=0,
            network_request_count=1,
            raw_finish_reason="stop",
        )

    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample.execute_output_protocol_turn",
        fake_execute_output_protocol_turn,
    )

    sample = DatasetSample(
        dataset="gsm8k",
        sample_id="demo",
        question="What is 40 + 2?",
        reference_answer="42",
        prompt_context="",
        metadata={},
    )
    row = _execute_control_turn(
        run_id="run1",
        dataset="gsm8k",
        split_name="count20_seed42",
        sample=sample,
        method_name="cot_1",
        method_type="control",
        round_index=0,
        agent_id=1,
        role="control",
        visible_peer_count=0,
        messages=[{"role": "system", "content": "demo"}, {"role": "user", "content": "demo"}],
        backbone=SimpleNamespace(name="demo"),
        provider=SimpleNamespace(),
        cache=SimpleNamespace(),
        throttle=SimpleNamespace(),
        temperature=0.7,
        top_p=1.0,
        seed=42,
        output_protocol=FREE_TEXT_ANSWER_PROTOCOL_V1,
    )

    assert captured["output_protocol"] == FREE_TEXT_ANSWER_PROTOCOL_V1
    assert row["output_protocol"] == FREE_TEXT_ANSWER_PROTOCOL_V1
    assert row["protocol_parse_status"] == "ok"
    assert row["reason_present"] is True
    assert row["raw_finish_reason"] == "stop"
    assert row["network_request_count"] == 1


def test_same_context_main_v5_configs_load_mainline_and_legacy_contracts() -> None:
    mainline = load_experiment_config(
        "configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml"
    )
    legacy = load_experiment_config(
        "configs/families/adaptive_sparse_mad/experiments/same_context_main_v5_json_legacy.toml"
    )
    controls = load_control_catalog(mainline.control_catalog)

    assert list(controls) == ["cot_1", "sc_3", "sc_5"]
    assert mainline.control_prompt_version == "single_agent_free_text_v1"
    assert mainline.control_output_protocol == "free_text_answer_v1"
    assert mainline.stage_a_response_format_mode == "free_text"
    assert mainline.adaptive_response_format_mode == "free_text"
    assert mainline.legacy_json_mode is False
    assert legacy.stage_a_response_format_mode == "json_object"
    assert legacy.adaptive_response_format_mode == "json_object"
    assert legacy.legacy_json_mode is True


def test_same_context_main_v9_config_loads_global_sync_free_text_contract() -> None:
    mainline = load_experiment_config(
        "configs/families/adaptive_sparse_mad/experiments/same_context_main_v9.toml"
    )
    protocol = load_protocol_config(mainline.protocol)
    controls = load_control_catalog(mainline.control_catalog)

    assert list(controls) == ["cot_1", "sc_3", "sc_5"]
    assert mainline.aggregate_methods == (COT_MAD_GLOBAL_SYNC_METHOD,)
    assert mainline.stage_a_prompt_version == COT_GLOBAL_SYNC_PROMPT_VERSION
    assert mainline.adaptive_prompt_version == COT_GLOBAL_SYNC_PROMPT_VERSION
    assert mainline.stage_a_response_format_mode == "free_text"
    assert mainline.adaptive_response_format_mode == "free_text"
    assert mainline.legacy_json_mode is False
    assert protocol.sync_board_max_chars == 1600
    assert protocol.family_evidence_max_chars == 180
    assert protocol.own_prior_max_chars == 120
    assert protocol.certificate_max_tokens == 256


def test_same_context_main_v5_control_messages_match_baseline_free_text_prompt() -> None:
    controls = load_control_catalog(
        "configs/families/adaptive_sparse_mad/controls/same_context_main_v5_controls.toml"
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )

    for control_name in ("cot_1", "sc_3", "sc_5"):
        method = controls[control_name]
        captured_messages: list[list[dict[str, str]]] = []
        captured_seeds: list[int | None] = []

        def fake_execute_turn(**kwargs):
            captured_messages.append(kwargs["messages"])
            captured_seeds.append(kwargs["seed"])
            return {
                "normalized_answer": "French",
                "prediction": "French",
                "score": 1.0,
                "output_status": "ok",
            }

        run_unified_control_sample(
            run_id="run1",
            benchmark_slug="hotpotqa",
            split_name="count20_seed42",
            sample=sample,
            control_name=control_name,
            method=method,
            backbone=SimpleNamespace(name="demo"),
            provider=SimpleNamespace(),
            cache=SimpleNamespace(),
            throttle=SimpleNamespace(),
            global_seed=42,
            prompt_version=FREE_TEXT_V1_PROMPT_VERSION,
            execute_turn=fake_execute_turn,
            build_prediction_row=lambda **kwargs: {"method_name": kwargs["control_name"]},
        )

        expected_messages = [
            build_cot_messages(sample, replicate_id + 1, FREE_TEXT_V1_PROMPT_VERSION)
            for replicate_id in range(method.budget_calls)
        ]
        assert captured_messages == expected_messages
        if control_name == "cot_1":
            assert captured_seeds == [42]
        else:
            assert captured_seeds == list(range(42, 42 + method.budget_calls))


def test_stage_a_answer_slot_safeguard_appends_language_for_hotpot_language_questions() -> None:
    repaired = _apply_stage_a_answer_slot_safeguard(
        "Tugurt",
        reasoning="The Tugurt language is closely related to Tumzabt and Teggargrent.",
        question="What language is traditionally written with the ancient Libyco-Berber script and closely related to Tumzabt and Teggargrent?",
        dataset="hotpotqa",
    )

    assert repaired == "Tugurt language"


def test_stage_a_answer_slot_safeguard_expands_hotpot_city_with_state() -> None:
    repaired = _apply_stage_a_answer_slot_safeguard(
        "Hollywood",
        reasoning="The Primetime Race Group is from Hollywood, Florida.",
        question="Which City in the Miami metropolitan area is home to the Primetime Race Group?",
        dataset="hotpotqa",
    )

    assert repaired == "Hollywood Florida"


def test_stage_a_answer_slot_safeguard_appends_students_for_count_question() -> None:
    repaired = _apply_stage_a_answer_slot_safeguard(
        "1840",
        reasoning="In the 2010-2011 school year, there were 1,840 students enrolled.",
        question="How many students were enrolled in American professional bowler Chris Barnes' high school in the 2010-2011 school year?",
        dataset="hotpotqa",
    )

    assert repaired == "1840 students"


def test_stage_a_answer_slot_safeguard_merges_england_location_parts() -> None:
    repaired = _apply_stage_a_answer_slot_safeguard(
        "South West England",
        reasoning="Belmont is near Lyme Regis in West Dorset, South West England.",
        question="John Fowles' country house was near Lyme Regis in what part of England?",
        dataset="hotpotqa",
    )

    assert repaired == "West Dorset South West England"


def test_stage_a_answer_slot_safeguard_strips_wbc_title_wrapper() -> None:
    repaired = _apply_stage_a_answer_slot_safeguard(
        "WBC cruiserweight title",
        reasoning="Tony Bellew held the WBC cruiserweight title from 2016 to 2017.",
        question="Creed features the boxer who held what WBC title from 2016 to 2017?",
        dataset="hotpotqa",
    )

    assert repaired == "cruiserweight"


def test_stage_a_answer_slot_safeguard_maps_multiple_choice_option_text() -> None:
    sample = DatasetSample(
        dataset="gpqa_diamond",
        sample_id="demo",
        question="Which option is correct?",
        reference_answer="B|||polyA tail",
        prompt_context="A. cap\nB. polyA tail\nC. spliceosome\nD. ribosome",
        metadata={"options": ["cap", "polyA tail", "spliceosome", "ribosome"]},
    )

    repaired = _apply_stage_a_answer_slot_safeguard(
        "polyA tail",
        reasoning="The best answer is the polyA tail.",
        question=sample.question,
        dataset="gpqa_diamond",
        sample=sample,
    )

    assert repaired == "B"


def test_stage_a_answer_slot_safeguard_marks_invalid_multiple_choice_answer_unknown() -> None:
    sample = DatasetSample(
        dataset="gpqa_diamond",
        sample_id="demo",
        question="Which option is correct?",
        reference_answer="B|||polyA tail",
        prompt_context="A. cap\nB. polyA tail\nC. spliceosome\nD. ribosome",
        metadata={"options": ["cap", "polyA tail", "spliceosome", "ribosome"]},
    )

    repaired = _apply_stage_a_answer_slot_safeguard(
        "no",
        reasoning="The answer is no.",
        question=sample.question,
        dataset="gpqa_diamond",
        sample=sample,
    )

    assert repaired == "unknown"


def test_build_policy_diagnostics_reports_stage_a_only_default() -> None:
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "overall",
                "method_name": "hetero_vote_3",
                "score": 1.0,
                "method_kind": "aggregate",
            }
        ],
        router_eval_payload={"summary_rows": []},
    )

    assert payload["policy_rows"] == []
    assert payload["recommended_next_default_policy"] == {
        "selected_policy": "hetero_vote_3",
        "reason": "stage_a_only_current_default",
    }


def test_build_policy_diagnostics_selects_adaptive_gate_when_it_wins() -> None:
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "adaptive_gate_v4",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 120.0,
                "completion_tokens_per_question": 60.0,
                "total_tokens_per_question": 180.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 12.0,
                "calls_per_question": 4,
                "triggered": True,
                "early_exit": False,
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
        ],
        router_eval_payload={"summary_rows": []},
    )

    assert payload["recommended_next_default_policy"]["selected_policy"] == "adaptive_gate_v4"
    assert payload["policy_rows"]
    pairwise_row = next(row for row in payload["pairwise_rows"] if row["method_name"] == "adaptive_gate_v4")
    assert "bootstrap_ci_low" in pairwise_row
    assert "bootstrap_ci_high" in pairwise_row
    assert "holm_adjusted_p" in pairwise_row
    assert any(
        row["dataset"] == "hotpotqa" and row["method_name"] == "adaptive_gate_v4"
        for row in payload["pairwise_rows"]
    )


def test_build_router_eval_payload_summarizes_adaptive_router_rows() -> None:
    payload = build_router_eval_payload(
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "policy_name": "adaptive_gate_v4",
                "triggered": True,
                "selected_addon_solver": "solver_evidence",
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "support_gap": 0.2,
                "avg_confidence": 0.4,
                "false_consensus_risk": True,
                "probe_accepted": True,
                "debate_after_probe_triggered": False,
                "baseline_score": 0.0,
                "stage_a_oracle_correct": True,
                "pre_route_correct": True,
                "stage_a_error_bucket": "clean_pseudo_majority",
                "high_value_bucket": True,
                "override_accepted": True,
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "s2",
                "policy_name": "adaptive_gate_v4",
                "triggered": False,
                "selected_addon_solver": "",
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "support_gap": 0.6,
                "avg_confidence": 0.8,
                "false_consensus_risk": False,
                "probe_accepted": False,
                "debate_after_probe_triggered": False,
                "baseline_score": 1.0,
                "stage_a_oracle_correct": True,
                "pre_route_correct": True,
                "stage_a_error_bucket": "stage_a_correct",
                "high_value_bucket": False,
                "override_accepted": False,
            },
        ]
    )

    overall = next(row for row in payload["summary_rows"] if row["dataset"] == "overall")
    assert overall["trigger_rate"] == 0.5
    assert overall["corrected_count"] == 1
    assert overall["false_consensus_risk_rate"] == 0.5
    assert overall["probe_accepted_count"] == 1
    assert overall["stage_a_oracle_accuracy"] == 1.0
    assert overall["oracle_gap_vs_hetero"] == 0.5
    assert overall["oracle_gap_capture_by_preroute"] == 1.0
    assert overall["high_value_trigger_precision"] == 1.0
    assert overall["high_value_trigger_recall"] == 1.0
    assert overall["correct_to_wrong_rate_on_stage_a_correct"] == 0.0
    clean_bucket = next(
        row
        for row in payload["bucket_rows"]
        if row["dataset"] == "overall" and row["stage_a_error_bucket"] == "clean_pseudo_majority"
    )
    assert clean_bucket["override_accepted_rate"] == 1.0
    assert clean_bucket["corrected_count"] == 1


def test_build_router_eval_payload_separates_policy_variants() -> None:
    payload = build_router_eval_payload(
        [
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "policy_name": "adaptive_gate_v4",
                "triggered": True,
                "selected_addon_solver": "solver_evidence",
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "support_gap": 0.2,
                "avg_confidence": 0.4,
                "false_consensus_risk": False,
                "probe_accepted": False,
                "debate_after_probe_triggered": False,
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "policy_name": "adaptive_dual_open_v5",
                "triggered": True,
                "selected_addon_solver": "solver_evidence",
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "support_gap": 0.2,
                "avg_confidence": 0.4,
                "false_consensus_risk": True,
                "probe_accepted": True,
                "debate_after_probe_triggered": True,
            },
        ]
    )

    adaptive_row = next(
        row
        for row in payload["summary_rows"]
        if row["dataset"] == "overall" and row["policy_name"] == "adaptive_gate_v4"
    )
    dual_open_row = next(
        row
        for row in payload["summary_rows"]
        if row["dataset"] == "overall" and row["policy_name"] == "adaptive_dual_open_v5"
    )
    assert adaptive_row["corrected_count"] == 1
    assert dual_open_row["corrected_count"] == 0
    assert dual_open_row["false_consensus_risk_rate"] == 1.0
    assert dual_open_row["probe_accepted_count"] == 1


def test_build_policy_diagnostics_preserves_router_summary_and_bucket_rows() -> None:
    router_eval_payload = {
        "summary_rows": [
            {
                "dataset": "overall",
                "policy_name": "adaptive_sparse_meta_route_v7",
                "trigger_rate": 0.3,
            }
        ],
        "bucket_rows": [
            {
                "dataset": "overall",
                "policy_name": "adaptive_sparse_meta_route_v7",
                "stage_a_error_bucket": "clean_pseudo_majority",
                "trigger_rate": 0.8,
            }
        ],
    }
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "overall",
                "method_name": "hetero_vote_3",
                "score": 1.0,
                "method_kind": "aggregate",
            }
        ],
        router_eval_payload=router_eval_payload,
    )

    assert payload["router_summary_rows"] == router_eval_payload["summary_rows"]
    assert payload["router_bucket_rows"] == router_eval_payload["bucket_rows"]


def test_default_meta_router_payload_assigns_typed_sequences() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )

    false_consensus_payload = _default_meta_router_payload(
        sample=sample,
        protocol=protocol,
        stage_a_rows=[
            {"solver_mode": "solver_cot", "normalized_answer": "french", "confidence_value": 0.9},
            {"solver_mode": "solver_l2m", "normalized_answer": "french", "confidence_value": 0.8},
            {"solver_mode": "solver_skeptic", "normalized_answer": "french", "confidence_value": 0.7},
        ],
        stage_a_answer="french",
        support={"french": 2.4},
        gate_decision={"false_consensus_risk": True, "avg_confidence": 0.8, "top_support": 2.4},
    )
    pseudo_majority_payload = _default_meta_router_payload(
        sample=sample,
        protocol=protocol,
        stage_a_rows=[
            {"solver_mode": "solver_cot", "normalized_answer": "french", "confidence_value": 0.9},
            {"solver_mode": "solver_l2m", "normalized_answer": "english", "confidence_value": 0.8},
            {"solver_mode": "solver_skeptic", "normalized_answer": "french", "confidence_value": 0.7},
        ],
        stage_a_answer="french",
        support={"french": 1.6, "english": 0.8},
        gate_decision={"avg_confidence": 0.8, "top_support": 1.6},
    )
    all_three_wrong_payload = _default_meta_router_payload(
        sample=sample,
        protocol=protocol,
        stage_a_rows=[
            {"solver_mode": "solver_cot", "normalized_answer": "french", "confidence_value": 0.9},
            {"solver_mode": "solver_l2m", "normalized_answer": "english", "confidence_value": 0.8},
            {"solver_mode": "solver_skeptic", "normalized_answer": "german", "confidence_value": 0.7},
        ],
        stage_a_answer="french",
        support={"french": 0.9, "english": 0.8, "german": 0.7},
        gate_decision={"avg_confidence": 0.8, "top_support": 0.9},
    )

    assert false_consensus_payload["error_mode"] == "false_consensus"
    assert false_consensus_payload["recommended_solver_sequence"] == ["solver_evidence"]
    assert pseudo_majority_payload["error_mode"] == "pseudo_majority"
    assert pseudo_majority_payload["recommended_solver_sequence"] == ["solver_evidence"]
    assert all_three_wrong_payload["error_mode"] == "all_three_wrong_suspect"
    assert all_three_wrong_payload["selected_candidate"] == "no_confident_candidate"
    assert all_three_wrong_payload["recommended_solver_sequence"] == ["solver_disconfirm", "solver_evidence"]


def test_resolve_v7_single_step_override_accepts_grounded_verifier_flip() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
        typed_override_margin=0.15,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "english",
            "prediction": "english",
            "score": 0.0,
            "confidence_value": 0.8,
            "claim_span": "English",
            "key_evidence": "English appears nearby",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "french",
            "prediction": "french",
            "score": 1.0,
            "confidence_value": 0.7,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "french",
            "prediction": "french",
            "score": 1.0,
            "confidence_value": 0.6,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_evidence",
            "normalized_answer": "french",
            "prediction": "french",
            "score": 1.0,
            "confidence_value": 0.9,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
    ]

    answer, support, resolver, override = _resolve_v7_single_step_override(
        sample=sample,
        benchmark_slug="hotpotqa",
        protocol=protocol,
        rows=rows,
        pre_route_answer="english",
        pre_route_support={"english": 0.8},
        pre_route_resolver="meta_router_head_v1:solver_cot",
    )

    assert answer == "french"
    assert resolver == "evidence_grounded_score_vote"
    assert support["french"] > support["english"]
    assert override["override_accepted"] is True
    assert override["override_rule"] == "typed_margin_override"
    assert override["override_margin"] >= 0.15


def test_resolve_v7_all_three_wrong_override_requires_double_support_and_novel_family() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    base_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "english",
            "prediction": "english",
            "score": 0.0,
            "confidence_value": 0.8,
            "claim_span": "English",
            "key_evidence": "English appears nearby",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "spanish",
            "prediction": "spanish",
            "score": 0.0,
            "confidence_value": 0.7,
            "claim_span": "Spanish",
            "key_evidence": "Spanish appears nearby",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "german",
            "prediction": "german",
            "score": 0.0,
            "confidence_value": 0.6,
            "claim_span": "German",
            "key_evidence": "German appears nearby",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
    ]
    supportive_addon_rows = [
        {
            "solver_mode": "solver_disconfirm",
            "normalized_answer": "french",
            "prediction": "french",
            "score": 1.0,
            "confidence_value": 0.8,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_evidence",
            "normalized_answer": "french",
            "prediction": "french",
            "score": 1.0,
            "confidence_value": 0.9,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
    ]

    accepted_answer, _accepted_support, accepted_resolver, accepted_override = _resolve_v7_all_three_wrong_override(
        sample=sample,
        benchmark_slug="hotpotqa",
        protocol=protocol,
        rows=[*base_rows, *supportive_addon_rows],
        addon_rows=supportive_addon_rows,
        pre_route_answer="english",
        pre_route_support={"english": 0.8},
        pre_route_resolver="meta_router_head_v1:solver_cot",
    )

    assert accepted_answer == "french"
    assert accepted_resolver in {"family_slot_grounded_score_vote", "evidence_grounded_score_vote"}
    assert accepted_override["override_accepted"] is True
    assert accepted_override["all_three_wrong_chain_supported"] is True

    rejected_answer, _rejected_support, rejected_resolver, rejected_override = _resolve_v7_all_three_wrong_override(
        sample=sample,
        benchmark_slug="hotpotqa",
        protocol=protocol,
        rows=[*base_rows, *supportive_addon_rows],
        addon_rows=[
            supportive_addon_rows[0],
            {
                **supportive_addon_rows[1],
                "normalized_answer": "spanish",
                "prediction": "spanish",
                "score": 0.0,
                "claim_span": "Spanish",
                "key_evidence": "Spanish is spoken",
            },
        ],
        pre_route_answer="english",
        pre_route_support={"english": 0.8},
        pre_route_resolver="meta_router_head_v1:solver_cot",
    )

    assert rejected_answer == "english"
    assert rejected_resolver == "meta_router_head_v1:solver_cot"
    assert rejected_override["override_accepted"] is False


def test_family_slot_grounded_stage_a_prefers_short_exact_open_qa_span() -> None:
    rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "part of flotilla",
            "confidence_value": 0.5,
            "claim_span": "part of flotilla",
            "key_evidence": "part of the flotilla that attacked the radar station",
            "validated_output": {"answer_type": "phrase", "key_constraints": "short exact answer span"},
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "flotilla",
            "confidence_value": 0.5,
            "claim_span": "flotilla",
            "key_evidence": "part of the flotilla that attacked the radar station",
            "validated_output": {"answer_type": "noun phrase", "key_constraints": "short exact answer span"},
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "flotilla",
            "confidence_value": 0.5,
            "claim_span": "flotilla",
            "key_evidence": "part of the flotilla that attacked the radar station",
            "validated_output": {"answer_type": "noun phrase", "key_constraints": "short exact answer span"},
        },
    ]

    answer, support, resolver = aggregate_family_slot_grounded_stage_a(
        rows,
        dataset="hotpotqa",
        question="What was he on during the attack?",
    )

    assert answer == "flotilla"
    assert resolver in {"family_slot_grounded_score_vote", "family_slot_grounded_rescue"}
    assert support


def test_build_policy_diagnostics_handles_dual_open_variant() -> None:
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3.0,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "s1",
                "method_name": "adaptive_dual_open_v5",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 120.0,
                "completion_tokens_per_question": 60.0,
                "total_tokens_per_question": 180.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 12.0,
                "calls_per_question": 3.0,
                "triggered": True,
                "early_exit": False,
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
        ],
        router_eval_payload={"summary_rows": []},
    )

    assert any(row["method_name"] == "adaptive_dual_open_v5" for row in payload["policy_rows"])
    assert any(row["method_name"] == "adaptive_dual_open_v5" for row in payload["pairwise_rows"])


def test_adaptive_gate_decision_skips_strong_clean_consensus() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is used there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "french",
            "confidence_value": 0.98,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "french",
            "confidence_value": 0.97,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "french",
            "confidence_value": 0.96,
            "claim_span": "French",
            "key_evidence": "French is spoken",
            "answer_type": "language",
            "key_constraints": "short exact span",
        },
    ]

    decision = _build_adaptive_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        support={"french": 2.91},
    )

    assert decision["triggered"] is False
    assert decision["trigger_reasons"] == []


def test_adaptive_gate_decision_triggers_on_disagreement_with_structural_issue() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which person led the force?",
        reference_answer="Captain John Underhill",
        prompt_context="Captain John Underhill led the force.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "john underhill",
            "confidence_value": 0.98,
            "claim_span": "John Underhill",
            "key_evidence": "John Underhill led the force",
            "answer_type": "person",
            "key_constraints": "exact person name",
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "john underhill",
            "confidence_value": 0.97,
            "claim_span": "John Underhill",
            "key_evidence": "John Underhill led the force",
            "answer_type": "person",
            "key_constraints": "exact person name",
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "captain john underhill",
            "confidence_value": 0.95,
            "claim_span": "Captain John Underhill",
            "key_evidence": "Captain John Underhill led the force",
            "answer_type": "named person with title",
            "key_constraints": "include the title when it is part of the exact answer span",
        },
    ]

    decision = _build_adaptive_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        support={"john underhill": 1.95, "captain john underhill": 0.95},
    )

    assert decision["triggered"] is True
    assert "answer_disagreement" in decision["trigger_reasons"]
    assert "answer_type_conflict" in decision["trigger_reasons"]


def test_adaptive_gate_decision_does_not_treat_missing_confidence_as_low_confidence() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="competition_math",
        sample_id="demo",
        question="Find the value of x.",
        reference_answer="3",
        prompt_context=None,
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "3",
            "confidence_value": 0.5,
            "confidence_valid": False,
        },
        {
            "solver_mode": "solver_l2m",
            "normalized_answer": "3",
            "confidence_value": 0.5,
            "confidence_valid": False,
        },
        {
            "solver_mode": "solver_skeptic",
            "normalized_answer": "3",
            "confidence_value": 0.5,
            "confidence_valid": False,
        },
    ]

    decision = _build_adaptive_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        support={"3": 1.5},
    )

    assert decision["triggered"] is False
    assert "low_confidence_consensus" not in decision["trigger_reasons"]
    assert decision["valid_confidence_count"] == 0
    assert decision["avg_confidence"] is None


def test_adaptive_gate_decision_uses_protocol_thresholds_without_hidden_floor() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is correct?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.92, "confidence_valid": True},
        {"solver_mode": "solver_l2m", "normalized_answer": "A", "confidence_value": 0.9, "confidence_valid": True},
        {"solver_mode": "solver_skeptic", "normalized_answer": "B", "confidence_value": 0.88, "confidence_valid": True},
    ]

    decision = _build_adaptive_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        support={"A": 1.82, "B": 0.88},
    )

    assert decision["triggered"] is False
    assert "narrow_support_gap" not in decision["trigger_reasons"]


def test_global_sync_gate_decision_skips_clean_five_way_consensus() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "french",
            "confidence_value": 0.91,
            "confidence_valid": True,
            "key_evidence": "French is spoken in Quebec.",
            "answer_type": "language",
            "key_constraints": "short exact span",
        }
        for _ in range(5)
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="french",
        stage_a_vote_counts={"french": 5},
    )

    assert decision["triggered"] is False
    assert decision["trigger_reasons"] == []
    assert decision["vote_pattern"] == "5"


def test_global_sync_gate_decision_skips_four_to_one_pattern_without_majority_risk() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is correct?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.88, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.86, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.84, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.82, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "B", "confidence_value": 0.74, "confidence_valid": True},
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="A",
        stage_a_vote_counts={"A": 4, "B": 1},
    )

    assert decision["triggered"] is False
    assert decision["trigger_reasons"] == []
    assert decision["vote_pattern"] == "4-1"


def test_global_sync_gate_decision_triggers_on_three_to_two_pattern() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="strategyqa",
        sample_id="demo",
        question="Could a black widow woman have use for peaches?",
        reference_answer="yes",
        prompt_context="",
        metadata={},
    )
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "yes", "confidence_value": 0.76, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "yes", "confidence_value": 0.74, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "yes", "confidence_value": 0.72, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "no", "confidence_value": 0.71, "confidence_valid": True},
        {"solver_mode": "solver_cot", "normalized_answer": "no", "confidence_value": 0.69, "confidence_valid": True},
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="yes",
        stage_a_vote_counts={"yes": 3, "no": 2},
    )

    assert decision["triggered"] is True
    assert "non_unanimous_vote" in decision["trigger_reasons"]
    assert decision["vote_pattern"] == "3-2"


def test_global_sync_gate_decision_records_protocol_failure_consensus_without_triggering_certificate_sync() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "french",
            "confidence_value": 0.87,
            "confidence_valid": True,
        }
        for _ in range(4)
    ] + [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "french",
            "confidence_value": 0.87,
            "confidence_valid": True,
            "stage_a_safe_retry_used": True,
        }
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="french",
        stage_a_vote_counts={"french": 5},
    )

    assert decision["triggered"] is False
    assert "protocol_failure_risk" in decision["trigger_reasons"]


def test_global_sync_gate_decision_records_low_confidence_unanimous_consensus_without_triggering_certificate_sync() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="mmlu_pro",
        sample_id="demo",
        question="Which option is correct?",
        reference_answer="A|||alpha",
        prompt_context="A. alpha\nB. beta",
        metadata={"options": ["alpha", "beta"]},
    )
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A", "confidence_value": 0.42, "confidence_valid": True}
        for _ in range(5)
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="A",
        stage_a_vote_counts={"A": 5},
    )

    assert decision["triggered"] is False
    assert "low_confidence_consensus" in decision["trigger_reasons"]


def test_global_sync_gate_decision_does_not_treat_failure_risk_none_as_signal() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="math500",
        sample_id="demo",
        question="What is 1 + 1?",
        reference_answer="2",
        prompt_context="",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "2",
            "confidence_value": 1.0,
            "confidence_valid": True,
            "failure_risk": "none",
            "uncertainty_type": "",
        }
        for _ in range(5)
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="2",
        stage_a_vote_counts={"2": 5},
    )

    assert decision["triggered"] is False
    assert "repeated_risk_signals" not in decision["trigger_reasons"]


def test_global_sync_gate_decision_triggers_on_four_to_one_pattern_with_majority_invalid() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=5,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which court decided the case?",
        reference_answer="Supreme Court",
        prompt_context="The Supreme Court decided the case.",
        metadata={},
    )
    stage_a_rows = [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "court",
            "confidence_value": 0.9,
            "confidence_valid": True,
            "validated_output": {"format_warning": "answer_outside_expected_slot"},
        }
        for _ in range(4)
    ] + [
        {
            "solver_mode": "solver_cot",
            "normalized_answer": "supreme court",
            "confidence_value": 0.8,
            "confidence_valid": True,
        }
    ]

    decision = _build_global_sync_gate_decision(
        sample=sample,
        protocol=protocol,
        stage_a_rows=stage_a_rows,
        stage_a_majority_answer="court",
        stage_a_vote_counts={"court": 4, "supreme court": 1},
    )

    assert decision["triggered"] is True
    assert "invalid_majority_answer" in decision["trigger_reasons"]


def test_build_policy_diagnostics_emits_dataset_and_overall_pairwise_rows() -> None:
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "adaptive_gate_v4",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 120.0,
                "completion_tokens_per_question": 60.0,
                "total_tokens_per_question": 180.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 12.0,
                "calls_per_question": 4,
                "triggered": True,
                "early_exit": False,
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "g1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 90.0,
                "completion_tokens_per_question": 30.0,
                "total_tokens_per_question": 120.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 9.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "g1",
                "method_name": "adaptive_gate_v4",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 110.0,
                "completion_tokens_per_question": 40.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 11.0,
                "calls_per_question": 4,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "judge_fallback_used": False,
                "model_name": "demo",
            },
        ],
        router_eval_payload={"summary_rows": []},
    )

    dataset_pairs = [
        row for row in payload["pairwise_rows"]
        if row["method_name"] == "adaptive_gate_v4" and row["baseline_method_name"] == "hetero_vote_3"
    ]
    assert {row["dataset"] for row in dataset_pairs} == {"gsm8k", "hotpotqa", "overall"}


def test_build_policy_diagnostics_exposes_promotion_gate_summary() -> None:
    payload = build_policy_diagnostics(
        prediction_rows=[
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "model_name": "demo",
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h1",
                "method_name": "adaptive_counterfactual_v1",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 120.0,
                "completion_tokens_per_question": 60.0,
                "total_tokens_per_question": 180.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 12.0,
                "calls_per_question": 4,
                "triggered": True,
                "early_exit": False,
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "model_name": "demo",
            },
            {
                "dataset": "mmlu_pro",
                "sample_id": "m1",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 0.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "model_name": "demo",
            },
            {
                "dataset": "mmlu_pro",
                "sample_id": "m1",
                "method_name": "adaptive_counterfactual_v1",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 120.0,
                "completion_tokens_per_question": 60.0,
                "total_tokens_per_question": 180.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 12.0,
                "calls_per_question": 4,
                "triggered": True,
                "early_exit": False,
                "changed_answer": True,
                "corrected_by_method": True,
                "harmed_by_method": False,
                "model_name": "demo",
            },
            {
                "dataset": "hotpotqa",
                "sample_id": "h2",
                "method_name": "hetero_vote_3",
                "method_kind": "aggregate",
                "score": 1.0,
                "prompt_tokens_per_question": 100.0,
                "completion_tokens_per_question": 50.0,
                "total_tokens_per_question": 150.0,
                "communication_tokens_per_question": 0.0,
                "latency_ms_per_question": 10.0,
                "calls_per_question": 3,
                "triggered": False,
                "early_exit": True,
                "changed_answer": False,
                "corrected_by_method": False,
                "harmed_by_method": False,
                "model_name": "demo",
            },
        ],
        router_eval_payload={"summary_rows": []},
    )

    promotion_gate = payload["promotion_gate"]
    candidate_lookup = {row["method_name"]: row for row in promotion_gate["candidate_rows"]}
    assert candidate_lookup["adaptive_counterfactual_v1"]["promote_to_count100"] is True
    assert candidate_lookup["adaptive_counterfactual_v1"]["positive_categories"] == ["mcqa", "open_qa"]
    assert candidate_lookup["adaptive_counterfactual_v1"]["mainline_ready_signal"] is True
    mainline_gate = payload["mainline_gate"]
    mainline_lookup = {row["method_name"]: row for row in mainline_gate["candidate_rows"]}
    assert mainline_lookup["adaptive_counterfactual_v1"]["eligible_for_mainline_assessment"] is False
    assert mainline_lookup["adaptive_counterfactual_v1"]["mainline_ready"] is False


def test_build_policy_diagnostics_marks_mainline_ready_when_count100_conditions_hold() -> None:
    prediction_rows: list[dict[str, object]] = []
    positive_samples = {
        *(("hotpotqa", index) for index in range(10)),
        *(("mmlu_pro", index) for index in range(8)),
        *(("math500", index) for index in range(6)),
    }
    dataset_order = [
        "hotpotqa",
        "competition_math",
        "mmlu_pro",
        "gpqa_diamond",
        "math500",
        "strategyqa",
        "gsm8k",
    ]
    for dataset in dataset_order:
        for index in range(100):
            sample_id = f"{dataset}-{index:03d}"
            baseline_correct = True
            method_correct = True
            if (dataset, index) in positive_samples:
                baseline_correct = False
            score_baseline = 1.0 if baseline_correct else 0.0
            score_method = 1.0 if method_correct else 0.0
            prediction_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "method_name": "hetero_vote_3",
                    "method_kind": "aggregate",
                    "score": score_baseline,
                    "prompt_tokens_per_question": 100.0,
                    "completion_tokens_per_question": 50.0,
                    "total_tokens_per_question": 150.0,
                    "communication_tokens_per_question": 0.0,
                    "latency_ms_per_question": 10.0,
                    "calls_per_question": 3,
                    "triggered": False,
                    "early_exit": True,
                    "changed_answer": False,
                    "corrected_by_method": False,
                    "harmed_by_method": False,
                    "model_name": "demo",
                }
            )
            prediction_rows.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "method_name": "adaptive_counterfactual_v1",
                    "method_kind": "aggregate",
                    "score": score_method,
                    "prompt_tokens_per_question": 120.0,
                    "completion_tokens_per_question": 60.0,
                    "total_tokens_per_question": 180.0,
                    "communication_tokens_per_question": 0.0,
                    "latency_ms_per_question": 12.0,
                    "calls_per_question": 4,
                    "triggered": dataset in {"hotpotqa", "mmlu_pro", "math500"},
                    "early_exit": dataset not in {"hotpotqa", "mmlu_pro", "math500"},
                    "changed_answer": (dataset, index) in positive_samples,
                    "corrected_by_method": (dataset, index) in positive_samples,
                    "harmed_by_method": False,
                    "model_name": "demo",
                }
            )

    payload = build_policy_diagnostics(
        prediction_rows=prediction_rows,
        router_eval_payload={"summary_rows": []},
    )

    mainline_lookup = {row["method_name"]: row for row in payload["mainline_gate"]["candidate_rows"]}
    candidate = mainline_lookup["adaptive_counterfactual_v1"]
    assert candidate["eligible_for_mainline_assessment"] is True
    assert candidate["mainline_ready"] is True
    assert candidate["core_category_positive_count"] == 3
    assert candidate["negative_datasets"] == []


def test_load_experiment_config_requires_v4_stage_a_for_structured_methods(tmp_path: Path) -> None:
    experiment_path = tmp_path / "bad_adaptive.toml"
    experiment_path.write_text(
        '\n'.join(
            [
                'name = "bad_adaptive"',
                'description = "demo"',
                'benchmark_configs = ["configs/core/shared/benchmarks/gsm8k/test.toml"]',
                'protocol = "configs/families/adaptive_sparse_mad/protocols/shared_pair_sparse.toml"',
                'control_catalog = "configs/families/adaptive_sparse_mad/controls/same_context_controls.toml"',
                'aggregate_methods = ["hetero_vote_3", "adaptive_gate_v4"]',
                "global_seed = 42",
                'prompt_version = "adaptive_sparse_mad_v4_evidence_gate"',
                'stage_a_prompt_version = "adaptive_sparse_mad_v2_task_schema"',
                'adaptive_prompt_version = "adaptive_sparse_mad_v4_evidence_gate"',
                'primary_model_ref = "xiaomimimo/mimo-v2.5"',
                "",
                "[phases.count20]",
                'split_overrides = { gsm8k = "count20_seed42" }',
            ]
        ),
        encoding="utf-8",
    )

    from research_experiments.families.adaptive_sparse_mad.config import load_experiment_config

    with pytest.raises(ValueError, match="stage_a_prompt_version in"):
        load_experiment_config(experiment_path)


def test_load_experiment_config_rejects_json_mode_for_global_sync_method(tmp_path: Path) -> None:
    experiment_path = tmp_path / "bad_global_sync.toml"
    experiment_path.write_text(
        "\n".join(
            [
                'name = "bad_global_sync"',
                'description = "demo"',
                'benchmark_configs = ["configs/core/shared/benchmarks/gsm8k/test.toml"]',
                'protocol = "configs/families/adaptive_sparse_mad/protocols/global_sync_cot5.toml"',
                'control_catalog = "configs/families/adaptive_sparse_mad/controls/same_context_main_v5_controls.toml"',
                f'aggregate_methods = ["{COT_MAD_GLOBAL_SYNC_METHOD}"]',
                "global_seed = 42",
                f'prompt_version = "{COT_GLOBAL_SYNC_PROMPT_VERSION}"',
                f'stage_a_prompt_version = "{COT_GLOBAL_SYNC_PROMPT_VERSION}"',
                f'adaptive_prompt_version = "{COT_GLOBAL_SYNC_PROMPT_VERSION}"',
                'stage_a_response_format_mode = "json_object"',
                'adaptive_response_format_mode = "json_object"',
                'primary_model_ref = "xiaomimimo/mimo-v2.5"',
                "",
                "[phases.count20]",
                'split_overrides = { gsm8k = "count20_seed42" }',
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="free-text"):
        load_experiment_config(experiment_path)


def test_adaptive_dual_open_v5_uses_slot_contrast_on_severe_open_qa_uncertainty() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which gaming console were both games released on?",
        reference_answer="PlayStation 4",
        prompt_context="Both games were released for PlayStation 3 and PlayStation 4.",
        metadata={},
    )
    gate_decision = {
        "triggered": True,
        "selected_addon_solver": "solver_evidence",
        "trigger_reasons": ["answer_disagreement", "narrow_support_gap"],
    }

    sequence = _select_adaptive_addon_solver_sequence(
        method_name="adaptive_dual_open_v5",
        sample=sample,
        gate_decision=gate_decision,
    )

    assert sequence == ["solver_evidence", "solver_slot_contrast"]


def test_adaptive_counterfactual_v1_uses_counterfactual_sequence_on_collapse_risk() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which gaming console were both games released on?",
        reference_answer="PlayStation 4",
        prompt_context="Both games were released for PlayStation 3 and PlayStation 4.",
        metadata={},
    )
    gate_decision = {
        "triggered": True,
        "selected_addon_solver": "solver_evidence",
        "trigger_reasons": ["degraded_output", "self_reported_risk"],
        "unique_answer_count": 1,
        "support_gap": 0.0,
        "degraded_count": 1,
        "unknown_count": 0,
    }

    sequence = _select_adaptive_addon_solver_sequence(
        method_name="adaptive_counterfactual_v1",
        sample=sample,
        gate_decision=gate_decision,
    )

    assert sequence == ["solver_evidence", "solver_counterfactual"]


def test_counterfactual_override_rejects_strong_single_family_consensus() -> None:
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which Blackzilians fighter is currently competing in the UFC Middleweight division?",
        reference_answer="Vitor Belfort",
        prompt_context="The Blackzilians includes Vitor Belfort and Rashad Evans. Both are currently competing in the UFC Middleweight division.",
        metadata={},
    )
    gate_decision = {
        "trigger_reasons": ["degraded_output"],
        "unique_answer_count": 1,
        "support_gap": 2.0,
        "top_support": 2.0,
        "avg_confidence": 1.0,
    }
    counterfactual_row = {
        "normalized_answer": "rashad evans",
        "answer_type": "fighter name",
        "key_constraints": "Must be a Blackzilians fighter, currently competing in UFC Middleweight division.",
        "claim_span": "Rashad Evans",
        "key_evidence": "Rashad Evans is currently competing in the Middleweight division.",
    }

    accepted = _should_accept_counterfactual_override(
        counterfactual_row=counterfactual_row,
        baseline_answer="vitor belfort",
        gate_decision=gate_decision,
        sample=sample,
    )

    assert accepted is False


def test_counterfactual_override_rejects_high_confidence_mcqa_collapse_without_unknown_signal() -> None:
    sample = DatasetSample(
        dataset="gpqa_diamond",
        sample_id="demo",
        question="Which option best explains the effect?",
        reference_answer="C|||correct",
        prompt_context="A. introns\nB. centromeres\nC. active promoters and enhancers\nD. telomeres",
        metadata={"options": ["introns", "centromeres", "active promoters and enhancers", "telomeres"]},
    )
    gate_decision = {
        "trigger_reasons": ["degraded_output", "self_reported_risk"],
        "unique_answer_count": 1,
        "support_gap": 2.15,
        "top_support": 2.15,
        "avg_confidence": 0.825,
    }
    counterfactual_row = {
        "normalized_answer": "A",
        "answer_type": "option",
        "key_constraints": "Single best option; final answer only letter",
        "claim_span": "A",
        "key_evidence": "DSG stabilizes intronic interactions.",
    }

    accepted = _should_accept_counterfactual_override(
        counterfactual_row=counterfactual_row,
        baseline_answer="C",
        gate_decision=gate_decision,
        sample=sample,
    )

    assert accepted is False


def test_answers_share_family_distinguishes_family_expansion_from_real_contrast() -> None:
    assert _answers_share_family("1840 students", "1840 students enrolled") is True
    assert _answers_share_family("1991 perfect storm", "perfect storm") is True
    assert _answers_share_family("playstation 3", "playstation 4") is False


def test_build_stage_a_resolver_breakdown_payload_counts_resolver_hits() -> None:
    stage_a_rows = [
        {"dataset": "gsm8k", "sample_id": "s1", "solver_mode": "solver_cot", "normalized_answer": "A", "score": 1.0},
        {"dataset": "gsm8k", "sample_id": "s1", "solver_mode": "solver_l2m", "normalized_answer": "B", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "s1", "solver_mode": "solver_skeptic", "normalized_answer": "B", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "s2", "solver_mode": "solver_cot", "normalized_answer": "A", "score": 0.0},
        {"dataset": "gsm8k", "sample_id": "s2", "solver_mode": "solver_l2m", "normalized_answer": "B", "score": 1.0},
        {"dataset": "gsm8k", "sample_id": "s2", "solver_mode": "solver_skeptic", "normalized_answer": "B", "score": 1.0},
    ]
    prediction_rows = [
        {
            "dataset": "gsm8k",
            "sample_id": "s1",
            "method_name": "hetero_vote_3",
            "stage_a_resolver": "constraint_aware_clean_anchor_minority_override",
            "prediction": "A",
            "score": 1.0,
            "stage_a_weighted_support": {"A": 0.5, "B": 1.0},
        },
        {
            "dataset": "gsm8k",
            "sample_id": "s2",
            "method_name": "hetero_vote_3",
            "stage_a_resolver": "constraint_aware_anchor_vote",
            "prediction": "A",
            "score": 0.0,
            "stage_a_weighted_support": {"A": 0.5, "B": 1.0},
        },
    ]

    payload = build_stage_a_resolver_breakdown_payload(stage_a_rows, prediction_rows)
    summary_lookup = {
        (row["dataset"], row["resolver"]): row
        for row in payload["summary_rows"]
    }

    assert summary_lookup[("gsm8k", "constraint_aware_clean_anchor_minority_override")] == {
        "dataset": "gsm8k",
        "resolver": "constraint_aware_clean_anchor_minority_override",
        "total": 1,
        "correct": 1,
        "wrong": 0,
        "accuracy_mean": 1.0,
    }
    assert summary_lookup[("gsm8k", "constraint_aware_anchor_vote")] == {
        "dataset": "gsm8k",
        "resolver": "constraint_aware_anchor_vote",
        "total": 1,
        "correct": 0,
        "wrong": 1,
        "accuracy_mean": 0.0,
    }
    assert payload["example_rows"][0]["resolver"] == "constraint_aware_anchor_vote"


def test_build_stage_a_error_bucket_payload_classifies_remaining_error_types() -> None:
    stage_a_rows = [
        {
            "dataset": "overall",
            "sample_id": "all_wrong",
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "score": 0.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "all_wrong",
            "solver_mode": "solver_l2m",
            "normalized_answer": "B",
            "score": 0.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "all_wrong",
            "solver_mode": "solver_skeptic",
            "normalized_answer": "C",
            "score": 0.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "pseudo_majority",
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "score": 0.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "pseudo_majority",
            "solver_mode": "solver_l2m",
            "normalized_answer": "A",
            "score": 0.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "pseudo_majority",
            "solver_mode": "solver_skeptic",
            "normalized_answer": "B",
            "score": 1.0,
            "confidence_valid": False,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "confidence_case",
            "solver_mode": "solver_cot",
            "normalized_answer": "A",
            "score": 0.0,
            "confidence_valid": True,
            "confidence_value": 0.95,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "confidence_case",
            "solver_mode": "solver_l2m",
            "normalized_answer": "A",
            "score": 0.0,
            "confidence_valid": True,
            "confidence_value": 0.88,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "confidence_case",
            "solver_mode": "solver_skeptic",
            "normalized_answer": "B",
            "score": 1.0,
            "confidence_valid": True,
            "confidence_value": 0.22,
            "validated_output": {},
        },
        {
            "dataset": "overall",
            "sample_id": "constraint_case",
            "solver_mode": "solver_cot",
            "normalized_answer": "yes",
            "score": 0.0,
            "confidence_valid": False,
            "answer_type": "option_letter",
            "validated_output": {"answer_type": "option_letter"},
        },
        {
            "dataset": "overall",
            "sample_id": "constraint_case",
            "solver_mode": "solver_l2m",
            "normalized_answer": "yes",
            "score": 0.0,
            "confidence_valid": False,
            "answer_type": "option_letter",
            "validated_output": {"answer_type": "option_letter"},
        },
        {
            "dataset": "overall",
            "sample_id": "constraint_case",
            "solver_mode": "solver_skeptic",
            "normalized_answer": "B",
            "score": 1.0,
            "confidence_valid": False,
            "answer_type": "option_letter",
            "validated_output": {"answer_type": "option_letter"},
        },
    ]
    prediction_rows = [
        {"dataset": "overall", "sample_id": "all_wrong", "method_name": "hetero_vote_3", "prediction": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "pseudo_majority", "method_name": "hetero_vote_3", "prediction": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "confidence_case", "method_name": "hetero_vote_3", "prediction": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "constraint_case", "method_name": "hetero_vote_3", "prediction": "yes", "score": 0.0},
    ]

    payload = build_stage_a_error_bucket_payload(stage_a_rows, prediction_rows)

    assert payload["summary"]["error_count"] == 4
    assert payload["summary"]["all_three_wrong"] == 1
    assert payload["summary"]["clean_pseudo_majority"] == 1
    assert payload["summary"]["confidence_miscalibration"] == 1
    assert payload["summary"]["constraint_mismatch"] == 1
    bucket_lookup = {row["sample_id"]: row["bucket"] for row in payload["sample_rows"]}
    assert bucket_lookup["all_wrong"] == "all_three_wrong"
    assert bucket_lookup["pseudo_majority"] == "clean_pseudo_majority"
    assert bucket_lookup["confidence_case"] == "confidence_miscalibration"
    assert bucket_lookup["constraint_case"] == "constraint_mismatch"


def test_build_stage_a_solver_contribution_payload_counts_unique_and_shared_correctness() -> None:
    stage_a_rows = [
        {"dataset": "overall", "sample_id": "s1", "solver_mode": "solver_cot", "normalized_answer": "A", "score": 1.0},
        {"dataset": "overall", "sample_id": "s1", "solver_mode": "solver_l2m", "normalized_answer": "A", "score": 1.0},
        {"dataset": "overall", "sample_id": "s1", "solver_mode": "solver_skeptic", "normalized_answer": "B", "score": 0.0},
        {"dataset": "overall", "sample_id": "s2", "solver_mode": "solver_cot", "normalized_answer": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "s2", "solver_mode": "solver_l2m", "normalized_answer": "B", "score": 1.0},
        {"dataset": "overall", "sample_id": "s2", "solver_mode": "solver_skeptic", "normalized_answer": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "s3", "solver_mode": "solver_cot", "normalized_answer": "A", "score": 0.0},
        {"dataset": "overall", "sample_id": "s3", "solver_mode": "solver_l2m", "normalized_answer": "B", "score": 0.0},
        {"dataset": "overall", "sample_id": "s3", "solver_mode": "solver_skeptic", "normalized_answer": "C", "score": 0.0},
    ]

    payload = build_stage_a_solver_contribution_payload(stage_a_rows)
    overall = next(row for row in payload["summary_rows"] if row["dataset"] == "overall")

    assert overall["question_count"] == 3
    assert overall["all_three_wrong"] == 1
    assert overall["any_correct_solver_cot"] == 1
    assert overall["any_correct_solver_l2m"] == 2
    assert overall["any_correct_solver_skeptic"] == 0
    assert overall["solo_correct_solver_l2m"] == 1
    assert overall["majority_wrong_but_solver_right_solver_l2m"] == 1


def test_refresh_stage_a_prediction_rows_recomputes_hetero_answer() -> None:
    stage_a_rows = [
        {
            "dataset": "mmlu_pro",
            "sample_id": "s1",
            "solver_mode": "solver_cot",
            "normalized_answer": "E",
            "score": 0.0,
            "confidence_value": 0.5,
            "reasoning": "Exports exceed imports because foreign demand rises.",
            "validated_output": {},
        },
        {
            "dataset": "mmlu_pro",
            "sample_id": "s1",
            "solver_mode": "solver_l2m",
            "normalized_answer": "E",
            "score": 0.0,
            "confidence_value": 0.5,
            "reasoning": "Exports exceed imports because foreign demand rises.",
            "validated_output": {"answer_type": "multiple_choice", "key_constraints": "single option letter"},
        },
        {
            "dataset": "mmlu_pro",
            "sample_id": "s1",
            "solver_mode": "solver_skeptic",
            "normalized_answer": "A",
            "score": 1.0,
            "confidence_value": 0.5,
            "reasoning": "Low domestic income reduces import demand, increasing surplus.",
            "validated_output": {"answer_type": "multiple-choice", "key_constraints": "option letter only"},
        },
    ]
    prediction_rows = [
        {
            "dataset": "mmlu_pro",
            "sample_id": "s1",
            "method_name": "hetero_vote_3",
            "prediction": "E",
            "normalized_answer": "E",
            "score": 0.0,
            "stage_a_resolver": "constraint_aware_anchor_vote",
            "stage_a_weighted_support": {"E": 1.0, "A": 0.5},
            "average_confidence": 0.5,
        },
        {
            "dataset": "mmlu_pro",
            "sample_id": "s1",
            "method_name": "cot_1",
            "prediction": "E",
            "normalized_answer": "E",
            "score": 0.0,
        },
    ]

    refreshed = refresh_stage_a_prediction_rows(
        stage_a_rows,
        prediction_rows,
        prompt_version=STAGE_A_V2_PROMPT_VERSION,
    )

    hetero_row = next(row for row in refreshed if row["method_name"] == "hetero_vote_3")
    assert hetero_row["prediction"] == "A"
    assert hetero_row["score"] == 1.0
    assert hetero_row["stage_a_resolver"] == "constraint_aware_clean_skeptic_minority_override"
    control_row = next(row for row in refreshed if row["method_name"] == "cot_1")
    assert control_row["prediction"] == "E"


def test_refresh_prediction_rows_for_run_replays_adaptive_counterfactual_policy() -> None:
    protocol = AdaptiveSparseMadProtocolConfig(
        agent_count=3,
        top_p=1.0,
        stage_a_temperature=0.7,
        consensus_confidence_threshold=0.65,
        majority_confidence_threshold=0.6,
        majority_margin_threshold=0.25,
    )
    sample = DatasetSample(
        dataset="strategyqa",
        sample_id="s1",
        question="Could a black widow woman have use for peaches?",
        reference_answer="yes",
        prompt_context="",
        metadata={},
    )
    stage_a_rows = [
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "solver_mode": "solver_cot",
            "agent_id": 1,
            "normalized_answer": "no",
            "prediction": "no",
            "score": 0.0,
            "confidence_value": 0.95,
            "confidence_valid": True,
            "reasoning": "Black widow spiders are carnivorous.",
            "validated_output": {"answer_type": "yes/no", "key_constraints": "answer yes or no"},
        },
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "solver_mode": "solver_l2m",
            "agent_id": 2,
            "normalized_answer": "no",
            "prediction": "no",
            "score": 0.0,
            "confidence_value": 0.9,
            "confidence_valid": True,
            "stage_a_safe_retry_used": True,
            "reasoning": "Spiders do not eat peaches.",
            "validated_output": {"answer_type": "yes/no", "key_constraints": "answer yes or no"},
        },
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "solver_mode": "solver_skeptic",
            "agent_id": 3,
            "normalized_answer": "yes",
            "prediction": "yes",
            "score": 1.0,
            "confidence_value": 0.86,
            "confidence_valid": True,
            "reasoning": "A black widow woman could refer to a human.",
            "validated_output": {},
        },
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "solver_mode": "solver_counterfactual",
            "adaptive_policy_name": "adaptive_counterfactual_v1",
            "agent_id": 4,
            "normalized_answer": "yes",
            "prediction": "yes",
            "score": 1.0,
            "claim_span": "yes",
            "key_evidence": "Black widow woman could refer to a human.",
            "answer_type": "yes/no",
            "key_constraints": "answer yes or no",
            "validated_output": {"answer_type": "yes/no", "key_constraints": "answer yes or no"},
        },
    ]
    prediction_rows = [
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "method_name": "hetero_vote_3",
            "method_kind": "aggregate",
            "prediction": "no",
            "normalized_answer": "no",
            "score": 0.0,
        },
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "method_name": "adaptive_counterfactual_v1",
            "method_kind": "aggregate",
            "prediction": "no",
            "normalized_answer": "no",
            "score": 0.0,
        },
    ]
    router_rows = [
        {
            "run_id": "run1",
            "dataset": "strategyqa",
            "split": "count20_seed42",
            "sample_id": "s1",
            "policy_name": "adaptive_counterfactual_v1",
            "triggered": True,
            "selected_addon_solver": "solver_verify",
        }
    ]

    refreshed_predictions, refreshed_router_rows = refresh_prediction_rows_for_run(
        stage_a_rows,
        prediction_rows,
        router_rows,
        sample_lookup={("strategyqa", "s1"): sample},
        protocol=protocol,
        model_name="demo-model",
        prompt_version=STAGE_A_V4_PROMPT_VERSION,
    )

    adaptive_row = next(row for row in refreshed_predictions if row["method_name"] == "adaptive_counterfactual_v1")
    assert adaptive_row["prediction"] == "yes"
    assert adaptive_row["score"] == 1.0
    assert adaptive_row["corrected_by_method"] is True
    adaptive_router_row = next(row for row in refreshed_router_rows if row["policy_name"] == "adaptive_counterfactual_v1")
    assert adaptive_router_row["selected_addon_solver"] == "solver_counterfactual"
    assert adaptive_router_row["executed_addon_solvers"] == ["solver_counterfactual"]
    assert adaptive_router_row["final_answer"] == "yes"


def test_resolve_global_sync_audit_outcome_keeps_safe_stage_a_majority() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": False, "majority_invalid": False},
    )

    assert answer == "A"
    assert resolver == "cot_mad_global_sync_keep_stage_a_majority"
    assert summary["accepted_override"] is False


def test_resolve_global_sync_audit_outcome_accepts_three_of_five_challenger_with_two_of_five_stage_a_support() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "B",
            "selected_candidate": "challenger_family",
            "majority_error": "majority ignores option text evidence",
        },
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "B",
            "selected_candidate": "challenger_family",
            "majority_error": "majority conflicts with exact option wording",
        },
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "B",
            "selected_candidate": "challenger_family",
            "majority_error": "majority evidence is unsupported",
        },
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": False, "majority_invalid": False},
    )

    assert answer == "B"
    assert resolver == "cot_mad_global_sync_accept_supported_challenger"
    assert summary["accepted_override"] is True


def test_resolve_global_sync_audit_outcome_rejects_challenger_without_majority_error_certificate() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": False, "majority_invalid": False},
    )

    assert answer == "A"
    assert resolver == "cot_mad_global_sync_keep_stage_a_majority"
    assert summary["accepted_override"] is False


def test_resolve_global_sync_audit_outcome_rejects_singleton_minority_without_majority_risk() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "majority missed direct contradiction"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "majority missed direct contradiction"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "B", "selected_candidate": "challenger_family", "majority_error": "majority missed direct contradiction"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": False, "majority_invalid": False},
    )

    assert answer == "A"
    assert resolver == "cot_mad_global_sync_keep_stage_a_majority"
    assert summary["accepted_override"] is False


def test_resolve_global_sync_audit_outcome_accepts_four_of_five_singleton_when_majority_is_risky() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "perfect storm"},
        {"solver_mode": "solver_cot", "normalized_answer": "perfect storm"},
        {"solver_mode": "solver_cot", "normalized_answer": "perfect storm"},
        {"solver_mode": "solver_cot", "normalized_answer": "perfect storm"},
        {"solver_mode": "solver_cot", "normalized_answer": "1991 perfect storm"},
    ]
    audit_rows = [
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "1991 perfect storm",
            "selected_candidate": "challenger_family",
            "majority_error": "majority omits required year in answer slot",
        },
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "1991 perfect storm",
            "selected_candidate": "challenger_family",
            "majority_error": "majority omits required year in answer slot",
        },
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "1991 perfect storm",
            "selected_candidate": "challenger_family",
            "majority_error": "majority answer is underspecified",
        },
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "1991 perfect storm",
            "selected_candidate": "challenger_family",
            "majority_error": "majority answer is underspecified",
        },
        {"solver_mode": "audit_solver_cot", "normalized_answer": "perfect storm", "selected_candidate": "stage_a_majority", "majority_error": "none"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="hotpotqa",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="perfect storm",
        gate_decision={"majority_risky": True, "majority_invalid": False},
    )

    assert answer == "1991 perfect storm"
    assert resolver == "cot_mad_global_sync_accept_risky_majority_override"
    assert summary["accepted_override"] is True


def test_resolve_global_sync_audit_outcome_rejects_under_supported_novel_answer() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {"solver_mode": "audit_solver_cot", "normalized_answer": "C", "selected_candidate": "novel_answer", "majority_error": "majority option conflicts with evidence"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "C", "selected_candidate": "novel_answer", "majority_error": "majority option conflicts with evidence"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "C", "selected_candidate": "novel_answer", "majority_error": "majority option conflicts with evidence"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
        {"solver_mode": "audit_solver_cot", "normalized_answer": "A", "selected_candidate": "stage_a_majority", "majority_error": "none"},
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": False, "majority_invalid": False},
    )

    assert answer == "A"
    assert resolver == "cot_mad_global_sync_keep_stage_a_majority"
    assert summary["rejected_novel_answer"] is True


def test_resolve_global_sync_audit_outcome_falls_back_on_invalid_task_format() -> None:
    stage_a_rows = [
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "A"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
        {"solver_mode": "solver_cot", "normalized_answer": "B"},
    ]
    audit_rows = [
        {
            "solver_mode": "audit_solver_cot",
            "normalized_answer": "Option B because it fits best",
            "selected_candidate": "challenger_family",
            "majority_error": "majority option letter conflicts with evidence",
            "validated_output": {"format_warning": "multiple_choice_answer_not_single_letter"},
        }
        for _ in range(5)
    ]

    answer, _support, resolver, summary = _resolve_global_sync_audit_outcome(
        benchmark_slug="mmlu_pro",
        stage_a_rows=stage_a_rows,
        audit_rows=audit_rows,
        stage_a_answer="A",
        gate_decision={"majority_risky": True, "majority_invalid": True},
    )

    assert answer == "A"
    assert resolver == "cot_mad_global_sync_invalid_format_fallback"
    assert summary["invalid_format_fallback"] is True


def test_run_sample_global_sync_uses_five_cot_stage_a_calls_with_sc5_aligned_seeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    experiment = load_experiment_config(
        "configs/families/adaptive_sparse_mad/experiments/same_context_main_v9.toml"
    )
    controls = load_control_catalog(experiment.control_catalog)
    sample = DatasetSample(
        dataset="hotpotqa",
        sample_id="demo",
        question="Which language is spoken there?",
        reference_answer="French",
        prompt_context="The town is in Quebec, where French is spoken.",
        metadata={},
    )
    stage_a_calls: list[tuple[str, int, str]] = []

    def fake_execute_turn(**kwargs):
        if kwargs["stage_name"] != "stage_a":
            raise AssertionError(f"unexpected stage_name={kwargs['stage_name']}")
        solver_mode = str((kwargs.get("extra_fields") or {}).get("solver_mode") or "")
        stage_a_calls.append((kwargs["stage_name"], kwargs["seed"], solver_mode))
        return {
            "run_id": "run1",
            "dataset": "hotpotqa",
            "split": "count20_seed42",
            "sample_id": "demo",
            "stage_name": kwargs["stage_name"],
            "method_name": kwargs["method_name"],
            "round_index": kwargs["round_index"],
            "agent_id": kwargs["agent_id"],
            "role": kwargs["role"],
            "solver_mode": solver_mode,
            "prediction": "french",
            "normalized_answer": "french",
            "score": 1.0,
            "reasoning": "Quebec implies French.",
            "confidence_value": 0.88,
            "confidence_valid": True,
            "key_evidence": "French is spoken in Quebec.",
            "answer_type": "language",
            "key_constraints": "short exact span",
            "output_status": "ok",
            "prompt_tokens": 10.0,
            "completion_tokens": 5.0,
            "total_tokens": 15.0,
            "latency_ms": 1.0,
            "cache_hit": True,
            "request_error": None,
            "assistant_text": "demo",
            "provider_reasoning_text": "",
            "validated_output": {"final_answer": "french"},
            "request_started_at": "2026-06-21T00:00:00+00:00",
            "stage_a_safe_retry_used": False,
        }

    def fake_run_unified_control_sample(**kwargs):
        return (
            [],
            {
                "dataset": kwargs["benchmark_slug"],
                "sample_id": kwargs["sample"].sample_id,
                "method_name": kwargs["control_name"],
                "method_kind": "control",
                "score": 1.0,
            },
        )

    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample._execute_turn",
        fake_execute_turn,
    )
    monkeypatch.setattr(
        "research_experiments.families.adaptive_sparse_mad.run.sample.run_unified_control_sample",
        fake_run_unified_control_sample,
    )

    result = _run_sample(
        sample,
        run_id="run1",
        benchmark_slug="hotpotqa",
        split_name="count20_seed42",
        protocol=AdaptiveSparseMadProtocolConfig(
            agent_count=5,
            top_p=1.0,
            stage_a_temperature=0.7,
            consensus_confidence_threshold=0.65,
            majority_confidence_threshold=0.6,
            majority_margin_threshold=0.25,
            debate_rounds=1,
            debate_temperature=0.7,
        ),
        controls=controls,
        experiment=experiment,
        backbone=SimpleNamespace(name="demo-model"),
        provider=SimpleNamespace(),
        cache=SimpleNamespace(),
        throttle=SimpleNamespace(),
    )

    assert len(stage_a_calls) == 5
    assert [seed for _stage_name, seed, _solver_mode in stage_a_calls] == [42, 43, 44, 45, 46]
    assert {solver_mode for _stage_name, _seed, solver_mode in stage_a_calls} == {"solver_cot"}
    assert len(result.stage_a_turns) == 5


def test_render_report_includes_control_and_runtime_contract_fields(tmp_path: Path) -> None:
    write_registered_family_manifest(
        tmp_path / "manifest.json",
        family_name="adaptive_sparse_mad",
        payload={
            "created_at": "2026-06-17T00:00:00+00:00",
            "experiment_name": "same_context_main_v5",
            "phase_name": "count20",
            "resolved_model": {"name": "xiaomimimo/mimo-v2.5"},
            "prompt_version": "adaptive_sparse_mad_free_text_debate_v1",
            "stage_a_prompt_version": "adaptive_sparse_mad_free_text_debate_v1",
            "adaptive_prompt_version": "adaptive_sparse_mad_free_text_debate_v1",
            "control_prompt_version": "single_agent_free_text_v1",
            "control_output_protocol": "free_text_answer_v1",
            "stage_a_response_format_mode": "free_text",
            "adaptive_response_format_mode": "free_text",
            "legacy_json_mode": False,
        },
    )
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {
                    "dataset": "overall",
                    "method_name": "cot_1",
                    "display_name": "cot_1",
                    "accuracy_mean": 0.4,
                    "communication_tokens_mean": 0.0,
                    "total_tokens_mean": 100.0,
                    "calls_per_question_mean": 1.0,
                    "acc_per_1k_tokens": 4.0,
                },
                {
                    "dataset": "overall",
                    "method_name": "sc_3",
                    "display_name": "sc_3",
                    "accuracy_mean": 0.5,
                    "communication_tokens_mean": 0.0,
                    "total_tokens_mean": 180.0,
                    "calls_per_question_mean": 3.0,
                    "acc_per_1k_tokens": 2.777778,
                },
                {
                    "dataset": "overall",
                    "method_name": "hetero_vote_3",
                    "display_name": "hetero_vote_3",
                    "accuracy_mean": 0.6,
                    "communication_tokens_mean": 0.0,
                    "total_tokens_mean": 220.0,
                    "calls_per_question_mean": 3.0,
                    "acc_per_1k_tokens": 2.727273,
                },
            ]
        },
    )
    write_json(tmp_path / "diagnostics" / "router_eval.json", {"summary_rows": []})
    write_json(
        tmp_path / "diagnostics" / "policy_diagnostics.json",
        {
            "policy_rows": [],
            "pairwise_rows": [],
            "promotion_gate": {"candidate_rows": []},
            "mainline_gate": {"candidate_rows": []},
            "recommended_next_default_policy": {"selected_policy": "hetero_vote_3"},
        },
    )
    write_json(tmp_path / "diagnostics" / "stage_a_resolver_breakdown.json", {"summary_rows": [], "example_rows": []})
    write_json(
        tmp_path / "diagnostics" / "stage_a_error_buckets.json",
        {"summary": {}, "dataset_rows": [], "example_rows": []},
    )
    write_json(tmp_path / "diagnostics" / "stage_a_solver_contributions.json", {"summary_rows": []})

    payload = render_report(tmp_path, publish_dir=tmp_path / "published")
    local_report = Path(payload["local_report"]).read_text(encoding="utf-8")

    assert "Control Prompt" in local_report
    assert "Control Output Protocol" in local_report
    assert "Stage A Response Format" in local_report
    assert "Legacy JSON Mode" in local_report
    assert "sc_3" in local_report


def test_summarize_run_reads_metrics(tmp_path: Path) -> None:
    write_registered_family_manifest(tmp_path / "manifest.json", family_name="adaptive_sparse_mad")
    write_json(
        tmp_path / "views" / "metrics.json",
        {
            "summary": [
                {"dataset": "gsm8k", "method_name": "hetero_vote_3", "accuracy_mean": 0.8},
                {"dataset": "overall", "method_name": "hetero_vote_3", "accuracy_mean": 0.8},
            ]
        },
    )

    payload = summarize_run(tmp_path)

    assert payload["row_count"] == 2
    assert payload["datasets"] == ["gsm8k", "overall"]
