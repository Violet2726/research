
"""覆盖结构化输出注册表、恢复逻辑与各类 schema 校验。"""

from __future__ import annotations

import json

import pytest

from research_experiments.core.structured_outputs import (
    SCHEMA_ANSWER_CORE,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
    SCHEMA_AUDIT_VERDICT,
    SCHEMA_BELIEF_UPDATE_DELTA,
    SCHEMA_CUE_BLACKBOX_PACKET,
    SCHEMA_DELIBERATION_PACKET,
    SCHEMA_SPLIT_CONTEXT_BELIEF,
    SCHEMA_SPLIT_CONTEXT_SOLVER,
    parse_proxy_signal_answer,
    validate_or_recover_structured_output,
    validate_structured_output,
)


def test_validate_core_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning": "short",
            }
        ),
        SCHEMA_ANSWER_CORE,
    )
    assert payload["final_answer"] == "yes"
    assert payload["reasoning"] == "short"

def test_validate_or_recover_core_output_from_truncated_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 42, "reasoning": "simple arithmetic"',
        SCHEMA_ANSWER_CORE,
    )
    assert payload["final_answer"] == "42"
    assert payload["reasoning"] == "simple arithmetic"

def test_validate_selective_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": 0.4,
                "reasoning": "short",
                "claim_span": "supporting span",
                "uncertainty_type": "evidence_selection",
                "key_evidence": "one clue",
                "uncertain_point": None,
            }
        ),
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
    )
    assert payload["confidence_raw"] == 0.4
    assert payload["claim_span"] == "supporting span"
    assert payload["uncertainty_type"] == "evidence_selection"
    assert payload["key_evidence"] == "one clue"
    assert payload["uncertain_point"] is None

def test_validate_selective_structured_output_allows_missing_confidence() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "42",
                "claim_span": "5 times 8 plus 2",
                "uncertainty_type": "calculation",
            }
        ),
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
    )
    assert payload["final_answer"] == "42"
    assert payload["confidence_raw"] is None

def test_validate_or_recover_selective_output_uses_reasoning_fallback() -> None:
    payload = validate_or_recover_structured_output(
        "[answer]",
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        dataset="gsm8k",
        provider_reasoning_text="Therefore, the final answer is 36.",
    )
    assert payload["final_answer"] == "36"
    assert payload["uncertainty_type"] == "calculation"

def test_validate_selective_structured_output_rejects_invalid_uncertainty_type() -> None:
    with pytest.raises(ValueError):
        validate_structured_output(
            json.dumps(
                {
                    "final_answer": "yes",
                    "confidence_raw": 0.4,
                    "claim_span": "supporting span",
                    "uncertainty_type": "bad_label",
                }
            ),
            SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE,
        )

def test_parse_selective_output_accepts_tagged_lines() -> None:
    payload = parse_proxy_signal_answer(
        "\n".join(
            [
                "FINAL_ANSWER: 36",
                "CLAIM_SPAN: 54 - 18 = 36",
                "UNCERTAINTY_TYPE: calculation",
                "CONFIDENCE: 0.82",
                "REASON: students take the remaining seats",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "36"
    assert payload["claim_span"] == "54 - 18 = 36"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] == "0.82"

def test_parse_selective_output_recovers_from_free_form_math_text() -> None:
    payload = parse_proxy_signal_answer(
        "To solve it, total seats are 72, admins use 18, parents use 18, so students are 36. The final answer is 36.",
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "36"
    assert payload["claim_span"] == "36"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] is None

def test_parse_selective_output_prefers_last_real_label_after_thought_block() -> None:
    payload = parse_proxy_signal_answer(
        "\n".join(
            [
                "{thought}",
                "The model is planning its answer.",
                "CONFIDENCE: <0-1 number or NA>",
                "{/thought}",
                "FINAL_ANSWER: 52",
                "CLAIM_SPAN: 47, 52, 57 -> average 52",
                "UNCERTAINTY_TYPE: none",
                "CONFIDENCE: 1",
                "REASON: Simple arithmetic.",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "52"
    assert payload["confidence_raw"] == "1"
    assert payload["claim_span"] == "47, 52, 57 -> average 52"

def test_parse_selective_output_recovers_from_truncated_thought_math() -> None:
    payload = parse_proxy_signal_answer(
        "\n".join(
            [
                "{thought}",
                "Total ounces = 2 glasses * 8 ounces/glass = 16 ounces.",
                "Total calories = 16 ounces * 3 calories/ounce = 48 calories.",
                "The output must have five lines:",
                "FINAL_ANSWER: <answer>",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "48"
    assert payload["confidence_raw"] is None

def test_parse_selective_output_accepts_json_with_tagged_keys() -> None:
    payload = parse_proxy_signal_answer(
        json.dumps(
            {
                "FINAL_ANSWER": "5",
                "CLAIM_SPAN": "2 children * 5 changes/day / 2",
                "UNCERTAINTY_TYPE": "calculation",
                "CONFIDENCE": 1.0,
                "REASON": "Simple arithmetic with clear counts.",
            }
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "5"
    assert payload["claim_span"] == "2 children * 5 changes/day / 2"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] == 1.0
    assert payload["reasoning"] == "Simple arithmetic with clear counts."

def test_parse_selective_output_accepts_json_with_na_confidence() -> None:
    payload = parse_proxy_signal_answer(
        json.dumps(
            {
                "FINAL_ANSWER": "14",
                "CLAIM_SPAN": "28 remaining lollipops, 2 per bag -> 14 bags",
                "UNCERTAINTY_TYPE": "calculation",
                "CONFIDENCE": "NA",
                "REASON": "Simple subtraction and division.",
            }
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "14"
    assert payload["confidence_raw"] is None

def test_parse_selective_output_recovers_from_truncated_json() -> None:
    payload = parse_proxy_signal_answer(
        "\n".join(
            [
                "{",
                '  "FINAL_ANSWER": "14",',
                '  "CLAIM_SPAN": "28 remaining lollipops, 2 per bag -> 14 bags",',
                '  "UNCERTAINTY_TYPE": "calculation",',
                '  "CONFIDENCE": "NA",',
                '  "REASON": "Simple subtraction',
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "14"
    assert payload["claim_span"] == "28 remaining lollipops, 2 per bag -> 14 bags"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] is None

def test_validate_comm_necessary_solver_output_allows_empty_final_answer_as_abstention() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "",
                "reasoning_trace": "Visible evidence does not connect the entities.",
                "evidence_summary": "No grounded answer in this shard.",
                "supporting_facts": [],
                "confidence_raw": 0.0,
            }
        ),
        SCHEMA_SPLIT_CONTEXT_SOLVER,
    )
    assert payload["final_answer"] is None
    assert payload["evidence_summary"] == "No grounded answer in this shard."

def test_validate_comm_necessary_belief_output_normalizes_empty_answer_to_no_change() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": True,
                "final_answer": "",
                "reasoning_trace": "Peer packets still do not ground an answer.",
                "evidence_summary": "Insufficient cross-shard evidence.",
                "supporting_facts": [],
                "confidence_raw": 0.0,
            }
        ),
        SCHEMA_SPLIT_CONTEXT_BELIEF,
    )
    assert payload["changed_answer"] is False
    assert payload["final_answer"] is None

def test_validate_budget_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning_trace": "short trace",
                "claim_span": "one claim",
                "key_evidence": "one evidence",
                "keyword_clues": ["alpha", "beta"],
                "confidence_raw": 0.8,
                "uncertain_point": None,
            }
        ),
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    )
    assert payload["keyword_clues"] == ["alpha", "beta"]
    assert payload["confidence_raw"] == 0.8

def test_validate_or_recover_budget_solver_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 50, "confidence_raw": 0.8, "keyword_clues": ["50"], "reasoning_trace": "done"',
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    )
    assert payload["final_answer"] == "50"
    assert payload["confidence_raw"] == 0.8
    assert payload["keyword_clues"] == ["50"]

def test_validate_or_recover_budget_solver_defaults_missing_confidence() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer":"82","reasoning_trace":"done","claim_span":"capacity math","key_evidence":"5000-3755"}',
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET,
    )
    assert payload["final_answer"] == "82"
    assert payload["confidence_raw"] == 0.5
    assert payload["keyword_clues"] == ["capacity math"]

def test_validate_budget_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": False,
                "new_answer": "no",
                "confidence_delta": 0.1,
                "reason_for_change": "peer confirms it",
                "remaining_disagreement": None,
            }
        ),
        SCHEMA_BELIEF_UPDATE_DELTA,
    )
    assert payload["changed_answer"] is False
    assert payload["new_answer"] == "no"

def test_validate_comm_necessary_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "Scott Adkins",
                "reasoning_trace": "Actor match from visible evidence.",
                "evidence_summary": "Universal Soldier and Holby City overlap.",
                "supporting_facts": [{"title": "Scott Adkins", "sent_id": 0}],
                "confidence_raw": 0.6,
            }
        ),
        SCHEMA_SPLIT_CONTEXT_SOLVER,
    )
    assert payload["final_answer"] == "Scott Adkins"
    assert payload["supporting_facts"] == [{"title": "Scott Adkins", "sent_id": 0}]

def test_validate_or_recover_comm_necessary_belief_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"changed_answer": false, "final_answer": "Saoirse Ronan", "reasoning_trace": "peer confirms", '
        '"supporting_facts":[{"title":"Billy Howle","sent_id":2}], "confidence_raw": 0.7',
        SCHEMA_SPLIT_CONTEXT_BELIEF,
    )
    assert payload["changed_answer"] is False
    assert payload["final_answer"] == "Saoirse Ronan"
    assert payload["supporting_facts"] == [{"title": "Billy Howle", "sent_id": 2}]

def test_validate_or_recover_core_soft_rejection_fail_opens() -> None:
    payload = validate_or_recover_structured_output(
        "The request was rejected because it was considered high risk",
        SCHEMA_ANSWER_CORE,
    )
    assert payload["final_answer"] == "unknown"
    assert payload["reasoning"] == "provider_soft_rejection"

def test_validate_or_recover_comm_necessary_soft_rejection_fail_opens() -> None:
    payload = validate_or_recover_structured_output(
        "The request was rejected because it was considered high risk",
        SCHEMA_SPLIT_CONTEXT_SOLVER,
    )
    assert payload["final_answer"] == "unknown"
    assert payload["supporting_facts"] == []

def test_validate_sparc_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "42",
                "reasoning_trace": "short trace",
                "claim_span": "2+40",
                "confidence_raw": 0.7,
                "uncertain_point": "unit conversion",
                "key_evidence": "5 groups of 8 and 2 extra",
            }
        ),
        SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION,
    )
    assert payload["final_answer"] == "42"
    assert payload["confidence_raw"] == 0.7

def test_validate_or_recover_cue_solver_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 50, "confidence": 0.6, "top_claims": ["50"], "evidence_items": ["calc"], "reasoning_sketch": "done"',
        SCHEMA_CUE_BLACKBOX_PACKET,
    )
    assert payload["final_answer"] == "50"
    assert payload["confidence"] == 0.6
    assert payload["top_claims"] == ["50"]

def test_validate_sparc_message_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": "0.8",
                "claim_span": "the key factual claim",
            }
        ),
        SCHEMA_DELIBERATION_PACKET,
    )
    assert payload["claim_span"] == "the key factual claim"

def test_validate_sparc_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": True,
                "new_answer": "no",
                "confidence_delta": -0.1,
                "reason_for_change": "peer evidence is stronger",
                "remaining_disagreement": None,
            }
        ),
        SCHEMA_BELIEF_UPDATE_DELTA,
    )
    assert payload["changed_answer"] is True
    assert payload["new_answer"] == "no"

def test_validate_sparc_audit_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "decision": "resolve_for_a",
                "verified_answer": "48",
                "rationale": "candidate A matches the evidence",
            }
        ),
        SCHEMA_AUDIT_VERDICT,
    )
    assert payload["decision"] == "resolve_for_a"


@pytest.mark.parametrize(
    ("raw_text", "mode"),
    [
        ('```json\n{"final_answer":"yes","reasoning":"short"}\n```', SCHEMA_ANSWER_CORE),
        ("The answer is yes.", SCHEMA_ANSWER_CORE),
        ('{"final_answer":"yes","confidence_raw":"high","reasoning":"short","key_evidence":null,"uncertain_point":null}', SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE),
        ('{"final_answer":"yes","reasoning_trace":"short","claim_span":"claim","key_evidence":"evidence","keyword_clues":[],"confidence_raw":0.8,"uncertain_point":null}', SCHEMA_ANSWER_WITH_PROXY_SIGNALS_BUDGET),
        ('{"final_answer":"yes","reasoning":"short"}{"final_answer":"no","reasoning":"alt"}', SCHEMA_ANSWER_CORE),
        ('{"reasoning":"short"}', SCHEMA_ANSWER_CORE),
        ('{"final_answer":"yes","uncertainty_type":"bad_label"}', SCHEMA_ANSWER_WITH_PROXY_SIGNALS_SELECTIVE),
        ('{"final_answer":"yes","unexpected":"field"}', SCHEMA_ANSWER_CORE),
    ],
)
def test_validate_structured_output_rejects_malformed_payloads(raw_text: str, mode: str) -> None:
    with pytest.raises(ValueError):
        validate_structured_output(raw_text, mode)  # type: ignore[arg-type]

