from __future__ import annotations

import pytest

from research_experiments.family_runtime.json_tail_protocol import parse_json_object_tail_answer_output


def test_parse_json_object_tail_answer_output_parses_reasoning_and_tail_json() -> None:
    payload = parse_json_object_tail_answer_output(
        (
            "Check the decisive calculation.\n\n"
            '{"answer":"42","confidence":0.8,"risk_level":"low","key_evidence":"6*7=42"}'
        ),
        dataset="gsm8k",
    )

    assert payload["final_answer"] == "42"
    assert payload["answer"] == "42"
    assert payload["confidence"] == 0.8
    assert payload["risk_level"] == "low"
    assert payload["reasoning"] == "Check the decisive calculation."


def test_parse_json_object_tail_answer_output_rejects_final_tags() -> None:
    with pytest.raises(ValueError, match="Missing final JSON object|Final JSON object is invalid"):
        parse_json_object_tail_answer_output(
            '[FINAL]{"answer":"42","confidence":0.8,"risk_level":"low","key_evidence":"6*7=42"}[/FINAL]',
            dataset="gsm8k",
        )


def test_parse_json_object_tail_answer_output_requires_structured_risk() -> None:
    with pytest.raises(ValueError, match="risk_level"):
        parse_json_object_tail_answer_output(
            'Reasoning.\n{"answer":"42","confidence":0.8,"risk_level":"low risk","key_evidence":"6*7=42"}',
            dataset="gsm8k",
        )


def test_parse_json_object_tail_answer_output_flags_task_format_warning() -> None:
    payload = parse_json_object_tail_answer_output(
        'Reasoning.\n{"answer":"Option C","confidence":0.7,"risk_level":"medium","key_evidence":"option text"}',
        dataset="gpqa_diamond",
    )

    assert payload["format_warning"] == "multiple_choice_answer_not_single_letter"
