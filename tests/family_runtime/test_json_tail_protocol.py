from __future__ import annotations

import pytest

from research_experiments.family_runtime.json_tail_protocol import parse_json_tail_answer_output


def test_parse_json_tail_answer_output_parses_reasoning_and_tail_json() -> None:
    payload = parse_json_tail_answer_output(
        'Check the decisive calculation.\n\n[FINAL]\n{"answer":"42","confidence":0.8}\n[/FINAL]',
        dataset="gsm8k",
    )

    assert payload["final_answer"] == "42"
    assert payload["answer"] == "42"
    assert payload["confidence"] == 0.8
    assert payload["reasoning"] == "Check the decisive calculation."


def test_parse_json_tail_answer_output_rejects_missing_tail() -> None:
    with pytest.raises(ValueError, match="Missing final"):
        parse_json_tail_answer_output("answer is 42", dataset="gsm8k")


def test_parse_json_tail_answer_output_flags_task_format_warning() -> None:
    payload = parse_json_tail_answer_output(
        '[FINAL]{"answer":"Option C","confidence":0.7}[/FINAL]',
        dataset="gpqa_diamond",
    )

    assert payload["format_warning"] == "multiple_choice_answer_not_single_letter"
