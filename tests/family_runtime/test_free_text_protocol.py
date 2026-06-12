from __future__ import annotations

import pytest

from research_experiments.family_runtime.free_text_protocol import (
    parse_free_text_answer_output,
    task_format_ok,
)


def test_parse_free_text_answer_output_parses_two_line_answer() -> None:
    payload = parse_free_text_answer_output(
        "REASONING: Add the two numbers.\nFINAL_ANSWER: 42",
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "42"
    assert payload["reasoning"] == "Add the two numbers."


def test_parse_free_text_answer_output_accepts_reason_alias() -> None:
    payload = parse_free_text_answer_output(
        "REASON: Peer evidence ruled out A.\nFINAL_ANSWER: B",
        dataset="gpqa_diamond",
    )
    assert payload["final_answer"] == "B"
    assert payload["reasoning"] == "Peer evidence ruled out A."


def test_parse_free_text_answer_output_requires_reasoning_before_final_answer() -> None:
    with pytest.raises(ValueError, match="REASONING must appear before FINAL_ANSWER"):
        parse_free_text_answer_output(
            "FINAL_ANSWER: B\nREASONING: Peer evidence ruled out A.",
            dataset="gpqa_diamond",
        )


def test_task_format_ok_enforces_multiple_choice_letter() -> None:
    assert task_format_ok("gpqa_diamond", "C") is True
    assert task_format_ok("gpqa_diamond", "Option C") is False
