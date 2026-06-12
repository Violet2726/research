from __future__ import annotations

import pytest

from research_experiments.family_runtime.free_text_protocol import (
    parse_free_text_answer_output,
    task_format_ok,
)


def test_parse_free_text_answer_output_parses_two_line_answer() -> None:
    payload = parse_free_text_answer_output(
        "FINAL_ANSWER: 42\nREASON: Add the two numbers.",
        dataset="gsm8k",
        require_decision=False,
    )
    assert payload["final_answer"] == "42"
    assert payload["reasoning"] == "Add the two numbers."


def test_parse_free_text_answer_output_parses_debate_update() -> None:
    payload = parse_free_text_answer_output(
        "DECISION: revise\nFINAL_ANSWER: B\nREASON: Peer evidence ruled out A.",
        dataset="gpqa_diamond",
        require_decision=True,
    )
    assert payload["decision"] == "revise"
    assert payload["changed_answer"] is True
    assert payload["final_answer"] == "B"


def test_parse_free_text_answer_output_requires_decision_for_debate() -> None:
    with pytest.raises(ValueError, match="Missing DECISION line"):
        parse_free_text_answer_output(
            "FINAL_ANSWER: B\nREASON: Peer evidence ruled out A.",
            dataset="gpqa_diamond",
            require_decision=True,
        )


def test_task_format_ok_enforces_multiple_choice_letter() -> None:
    assert task_format_ok("gpqa_diamond", "C") is True
    assert task_format_ok("gpqa_diamond", "Option C") is False
