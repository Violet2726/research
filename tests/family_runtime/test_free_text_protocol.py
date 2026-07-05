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


def test_parse_free_text_answer_output_accepts_reasonning_alias() -> None:
    payload = parse_free_text_answer_output(
        "REASONNING: Peer evidence ruled out A.\nFINAL_ANSWER: B",
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


def test_parse_free_text_answer_output_recovers_embedded_final_answer_marker() -> None:
    payload = parse_free_text_answer_output(
        "The decisive option is C after comparing the choices. FINAL_ANSWER: C",
        dataset="gpqa_diamond",
    )

    assert payload["final_answer"] == "C"
    assert payload["protocol_recovery"] == "embedded_final_answer"


def test_parse_free_text_answer_output_recovers_final_marker_after_think_tag() -> None:
    payload = parse_free_text_answer_output(
        "</think>FINAL_ANSWER: J",
        dataset="mmlu_pro",
    )

    assert payload["final_answer"] == "J"
    assert payload["protocol_recovery"] == "embedded_final_answer"
    assert payload["reasoning"]


def test_parse_free_text_answer_output_recovers_mc_tail_answer_phrase() -> None:
    payload = parse_free_text_answer_output(
        "The elimination leaves option B, so the final answer is B.",
        dataset="mmlu_pro",
    )

    assert payload["final_answer"] == "B"
    assert payload["protocol_recovery"] == "mc_tail_answer_phrase"


def test_parse_free_text_answer_output_keeps_hotpot_empty_final_answer_failed() -> None:
    with pytest.raises(ValueError, match="FINAL_ANSWER"):
        parse_free_text_answer_output(
            "REASONING: The context does not provide a span.\nFINAL_ANSWER:",
            dataset="hotpotqa",
        )


def test_parse_free_text_answer_output_does_not_guess_hotpot_natural_language() -> None:
    with pytest.raises(ValueError):
        parse_free_text_answer_output(
            "The answer is probably Captain John Underhill.",
            dataset="hotpotqa",
        )


def test_task_format_ok_enforces_multiple_choice_letter() -> None:
    assert task_format_ok("gpqa_diamond", "C") is True
    assert task_format_ok("gpqa_diamond", "Option C") is False
