from __future__ import annotations

from research_experiments.families.madjudge.run.sample import _repair_madjudge_answer_output


def test_repair_madjudge_answer_output_prefers_reasoning_conclusion_for_gsm8k() -> None:
    payload = {
        "final_answer": "50.666666666666664",
        "reasoning": "The three measurements are 47, 52, and 57. Average = (47 + 52 + 57) / 3 = 156 / 3 = 52.",
    }

    repaired = _repair_madjudge_answer_output("gsm8k", payload)

    assert repaired["final_answer"] == "52"


def test_repair_madjudge_answer_output_keeps_answer_when_reasoning_mentions_it() -> None:
    payload = {
        "final_answer": "70",
        "reasoning": "Robots are 10, helmets are 20, footballs are 40, so the total is 70.",
    }

    repaired = _repair_madjudge_answer_output("gsm8k", payload)

    assert repaired["final_answer"] == "70"


def test_repair_madjudge_answer_output_skips_peer_comparison_reasoning() -> None:
    payload = {
        "final_answer": "70",
        "reasoning": "Other judges reported 50, but that arithmetic is wrong.",
    }

    repaired = _repair_madjudge_answer_output("gsm8k", payload)

    assert repaired["final_answer"] == "70"
