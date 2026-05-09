"""覆盖 split-context communication-necessary 纯逻辑的测试。"""

from __future__ import annotations

from comm_necessary.config import CommNecessaryProtocolConfig
from comm_necessary.dataset_views import build_hotpot_views
from comm_necessary.logic import (
    aggregate_supporting_facts,
    build_packet,
    majority_vote_with_counts,
    score_hotpot_prediction,
)
from experiment_core.foundation.datasets import DatasetSample


def _sample() -> DatasetSample:
    return DatasetSample(
        dataset="hotpotqa",
        sample_id="h1",
        question="Who connects A and B?",
        reference_answer="Alpha",
        prompt_context="[A] (0) A sent.\n[B] (0) B sent.",
        metadata={
            "type": "bridge",
            "level": "hard",
            "supporting_facts": {"title": ["A", "B"], "sent_id": [0, 0]},
            "raw_context": {
                "title": ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"],
                "sentences": [
                    ["A sent."],
                    ["B sent."],
                    ["C sent."],
                    ["D sent."],
                    ["E sent."],
                    ["F sent."],
                    ["G sent."],
                    ["H sent."],
                    ["I sent."],
                    ["J sent."],
                ],
            },
        },
    )


def test_hotpot_split_views_cover_supporting_titles_without_leaking_full_context() -> None:
    views = build_hotpot_views(_sample())
    split_views = [view for view in views if view.agent_id in {1, 2, 3}]
    covered = set()
    required = set()
    for view in split_views:
        assert view.includes_full_context is False
        assert view.view_context_hash != view.full_context_hash
        assert "(0)" in view.context_text
        covered.update(view.coverage_titles)
        required.update(view.required_titles)
    assert required == {"A", "B"}
    assert required.issubset(covered)


def test_packet_builder_respects_token_cap() -> None:
    row = {
        "agent_id": 1,
        "normalized_answer": "alpha",
        "confidence_raw": 0.8,
        "evidence_summary": "evidence " * 100,
        "reasoning_trace": "reasoning " * 100,
        "supporting_facts": [["A", 0], ["B", 0]],
    }
    packet = build_packet(row, packet_mode="full_packet", token_cap=80)
    assert packet["approx_packet_tokens"] <= 80
    assert "final_answer" in packet["packet_fields"]


def test_majority_vote_tie_break_keeps_first_answer() -> None:
    winner, counts, consensus = majority_vote_with_counts(["b", "a", "a", "b"])
    assert winner == "b"
    assert counts == {"b": 2, "a": 2}
    assert consensus is False


def test_support_fact_aggregation_prefers_winner_answer() -> None:
    rows = [
        {"normalized_answer": "a", "supporting_facts": [["A", 0]]},
        {"normalized_answer": "b", "supporting_facts": [["B", 0]]},
        {"normalized_answer": "a", "supporting_facts": [["A", 0], ["C", 1]]},
    ]
    assert aggregate_supporting_facts(rows, "a")[:2] == [("A", 0), ("C", 1)]


def test_hotpot_scorer_joint_metrics() -> None:
    scores = score_hotpot_prediction(
        predicted_answer="The Alpha",
        gold_answer="alpha",
        predicted_supporting_facts=[("A", 0), ("B", 0)],
        gold_supporting_facts=[("A", 0), ("B", 0)],
    )
    assert scores.answer_em == 1.0
    assert scores.supporting_em == 1.0
    assert scores.joint_em == 1.0
    assert scores.joint_f1 == 1.0

