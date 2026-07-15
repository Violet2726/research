from research_experiments.families.risk_controlled_trace_mad.algorithms import (
    build_candidate_board,
    build_support_blind_board,
    decide_override,
    homogeneous_stage_decision,
    reviewer_selected_key,
    stage_decision,
)


def _row(answer: str, agent: int, lineage: str = "qwen") -> dict:
    return {
        "normalized_answer": answer,
        "agent_id": agent,
        "model_lineage": lineage,
        "assistant_text": f"reason {answer}",
    }


def test_five_zero_does_not_trigger() -> None:
    rows = [_row("A", index) for index in range(1, 6)]
    decision = stage_decision(rows, qwen_rows=rows[:3])
    assert decision.anchor_answer == "A"
    assert decision.triggered is False


def test_two_two_one_tie_uses_qwen_three_then_fixed_order() -> None:
    rows = [_row("A", 1), _row("B", 2), _row("A", 3), _row("B", 1, "mimo"), _row("C", 2, "mimo")]
    decision = stage_decision(rows, qwen_rows=rows[:3])
    assert decision.anchor_answer == "A"
    assert decision.resolver == "qwen_three_tie_fallback"


def test_anonymous_board_permutation_preserves_candidate_set() -> None:
    rows = [_row("A", 1), _row("A", 2), _row("B", 3), _row("C", 4)]
    first, first_map, _ = build_candidate_board(
        rows, seed=42, sample_id="x", purpose="one", trace_max_chars=100, board_max_chars=2000
    )
    second, second_map, _ = build_candidate_board(
        rows, seed=42, sample_id="x", purpose="two", trace_max_chars=100, board_max_chars=2000
    )
    assert set(first_map.values()) == set(second_map.values()) == {"A", "B", "C"}
    assert "vote" not in first.casefold()
    assert first != second


def test_override_requires_agreement_two_challenger_passes_and_anchor_falsification() -> None:
    evidence = [
        {"target_answer": "B", "status": "pass", "claim_kind": "support"},
        {"target_answer": "A", "status": "pass", "claim_kind": "falsify"},
    ]
    audits = [
        {"preferred_answer": "B", "evidence_results": evidence},
        {
            "preferred_answer": "B",
            "evidence_results": [{"target_answer": "B", "status": "pass", "claim_kind": "support"}],
        },
    ]
    accepted, _ = decide_override(
        anchor="A", challenger="B", audits=audits, challenger_required_passes=2, anchor_required_falsifications=1
    )
    assert accepted is True
    audits[1]["preferred_answer"] = "A"
    assert (
        decide_override(
            anchor="A", challenger="B", audits=audits, challenger_required_passes=2, anchor_required_falsifications=1
        )[0]
        is False
    )


def test_homogeneous_answer_classes_merge_formatting_and_hash_ties() -> None:
    rows = [_row("(b)", 1), _row("b", 2), _row("[b]", 3), _row("c", 4), _row("c", 5)]
    decision = homogeneous_stage_decision(rows, dataset="bbeh", seed=42, sample_id="s")
    assert decision.vote_counts == {"b": 3, "c": 2}
    assert decision.anchor_key == "b"


def test_blind_board_permutation_and_unanimous_existing_non_anchor_gate() -> None:
    rows = [_row("A", 1), _row("A", 2), _row("B", 3), _row("C", 4), _row("C", 5)]
    first = build_support_blind_board(
        rows, dataset="bbeh", seed=42, sample_id="s", reviewer_index=1,
        trace_max_chars=100, board_max_chars=2000
    )
    second = build_support_blind_board(
        rows, dataset="bbeh", seed=42, sample_id="s", reviewer_index=2,
        trace_max_chars=100, board_max_chars=2000
    )
    assert set(first[1].values()) == set(second[1].values()) == {"a", "b", "c"}
    reviews = [
        {"output_status": "ok", "validated_output": {"picked_answer_class_key": "b"}}
        for _ in range(3)
    ]
    assert reviewer_selected_key(reviews, anchor_key="a", candidate_keys={"a", "b", "c"}, required=3)[0] == "b"
    reviews[2] = {"output_status": "protocol_fail", "validated_output": {}}
    assert reviewer_selected_key(reviews, anchor_key="a", candidate_keys={"a", "b", "c"}, required=3)[0] == "a"
