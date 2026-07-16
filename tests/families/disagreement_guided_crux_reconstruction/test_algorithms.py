from research_experiments.families.disagreement_guided_crux_reconstruction.algorithms import (
    CandidateClass,
    StageDecision,
    build_panel_labels,
    build_stage_decision,
    decide_override,
    exact_span_match,
    panel_successes,
    validate_crux_span,
)


def _stage() -> StageDecision:
    candidates = (CandidateClass("A", "A", 4), CandidateClass("B", "B", 1))
    return StageDecision("A", "A", candidates, {"A": 4, "B": 1}, 5)


def test_span_rejects_options_and_accepts_contiguous_question_body() -> None:
    question = "A decisive premise is present here.\nOptions:\n(A) one\n(B) two"
    span = validate_crux_span(question, start_char=2, end_char=29)
    assert span is not None and "DGCR_HIDDEN_CRUX" in span.masked_question
    assert validate_crux_span(question, start_char=question.index("Options"), end_char=len(question)) is None


def test_double_panel_override_requires_unique_challenger_and_anchor_failure() -> None:
    stage = _stage()
    answer, override, resolver = decide_override(stage, [{"A": False, "B": True}, {"A": False, "B": True}])
    assert (answer, override, resolver) == ("B", True, "unique_double_panel_override")
    assert decide_override(stage, [{"A": True, "B": True}, {"A": False, "B": True}])[1] is False
    multi = StageDecision(
        "A", "A", (CandidateClass("A", "A", 3), CandidateClass("B", "B", 1), CandidateClass("C", "C", 1)),
        {"A": 3, "B": 1, "C": 1}, 5,
    )
    assert decide_override(multi, [{"A": False, "B": True, "C": True}] * 2)[1] is False


def test_panel_requires_exact_label_coverage_and_exact_text() -> None:
    span = validate_crux_span("abcdefghij crucial text", start_char=0, end_char=10)
    assert span is not None
    mapping = {"A": "A", "B": "B"}
    assert panel_successes({"A": span.hidden_text, "B": "different"}, label_to_key=mapping, span=span) == {"A": True, "B": False}
    assert panel_successes({"A": span.hidden_text}, label_to_key=mapping, span=span) is None
    assert panel_successes({"A": span.hidden_text, "B": "x", "C": "novel"}, label_to_key=mapping, span=span) is None


def test_invalid_stage_outputs_cannot_trigger_and_match_is_only_nfkc_crlf_normalized() -> None:
    stage = build_stage_decision(
        [
            {"answer_class_key": "", "normalized_answer": "D"},
            {"answer_class_key": "D", "normalized_answer": "D"},
        ],
        seed=42,
        sample_id="x",
    )
    assert stage.valid_count == 1 and not stage.triggered
    assert exact_span_match("Ａ\r\nＢ", "A\nB")
    assert not exact_span_match("A\rB", "A\nB")
    assert validate_crux_span("abcdefgh", start_char="not-an-int", end_char=8) is None


def test_panel_labels_are_independently_permuted_without_support_counts() -> None:
    candidates = _stage().candidates
    first = build_panel_labels(candidates, seed=42, sample_id="x", panel_index=1)
    second = build_panel_labels(candidates, seed=42, sample_id="x", panel_index=2)
    assert set(first.values()) == {"A", "B"}
    assert set(second.values()) == {"A", "B"}
