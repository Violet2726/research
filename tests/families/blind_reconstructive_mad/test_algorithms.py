from __future__ import annotations

from research_experiments.families.blind_reconstructive_mad.algorithms import (
    build_reviewer_board,
    build_stage_a_decision,
    decide_existing_candidate_quorum,
    reviewer_error_correlation,
)


def _stage(answers: list[str]):
    return build_stage_a_decision(
        [
            {"normalized_answer": answer, "assistant_text": f"reasoning {index}", "agent_id": index}
            for index, answer in enumerate(answers, start=1)
        ]
    )


def test_five_zero_does_not_trigger_or_review() -> None:
    stage = _stage(["a"] * 5)
    decision = decide_existing_candidate_quorum(stage, [])

    assert not stage.triggered
    assert decision.final_answer == "a"
    assert not decision.quorum_met


def test_four_one_requires_three_minority_reviews() -> None:
    stage = _stage(["a", "a", "a", "a", "b"])

    assert decide_existing_candidate_quorum(stage, ["b", "b", "a"]).final_answer == "a"
    promoted = decide_existing_candidate_quorum(stage, ["b", "b", "b"])
    assert promoted.final_answer == "b"
    assert promoted.quorum_required == 3
    assert promoted.override_accepted


def test_three_two_and_two_two_one_use_two_vote_quorum() -> None:
    three_two = _stage(["a", "a", "a", "b", "b"])
    two_two_one = _stage(["a", "a", "b", "b", "c"])

    assert decide_existing_candidate_quorum(three_two, ["b", "b", "a"]).final_answer == "b"
    assert decide_existing_candidate_quorum(two_two_one, ["b", "b", "a"]).final_answer == "b"


def test_tied_reviewer_votes_fall_back_and_new_answers_stay_shadow() -> None:
    stage = _stage(["a", "a", "a", "b", "b"])
    decision = decide_existing_candidate_quorum(stage, ["b", "c", "d"])

    assert decision.final_answer == "a"
    assert not decision.override_accepted
    assert decision.shadow_answers == ("c", "d")


def test_anonymous_board_hides_support_and_is_deterministic_per_reviewer() -> None:
    stage = _stage(["a", "a", "a", "b", "b"])
    board_one = build_reviewer_board(stage, global_seed=7, sample_id="x", method_name="brd_quorum_3", reviewer_id=1, show_support=False)
    board_two = build_reviewer_board(stage, global_seed=7, sample_id="x", method_name="brd_quorum_3", reviewer_id=1, show_support=False)
    visible = build_reviewer_board(stage, global_seed=7, sample_id="x", method_name="brd_visible_support_3", reviewer_id=1, show_support=True)

    assert board_one == board_two
    assert set(board_one.label_to_answer().values()) == {"a", "b"}
    assert "Observed independent answers" not in board_one.rendered()
    assert "Observed independent answers supporting it" in visible.rendered()


def test_bounded_board_keeps_every_candidate_and_shares_rationale_budget() -> None:
    stage = build_stage_a_decision(
        [
            {
                "normalized_answer": f"answer-{index}",
                "assistant_text": f"start-{index} " + ("reasoning " * 2000) + f" end-{index}",
                "agent_id": index,
            }
            for index in range(1, 6)
        ]
    )
    board = build_reviewer_board(
        stage,
        global_seed=42,
        sample_id="long-board",
        method_name="gsa_shared_panel",
        reviewer_id=1,
        show_support=False,
    )

    rendered = board.rendered(max_chars=6000)

    assert len(rendered) <= 6000
    assert rendered.count("[representative truncated]") == 5
    for label in "ABCDE":
        assert f"Candidate {label}\n" in rendered
    for answer in stage.candidate_answers:
        assert f"Proposed final answer: {answer}\n" in rendered


def test_error_correlation_reports_degenerate_panels_explicitly() -> None:
    diagnostics = reviewer_error_correlation(
        [
            {"reviewer_correctness": [True, False, False]},
            {"reviewer_correctness": [False, True, False]},
            {"reviewer_correctness": [False, False, True]},
        ]
    )

    assert diagnostics["triggered_samples"] == 3
    assert len(diagnostics["pairwise"]) == 3
