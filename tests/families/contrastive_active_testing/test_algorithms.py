from __future__ import annotations

from research_experiments.families.contrastive_active_testing.algorithms import (
    CandidateClass,
    Commitment,
    DiagnosticTest,
    StageDecision,
    build_witness_packet,
    decide_direct_judges,
    decode_witnesses,
    effective_pair_coordinates,
    parse_witness_answers,
    select_tests,
    shuffle_commitments,
    validate_test_bank,
)
from research_experiments.families.contrastive_active_testing.algorithms import (
    TestOutcome as Outcome,
)


def _stage() -> StageDecision:
    return StageDecision(
        anchor_key="A",
        anchor_answer="A",
        candidates=(
            CandidateClass("A", "A", 3, "alpha evidence one two three", "sha-a"),
            CandidateClass("B", "B", 2, "beta evidence four five six", "sha-b"),
        ),
        vote_counts={"A": 3, "B": 2},
        valid_count=5,
    )


def _test(test_id: str, a_span: tuple[int, int], b_span: tuple[int, int]) -> DiagnosticTest:
    stage = _stage()
    a_text = stage.candidates[0].representative_reasoning[slice(*a_span)]
    b_text = stage.candidates[1].representative_reasoning[slice(*b_span)]
    return DiagnosticTest(
        test_id,
        f"Does diagnostic fact {test_id} hold?",
        (Outcome("O0", "yes"), Outcome("O1", "no")),
        {
            "A": Commitment("O0", *a_span, a_text, f"a-{test_id}"),
            "B": Commitment("O1", *b_span, b_text, f"b-{test_id}"),
        },
    )


def test_test_bank_requires_trace_backed_discriminating_commitments() -> None:
    stage = _stage()
    payload = {
        "tests": [
            {
                "test_id": "T0",
                "question": "Is the decisive quantity even?",
                "outcomes": [{"id": "O0", "text": "even"}, {"id": "O1", "text": "odd"}],
                "commitments": {
                    "H0": {"outcome_id": "O0", "trace_start": 0, "trace_end": 5},
                    "H1": {"outcome_id": "O1", "trace_start": 0, "trace_end": 4},
                },
            },
            {
                "test_id": "T1",
                "question": "Which option is the final answer?",
                "outcomes": [{"id": "O0", "text": "A"}, {"id": "O1", "text": "B"}],
                "commitments": {"H0": None, "H1": None},
            },
        ]
    }
    result = validate_test_bank(payload, stage=stage, hypothesis_to_key={"H0": "A", "H1": "B"})

    assert [test.test_id for test in result.tests] == ["T0"]
    assert result.tests[0].commitments["A"].evidence == "alpha"
    assert result.dropped == ({"test_id": "T1", "reason": "answer_or_candidate_leakage"},)


def test_overlapping_trace_spans_do_not_inflate_effective_distance() -> None:
    tests = (
        _test("T0", (0, 5), (0, 4)),
        _test("T1", (3, 9), (2, 8)),
        _test("T2", (15, 18), (14, 18)),
    )
    coordinates = effective_pair_coordinates(tests, "A", "B")
    assert len(coordinates) == 2
    assert "T2" in {test.test_id for test in coordinates}


def test_max_min_selection_and_double_witness_decoder_are_candidate_restricted() -> None:
    stage = _stage()
    tests = (
        _test("T0", (0, 5), (0, 4)),
        _test("T1", (6, 14), (5, 13)),
        _test("T2", (15, 18), (14, 18)),
    )
    selection = select_tests(tests, stage=stage, d_min=3, max_selected=4)
    assert selection.pair_distances == {"B": 3}

    decision = decode_witnesses(
        stage,
        selection.tests,
        [{"T0": "O1", "T1": "O1", "T2": "O1"}] * 2,
        d_min=3,
        margin=1,
    )
    assert decision.override_accepted
    assert decision.answer == "B"

    ambiguous = decode_witnesses(
        stage,
        selection.tests,
        [{"T0": "O0", "T1": "O1", "T2": "O0"}] * 2,
        d_min=3,
        margin=1,
    )
    assert not ambiguous.override_accepted and ambiguous.answer == "A"


def test_witness_permutations_map_ids_back_and_omissions_are_erasures() -> None:
    packet = build_witness_packet([_test("T0", (0, 5), (0, 4))], seed=42, sample_id="x", panel_index=1)
    public_test = next(iter(packet.public_test_to_internal))
    public_outcome = next(iter(packet.public_outcome_to_internal[public_test]))
    vector = parse_witness_answers(
        {"answers": [{"test_id": public_test, "outcome_id": public_outcome, "check": "local check"}]},
        packet=packet,
    )
    assert vector == {
        packet.public_test_to_internal[public_test]: packet.public_outcome_to_internal[public_test][public_outcome]
    }
    assert parse_witness_answers({"answers": []}, packet=packet) == {}
    assert parse_witness_answers(
        {"answers": [{"test_id": "UNKNOWN", "outcome_id": public_outcome, "check": "x"}]},
        packet=packet,
    ) is None


def test_direct_judge_cannot_introduce_a_novel_answer() -> None:
    stage = _stage()
    assert decide_direct_judges(stage, ["B", "B", "NOVEL"]) == (
        "B",
        True,
        "direct_judge_majority_override",
    )
    assert decide_direct_judges(stage, ["NOVEL", None, "X"]) == (
        "A",
        False,
        "no_valid_judge_vote",
    )


def test_signature_shuffle_is_a_non_identity_negative_control() -> None:
    stage = _stage()
    original = (_test("T0", (0, 5), (0, 4)),)
    shuffled = shuffle_commitments(original, stage=stage, seed=42, sample_id="x")
    assert shuffled[0].commitments["A"].outcome_id == original[0].commitments["B"].outcome_id
    assert shuffled[0].commitments["B"].outcome_id == original[0].commitments["A"].outcome_id


def test_decoder_corrects_every_received_word_inside_half_distance() -> None:
    stage = _stage()
    tests = (
        _test("T0", (0, 5), (0, 4)),
        _test("T1", (6, 14), (5, 13)),
        _test("T2", (15, 18), (14, 18)),
    )
    for error_index in (None, 0, 1, 2):
        vector = {test.test_id: "O1" for test in tests}
        if error_index is not None:
            vector[tests[error_index].test_id] = "O0"
        decision = decode_witnesses(stage, tests, [vector, vector], d_min=3, margin=1)
        assert decision.answer == "B" and decision.override_accepted


def test_multiple_challengers_passing_forces_abstention() -> None:
    stage = StageDecision(
        anchor_key="A",
        anchor_answer="A",
        candidates=(
            CandidateClass("A", "A", 3, "anchor zero one", "a"),
            CandidateClass("B", "B", 1, "bravo zero one", "b"),
            CandidateClass("C", "C", 1, "charlie zero one", "c"),
        ),
        vote_counts={"A": 3, "B": 1, "C": 1},
        valid_count=5,
    )
    tests = []
    for index, span in enumerate(((0, 3), (7, 10))):
        tests.append(
            DiagnosticTest(
                f"T{index}",
                f"Does shared challenger fact {index} hold?",
                (Outcome("O0", "anchor"), Outcome("O1", "challenger")),
                {
                    "A": Commitment("O0", *span, "a", f"a{index}"),
                    "B": Commitment("O1", *span, "b", f"b{index}"),
                    "C": Commitment("O1", *span, "c", f"c{index}"),
                },
            )
        )
    decision = decode_witnesses(
        stage,
        tests,
        [{"T0": "O1", "T1": "O1"}] * 2,
        d_min=2,
        margin=1,
    )
    assert not decision.override_accepted
    assert decision.answer == "A"
    assert decision.resolver == "multiple_challengers"
