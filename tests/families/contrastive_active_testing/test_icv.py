from __future__ import annotations

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.algorithms import (
    CandidateClass,
    StageDecision,
)
from research_experiments.families.contrastive_active_testing.icv import (
    IcvWitnessParseResult,
    build_icv_witness_packet,
    build_target_pairs,
    decode_icv,
    parse_icv_witness,
    segment_reasoning_evidence,
    segment_stage_evidence,
    validate_contrast_selector,
)


def _stage() -> StageDecision:
    candidates = (
        CandidateClass("A", "A", 3, "The alpha premise is established from source. The alpha implication follows from context. The alpha boundary condition remains satisfied.", "a"),
        CandidateClass("B", "B", 2, "The beta premise is established from source. The beta implication follows from context. The beta boundary condition remains satisfied.", "b"),
    )
    return StageDecision("A", "A", candidates, {"A": 3, "B": 2}, 5)


def _sample() -> DatasetSample:
    return DatasetSample("bbeh", "icv", "Source facts\nOptions:\n(A) alpha\n(B) beta", "B", "", {"task": "unit", "options": [{"label": "A", "text": "alpha"}, {"label": "B", "text": "beta"}]})


def _valid_selector(stage, pairs, evidence):
    pair = pairs[0]
    rows = []
    for index in range(3):
        rows.append(
            {
                "pair_id": pair.pair_id,
                "contrast_id": f"C{index}",
                "left_unit_ids": [f"L:E{index}"],
                "right_unit_ids": [f"R:E{index}"],
            }
        )
    return {"contrasts": rows}


def test_selector_ids_resolve_to_fixed_three_coordinate_packet() -> None:
    stage = _stage()
    pairs = build_target_pairs(stage, seed=42, sample_id="icv")
    evidence = segment_stage_evidence(_sample(), stage)
    result = validate_contrast_selector(_valid_selector(stage, pairs, evidence), pairs=pairs, evidence=evidence)

    assert result.protocol_error is None
    assert len(result.coordinates) == 3
    assert result.eligible_challengers == ("B",)
    assert all(item.left_span[0] < item.left_span[1] for item in result.coordinates)


def test_reused_units_and_unknown_ids_are_coordinate_erasure_not_packet_invention() -> None:
    stage = _stage()
    pairs = build_target_pairs(stage, seed=42, sample_id="icv")
    evidence = segment_stage_evidence(_sample(), stage)
    payload = _valid_selector(stage, pairs, evidence)
    payload["contrasts"][1]["left_unit_ids"] = payload["contrasts"][0]["left_unit_ids"]
    payload["contrasts"][2]["right_unit_ids"] = ["R:E99"]
    result = validate_contrast_selector(payload, pairs=pairs, evidence=evidence)

    assert len(result.coordinates) == 1
    assert not result.eligible_challengers
    assert {item["reason"] for item in result.dropped} == {
        "overlapping_or_reused_evidence",
        "unknown_evidence_id",
    }


def test_witness_left_right_permutation_is_inverted_before_decode() -> None:
    stage = _stage()
    pairs = build_target_pairs(stage, seed=42, sample_id="icv")
    evidence = segment_stage_evidence(_sample(), stage)
    validation = validate_contrast_selector(_valid_selector(stage, pairs, evidence), pairs=pairs, evidence=evidence)
    panels = []
    for panel_index in (1, 2):
        packet = build_icv_witness_packet(validation.coordinates, seed=42, sample_id="icv", panel_index=panel_index)
        answers = []
        for public_id in packet.public_to_internal:
            verdict = "LEFT_ONLY" if packet.public_left_to_candidate[public_id] == "B" else "RIGHT_ONLY"
            answers.append({"contrast_id": public_id, "verdict": verdict})
        panels.append(parse_icv_witness({"answers": answers}, packet=packet))
    decision = decode_icv(stage, validation.coordinates, tuple(panels))

    assert decision.override_accepted
    assert decision.answer_key == "B"


def test_repetition_code_corrects_one_error_or_erasure_and_abstains_on_two() -> None:
    stage = _stage()
    pairs = build_target_pairs(stage, seed=42, sample_id="icv")
    evidence = segment_stage_evidence(_sample(), stage)
    coordinates = validate_contrast_selector(_valid_selector(stage, pairs, evidence), pairs=pairs, evidence=evidence).coordinates
    ids = [item.contrast_id for item in coordinates]

    patterns = [("B", "B", "B")]
    for adverse in ("A", "ERASURE"):
        for position in range(3):
            values = ["B", "B", "B"]
            values[position] = adverse
            patterns.append(tuple(values))
    for values in patterns:
        observations = dict(zip(ids, values, strict=True))
        panel = IcvWitnessParseResult(True, observations, 3, 3, 2, ())
        assert decode_icv(stage, coordinates, (panel, panel)).answer_key == "B"

    observations = {ids[0]: "A", ids[1]: "ERASURE", ids[2]: "B"}
    panel = IcvWitnessParseResult(True, observations, 3, 3, 2, ())
    assert not decode_icv(stage, coordinates, (panel, panel)).override_accepted


def test_unknown_and_duplicate_witness_rows_erase_only_the_coordinate() -> None:
    stage = _stage()
    pairs = build_target_pairs(stage, seed=42, sample_id="icv")
    evidence = segment_stage_evidence(_sample(), stage)
    coordinates = validate_contrast_selector(_valid_selector(stage, pairs, evidence), pairs=pairs, evidence=evidence).coordinates
    packet = build_icv_witness_packet(coordinates, seed=42, sample_id="icv", panel_index=1)
    public = list(packet.public_to_internal)
    parsed = parse_icv_witness(
        {"answers": [
            {"contrast_id": public[0], "verdict": "LEFT_ONLY"},
            {"contrast_id": public[0], "verdict": "RIGHT_ONLY"},
            {"contrast_id": public[1], "verdict": "BOTH"},
            {"contrast_id": "unknown", "verdict": "LEFT_ONLY"},
        ]},
        packet=packet,
    )
    assert parsed.top_level_valid
    assert parsed.valid_coordinate_count == 1
    assert set(parsed.observations.values()) == {"ERASURE"}


def test_evidence_segmentation_rejects_answer_labels_and_final_conclusions() -> None:
    units = segment_reasoning_evidence(
        _sample(),
        candidate_key="B",
        answer="B",
        reasoning=(
            "The local premise is independently supported. "
            "Therefore the answer is option B. "
            "FINAL_ANSWER: B"
        ),
    )
    assert any(unit.eligible for unit in units)
    assert any(not unit.eligible for unit in units)
    assert all("FINAL_ANSWER" not in unit.text for unit in units if unit.eligible)

    for conclusion in (
        "Thus the answer is A.",
        "Hence, the option is A.",
        "因此最终答案是 A。",
    ):
        conclusion_units = segment_reasoning_evidence(
            _sample(),
            candidate_key="A",
            answer="A",
            reasoning=conclusion,
        )
        assert conclusion_units
        assert all(not unit.eligible for unit in conclusion_units)


def test_evidence_offsets_and_hashes_are_nfkc_stable() -> None:
    reasoning = (
        "\uff34\uff48\uff45 premise contains fullwidth text and remains independently checkable."
    )
    first = segment_reasoning_evidence(_sample(), candidate_key="A", answer="A", reasoning=reasoning)
    second = segment_reasoning_evidence(_sample(), candidate_key="A", answer="A", reasoning=reasoning)
    assert first == second
    assert first[0].text.startswith("The premise")
    assert first[0].start == 0
    assert first[0].end == len(first[0].text)
