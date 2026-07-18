from __future__ import annotations

from types import SimpleNamespace

from research_experiments.families.contrastive_active_testing.algorithms import WitnessParseResult
from research_experiments.families.contrastive_active_testing.icv import IcvWitnessParseResult
from research_experiments.families.contrastive_active_testing.run.preflight import (
    _designer_gate,
    _icv_selector_gate,
    _icv_witness_gate,
    _witness_gate,
    evaluate_icv_human_audit,
)


def test_designer_preflight_uses_structural_thresholds_without_gold() -> None:
    states = []
    for index in range(20):
        validation = SimpleNamespace(
            evidence_quote_count=5,
            aligned_evidence_quote_count=5 if index < 19 else 0,
            leakage_count=0,
            protocol_error=None,
        )
        selection = SimpleNamespace(pair_distances={"B": 2} if index < 12 else {"B": 1})
        states.append(
            SimpleNamespace(
                validation=validation,
                row={"request_error": None},
                selection=selection,
            )
        )
    gate = _designer_gate(states, expected_count=20, quote_threshold=0.95, coverage_threshold=0.60)
    assert gate["passed"]
    assert gate["d2_code_packet_coverage"] == 0.6
    assert gate["evidence_quote_alignment_rate"] == 0.95
    assert "gold" not in gate


def test_witness_preflight_requires_valid_coordinates_and_two_usable_panels() -> None:
    good = WitnessParseResult({"T0": "O0", "T1": "O1"}, True, 2, 2, ())
    results = [
        {"parsed": [good, good], "usable": index < 9}
        for index in range(10)
    ]
    gate = _witness_gate(results, coordinate_threshold=0.95, usable_pair_threshold=0.90)
    assert gate["passed"]

    failed = _witness_gate(
        [{"parsed": [WitnessParseResult(None, False, 2, 0, ()), good], "usable": False}],
        coordinate_threshold=0.95,
        usable_pair_threshold=0.90,
    )
    assert not failed["passed"]


def test_icv_preflight_freezes_selector_and_witness_measurement_thresholds() -> None:
    states = []
    for index in range(20):
        validation = SimpleNamespace(
            protocol_error=None,
            coordinates=(object(), object(), object()),
            dropped=(),
            leakage_count=0,
            eligible_challengers=("B",) if index < 12 else (),
        )
        states.append(SimpleNamespace(validation=validation, row={"request_error": None}))
    selector = _icv_selector_gate(states, expected_count=20, coverage_threshold=0.60)
    assert selector["passed"]
    assert selector["eligible_sample_rate"] == 0.60

    good_first = IcvWitnessParseResult(True, {"C0": "B", "C1": "B", "C2": "A"}, 3, 3, 3, ())
    good_second = IcvWitnessParseResult(True, {"C0": "B", "C1": "B", "C2": "A"}, 3, 3, 3, ())
    results = [
        {"parsed": [good_first, good_second], "eligible_pair_count": 1, "usable_pair_count": int(index < 9)}
        for index in range(10)
    ]
    witness = _icv_witness_gate(
        results,
        coordinate_threshold=0.95,
        decisive_threshold=0.80,
        usable_pair_threshold=0.90,
        agreement_threshold=0.70,
    )
    assert witness["passed"]
    assert witness["panel_agreement_rate"] == 1.0


def test_human_audit_is_recomputed_and_requires_disagreement_adjudication() -> None:
    hashes = {f"h{index}" for index in range(40)}
    items = []
    for index, coordinate_hash in enumerate(sorted(hashes)):
        first = {
            "decidable": True,
            "mutually_exclusive": True,
            "atomic": index >= 4,
            "answer_leakage": False,
        }
        second = dict(first)
        adjudication = None
        if index == 0:
            second["decidable"] = False
        items.append(
            {
                "coordinate_sha256": coordinate_hash,
                "annotator_1": first,
                "annotator_2": second,
                "adjudication": adjudication,
            }
        )
    payload = {
        "audit_version": "catch_v3_icv_blind_coordinate_audit_v1",
        "blind_to_gold_votes_and_candidate_answers": True,
        "items": items,
    }
    failed = evaluate_icv_human_audit(payload, expected_coordinate_hashes=hashes)
    assert not failed["passed"]
    assert failed["disagreement_count"] == 1
    assert failed["adjudicated_disagreement_count"] == 0

    items[0]["adjudication"] = {"decidable": True}
    passed = evaluate_icv_human_audit(payload, expected_coordinate_hashes=hashes)
    assert passed["passed"]
    assert passed["rates"]["answer_leakage"] == 0.0
    assert passed["adjudicated_disagreement_count"] == 1
