from __future__ import annotations

from types import SimpleNamespace

from research_experiments.families.contrastive_active_testing.algorithms import WitnessParseResult
from research_experiments.families.contrastive_active_testing.run.preflight import (
    _designer_gate,
    _witness_gate,
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
