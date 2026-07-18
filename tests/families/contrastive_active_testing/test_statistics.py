from __future__ import annotations

import pytest

from research_experiments.families.contrastive_active_testing.statistics import (
    _clopper_pearson_lower,
    _v3_observation_diagnostics,
    _v3_panel_dependence,
    materialize_development_catch,
)


def test_development_grid_selection_is_global_and_materializes_one_primary_method() -> None:
    predictions = []
    routers = []
    for index in range(10):
        base = {
            "sample_id": str(index),
            "dataset": "bbeh",
            "task": "t",
            "candidate_oracle_correct": True,
            "triggered": True,
            "total_tokens_per_question": 8,
        }
        predictions.extend(
            [
                {**base, "method_name": "catch_d2_m1", "score": 1, "override_accepted": True, "corrected_by_debate": index < 5, "harmed_by_debate": False},
                {**base, "method_name": "catch_d3_m2", "score": int(index < 7), "override_accepted": index < 7, "corrected_by_debate": index < 4, "harmed_by_debate": False},
            ]
        )
        routers.append(
            {
                "triggered": True,
                "catch_variants": [
                    {"d_min": 2, "margin": 1, "pair_distances": {"B": 2}},
                    {"d_min": 3, "margin": 2, "pair_distances": {"B": 3}},
                ],
            }
        )

    materialized, selection = materialize_development_catch(predictions, routers)
    assert selection["selected"]["method_name"] == "catch_d2_m1"
    assert len([row for row in materialized if row["method_name"] == "catch"]) == 10
    assert not any(row["method_name"] == "catch_d2_m1" for row in materialized)


def test_exact_one_sided_override_precision_bound_is_conservative() -> None:
    assert _clopper_pearson_lower(0, 0, alpha=0.05) == 0.0
    assert _clopper_pearson_lower(8, 10, alpha=0.05) < 0.65
    assert _clopper_pearson_lower(20, 20, alpha=0.05) > 0.85


def test_v3_reports_correlated_false_passes_and_position_diagnostics() -> None:
    routers = []
    for index, passes in enumerate(((True, True), (True, False), (False, False))):
        routers.append(
            {
                "protocol_version": "catch_v3",
                "gold_candidate_key": "A",
                "decision": {
                    "panel_diagnostics": [
                        {"challenger_key": "B", "panel_index": 1, "passed": passes[0]},
                        {"challenger_key": "B", "panel_index": 2, "passed": passes[1]},
                    ]
                },
                "witness_panels": [
                    {"observations": {"C0": "B" if index < 2 else "ERASURE"}},
                    {"observations": {"C0": "B" if index == 0 else "A"}},
                ],
            }
        )
    dependence = _v3_panel_dependence(routers)
    assert dependence["panel_1_false_pass_rate"] == 2 / 3
    assert dependence["panel_2_false_pass_rate"] == 1 / 3
    assert dependence["joint_false_pass_rate"] == 1 / 3
    assert dependence["bernoulli_correlation"] == pytest.approx(0.5)

    turns = [
        {
            "role": "icv_witness",
            "witness_packet": {"public_to_internal": {"X0": "C0", "X1": "C1"}},
            "validated_output": {
                "answers": [
                    {"contrast_id": "X0", "verdict": "LEFT_ONLY"},
                    {"contrast_id": "X1", "verdict": "RIGHT_ONLY"},
                ]
            },
        }
    ]
    observations = _v3_observation_diagnostics(routers, turns)
    assert observations["inverse_mapped_panel_agreement_rate"] == 0.5
    assert observations["left_only_share_among_decisive"] == 0.5
