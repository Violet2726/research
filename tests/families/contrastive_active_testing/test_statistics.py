from __future__ import annotations

from research_experiments.families.contrastive_active_testing.statistics import (
    _clopper_pearson_lower,
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

