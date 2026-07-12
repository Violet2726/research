from __future__ import annotations

from research_experiments.families.risk_controlled_trace_mad.analyze_cross_backbone import (
    PRE_REGISTERED_COMPETITORS,
    analyze_cross_backbone_rows,
)


def _rows(model: str) -> list[dict]:
    rows = []
    for dataset in ("omni_math_2_filtered", "bbeh"):
        for index in range(20):
            task = f"task-{index % 2}"
            rows.append(
                {
                    "dataset": dataset,
                    "sample_id": f"{dataset}-{index}",
                    "task": task,
                    "method_name": "rcta_1",
                    "model_name": model,
                    "score": 1.0,
                    "total_tokens_per_question": 5.0,
                    "corrected_by_debate": index < 10,
                    "harmed_by_debate": False,
                }
            )
            for method in PRE_REGISTERED_COMPETITORS:
                rows.append(
                    {
                        "dataset": dataset,
                        "sample_id": f"{dataset}-{index}",
                        "task": task,
                        "method_name": method,
                        "model_name": model,
                        "score": float(index % 2 == 0),
                        "total_tokens_per_question": 6.0,
                    }
                )
    return rows


def test_cross_backbone_analysis_uses_equal_cells_and_enforces_full_gate() -> None:
    result = analyze_cross_backbone_rows(
        {"qwen": _rows("qwen"), "mimo": _rows("mimo")},
        bootstrap_samples=2_000,
    )
    assert result["sota_gate_passed"] is True
    assert result["four_cell_gate"]["passed"] is True
    assert result["coverage_gate"]["harm_fraction_upper_95"] < 1 / 3
    assert len(result["comparisons"]) == 2 * len(PRE_REGISTERED_COMPETITORS)
