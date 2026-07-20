from __future__ import annotations

import json

import pytest

from research_experiments.families.contrastive_active_testing.mechanism_factorial import (
    summarize_factorial_results,
    write_factorial_template,
)


def test_factorial_template_has_120_cases_by_four_cells_and_attributes_losses(tmp_path) -> None:
    audit = {
        "cases": [
            {
                "sample_id": f"s{index}",
                "dataset": "bbeh",
                "task": "unit",
                "case_class": "recoverable_wrong",
            }
            for index in range(120)
        ]
    }
    audit_path = tmp_path / "audit.json"
    template_path = tmp_path / "factorial.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    payload = write_factorial_template(audit_path, template_path)
    assert payload["case_count"] == 120
    assert payload["row_count"] == 480

    accuracy_by_cell = {
        "model_certificate__model_verifier": 0.4,
        "human_certificate__model_verifier": 0.7,
        "model_certificate__human_or_deterministic_verifier": 0.6,
        "human_certificate__human_or_deterministic_verifier": 0.9,
    }
    counters = {key: 0 for key in accuracy_by_cell}
    for row in payload["rows"]:
        index = counters[row["cell_id"]]
        counters[row["cell_id"]] += 1
        row.update(
            {
                "completed": True,
                "initial_correct": False,
                "final_correct": index < 120 * accuracy_by_cell[row["cell_id"]],
                "correct_challenger_certificate_present": True,
                "mandatory_obligations_complete": True,
                "commitment_direction_correct": True,
                "verifier_outcome_correct": True,
                "false_pass": False,
                "false_reject": False,
            }
        )
    template_path.write_text(json.dumps(payload), encoding="utf-8")
    summary = summarize_factorial_results(template_path)
    assert summary["attribution"]["designer_loss"] == pytest.approx(0.3)
    assert summary["attribution"]["verifier_loss"] == pytest.approx(0.2)
    assert summary["paper_branch"] == "designer_bottleneck"
