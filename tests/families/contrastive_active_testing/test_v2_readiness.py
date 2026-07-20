from __future__ import annotations

import json

from research_experiments.families.contrastive_active_testing.run.report import render_report
from research_experiments.families.contrastive_active_testing.v2_readiness import (
    write_cert_v2_readiness_assessment,
)


def test_readiness_assessment_records_failures_without_becoming_a_gate(tmp_path) -> None:
    run = tmp_path / "run"
    (run / "turns").mkdir(parents=True)
    (run / "views").mkdir()
    (run / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "catch_cert_v2",
                "phase_name": "development",
                "run_id": "dev-run",
                "run_status": "completed",
                "frozen_config_sha256": "config",
            }
        ),
        encoding="utf-8",
    )
    (run / "run_validation.json").write_text(json.dumps({"artifact_valid": True}), encoding="utf-8")
    for relative in (
        "turns/router_decisions.jsonl",
        "turns/agent_turns.jsonl",
        "views/predictions.jsonl",
    ):
        (run / relative).write_text("", encoding="utf-8")

    audit = tmp_path / "audit.json"
    review = {"certificate_necessary": True, "certificate_sufficient": True}
    audit.write_text(
        json.dumps(
            {
                "seqbench_executor_golden_tests_passed": False,
                "cases": [{"reviewer_1": review, "reviewer_2": review} for _ in range(120)],
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "assessment.json"
    result = write_cert_v2_readiness_assessment(run, audit, output)

    assert result["enforcement"] == "advisory_only"
    assert result["blocks_execution"] is False
    assert result["all_recommended_conditions_met"] is False
    assert "seqbench_executor_golden_tests_passed" in result["unmet_conditions"]
    assert result["recommended_interpretation"] == "exploratory_diagnostic_evidence"
    assert json.loads(output.read_text(encoding="utf-8"))["sha256"] == result["sha256"]


def test_chinese_report_marks_unmet_readiness_as_advisory(tmp_path) -> None:
    (tmp_path / "views").mkdir()
    (tmp_path / "views" / "run_summary.json").write_text(json.dumps({"metrics": {}, "execution": {}}), encoding="utf-8")
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "protocol_version": "catch_cert_v2",
                "phase_name": "heldout",
                "readiness_assessment": {
                    "status": "available",
                    "blocks_execution": False,
                    "all_recommended_conditions_met": False,
                    "unmet_conditions": ["wrong_to_correct_exceeds_correct_to_wrong"],
                },
                "evidence_interpretation": "exploratory_diagnostic_evidence",
            }
        ),
        encoding="utf-8",
    )

    render_report(tmp_path)
    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "科研证据状态（非阻断）" in report
    assert "不会终止运行、删除失败样本或阻止后续阶段" in report
    assert "wrong_to_correct_exceeds_correct_to_wrong" in report
