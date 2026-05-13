"""覆盖无效运行与失效报告清理逻辑的测试。"""

from __future__ import annotations

from pathlib import Path

from research_experiments.tools.artifact_cleanup import (
    build_parser,
    cleanup_invalid_artifacts,
    collect_report_statuses,
    collect_run_statuses,
)
from testsupport.filesystem import touch_figure_contract, write_json, write_jsonl


def test_collect_run_statuses_flags_failed_multi_agent_run(tmp_path: Path) -> None:
    workspace_root = tmp_path
    runs_root = workspace_root / "runs"

    valid_run = runs_root / "multi_agent" / "valid_run"
    valid_run.mkdir(parents=True)
    write_json(valid_run / "manifest.json", {"run_id": "20260429T000001Z-valid"})
    write_jsonl(valid_run / "agent_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(valid_run / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(valid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    write_json(valid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(valid_run / "cost_breakdown.json", {"rows": []})
    write_json(valid_run / "debate_diagnostics.json", {"rows": []})
    touch_figure_contract(valid_run)

    invalid_run = runs_root / "multi_agent" / "invalid_run"
    invalid_run.mkdir(parents=True)
    write_json(invalid_run / "manifest.json", {"run_id": "20260429T000002Z-invalid"})
    write_jsonl(invalid_run / "agent_turns.jsonl", [{"output_status": "request_fail"}])
    write_jsonl(invalid_run / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(invalid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    write_json(invalid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(invalid_run / "cost_breakdown.json", {"rows": []})
    write_json(invalid_run / "debate_diagnostics.json", {"rows": []})

    statuses = collect_run_statuses(workspace_root=workspace_root, runs_root=runs_root)
    status_by_id = {status.run_id: status for status in statuses}

    assert status_by_id["20260429T000001Z-valid"].reason is None
    assert status_by_id["20260429T000002Z-invalid"].reason == "validation_failed"


def test_collect_report_statuses_flags_missing_and_failed_run_ids(tmp_path: Path) -> None:
    reports_root = tmp_path / "reports"
    reports_root.mkdir()
    valid_report = reports_root / "valid.md"
    valid_report.write_text("run 20260429T000001Z-valid", encoding="utf-8")
    missing_report = reports_root / "missing.md"
    missing_report.write_text("run 20260429T000099Z-missing", encoding="utf-8")
    failed_report = reports_root / "failed.md"
    failed_report.write_text("run 20260429T000002Z-failed", encoding="utf-8")

    run_status_by_id = {
        "20260429T000001Z-valid": _fake_run_status(True),
        "20260429T000002Z-failed": _fake_run_status(False),
    }

    statuses = collect_report_statuses(reports_root=reports_root, run_status_by_id=run_status_by_id)
    status_by_name = {status.report_path.name: status for status in statuses}

    assert status_by_name["valid.md"].is_invalid is False
    assert status_by_name["missing.md"].missing_run_ids == ("20260429T000099Z-missing",)
    assert status_by_name["failed.md"].failed_run_ids == ("20260429T000002Z-failed",)


def test_cleanup_invalid_artifacts_deletes_invalid_runs_and_reports(tmp_path: Path) -> None:
    workspace_root = tmp_path
    runs_root = workspace_root / "runs"
    reports_root = workspace_root / "reports"

    valid_run = runs_root / "multi_agent" / "valid_run"
    valid_run.mkdir(parents=True)
    write_json(valid_run / "manifest.json", {"run_id": "20260429T000001Z-valid"})
    write_jsonl(valid_run / "agent_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(valid_run / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(valid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    write_json(valid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(valid_run / "cost_breakdown.json", {"rows": []})
    write_json(valid_run / "debate_diagnostics.json", {"rows": []})
    touch_figure_contract(valid_run)

    invalid_run = runs_root / "multi_agent" / "invalid_run"
    invalid_run.mkdir(parents=True)
    write_json(invalid_run / "manifest.json", {"run_id": "20260429T000002Z-invalid"})
    write_jsonl(invalid_run / "agent_turns.jsonl", [{"output_status": "request_fail"}])
    write_jsonl(invalid_run / "debate_messages.jsonl", [{"x": 1}])
    write_jsonl(invalid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    write_json(invalid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(invalid_run / "cost_breakdown.json", {"rows": []})
    write_json(invalid_run / "debate_diagnostics.json", {"rows": []})

    reports_root.mkdir()
    (reports_root / "keep.md").write_text("20260429T000001Z-valid", encoding="utf-8")
    (reports_root / "drop_missing.md").write_text("20260429T000099Z-missing", encoding="utf-8")
    (reports_root / "drop_failed.md").write_text("20260429T000002Z-invalid", encoding="utf-8")

    summary = cleanup_invalid_artifacts(
        workspace_root=workspace_root,
        runs_root=runs_root,
        reports_root=reports_root,
        dry_run=False,
    )

    assert len(summary.invalid_runs) == 1
    assert len(summary.invalid_reports) == 2
    assert not invalid_run.exists()
    assert valid_run.exists()
    assert not (reports_root / "drop_missing.md").exists()
    assert not (reports_root / "drop_failed.md").exists()
    assert (reports_root / "keep.md").exists()


def test_collect_run_statuses_prefers_existing_run_validation(tmp_path: Path) -> None:
    workspace_root = tmp_path
    runs_root = workspace_root / "runs"
    legacy_run = runs_root / "single_agent" / "legacy_run"
    legacy_run.mkdir(parents=True)
    write_json(legacy_run / "manifest.json", {"run_id": "20260429T000003Z-legacy"})
    write_json(legacy_run / "run_validation.json", {"passed": True, "missing_files": []})

    statuses = collect_run_statuses(workspace_root=workspace_root, runs_root=runs_root)
    status = statuses[0]

    assert status.run_id == "20260429T000003Z-legacy"
    assert status.passed is True
    assert status.reason is None


def test_artifact_cleanup_parser_defaults_follow_local_workspace() -> None:
    args = build_parser().parse_args([])
    assert args.runs_root == "local/runs"
    assert args.reports_root == "local/reports"


def test_collect_run_statuses_recognizes_cue_family(tmp_path: Path) -> None:
    workspace_root = tmp_path
    runs_root = workspace_root / "local" / "runs"
    cue_run = runs_root / "cue" / "valid_run"
    cue_run.mkdir(parents=True)
    write_json(cue_run / "manifest.json", {"run_id": "20260429T000004Z-cue"})
    write_jsonl(cue_run / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    write_jsonl(cue_run / "communication_turns.jsonl", [])
    write_jsonl(cue_run / "audit_turns.jsonl", [])
    write_jsonl(
        cue_run / "policy_predictions.jsonl",
        [
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "always_communicate",
                "stage_a_trace_hash": "stage-a",
            },
            {
                "dataset": "gsm8k",
                "sample_id": "gsm8k-00001",
                "method_name": "cue_v1",
                "stage_a_trace_hash": "stage-a",
            },
        ],
    )
    write_json(cue_run / "policy_metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    write_json(cue_run / "policy_diagnostics.json", {"policy_rows": []})
    write_json(cue_run / "oracle_trigger_eval.json", {"summary_rows": []})
    (cue_run / "progress.json").write_text("{}", encoding="utf-8")
    touch_figure_contract(cue_run)

    statuses = collect_run_statuses(workspace_root=workspace_root, runs_root=runs_root, revalidate_runs=True)
    status = next(item for item in statuses if item.run_id == "20260429T000004Z-cue")
    assert status.reason is None

def _fake_run_status(passed: bool):
    from research_experiments.tools.artifact_cleanup import RunStatus

    return RunStatus(
        family_name="multi_agent",
        run_dir=Path("local/runs/multi_agent/fake"),
        run_id="fake",
        exists=True,
        passed=passed,
        reason=None if passed else "validation_failed",
    )


