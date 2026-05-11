"""覆盖无效运行与失效报告清理逻辑的测试。"""

from __future__ import annotations

from pathlib import Path
import json

from experiment_core.tools.artifact_cleanup import (
    build_parser,
    cleanup_invalid_artifacts,
    collect_report_statuses,
    collect_run_statuses,
)


def test_collect_run_statuses_flags_failed_multi_agent_run(tmp_path: Path) -> None:
    workspace_root = tmp_path
    runs_root = workspace_root / "runs"

    valid_run = runs_root / "multi_agent" / "valid_run"
    valid_run.mkdir(parents=True)
    _touch_json(valid_run / "manifest.json", {"run_id": "20260429T000001Z-valid"})
    _write_jsonl(valid_run / "agent_turns.jsonl", [{"output_status": "ok"}])
    _write_jsonl(valid_run / "debate_messages.jsonl", [{"x": 1}])
    _write_jsonl(valid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    _touch_json(valid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(valid_run / "cost_breakdown.json", {"rows": []})
    _touch_json(valid_run / "debate_diagnostics.json", {"rows": []})
    _touch_figure_contract(valid_run)

    invalid_run = runs_root / "multi_agent" / "invalid_run"
    invalid_run.mkdir(parents=True)
    _touch_json(invalid_run / "manifest.json", {"run_id": "20260429T000002Z-invalid"})
    _write_jsonl(invalid_run / "agent_turns.jsonl", [{"output_status": "request_fail"}])
    _write_jsonl(invalid_run / "debate_messages.jsonl", [{"x": 1}])
    _write_jsonl(invalid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    _touch_json(invalid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(invalid_run / "cost_breakdown.json", {"rows": []})
    _touch_json(invalid_run / "debate_diagnostics.json", {"rows": []})

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
    _touch_json(valid_run / "manifest.json", {"run_id": "20260429T000001Z-valid"})
    _write_jsonl(valid_run / "agent_turns.jsonl", [{"output_status": "ok"}])
    _write_jsonl(valid_run / "debate_messages.jsonl", [{"x": 1}])
    _write_jsonl(valid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    _touch_json(valid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(valid_run / "cost_breakdown.json", {"rows": []})
    _touch_json(valid_run / "debate_diagnostics.json", {"rows": []})
    _touch_figure_contract(valid_run)

    invalid_run = runs_root / "multi_agent" / "invalid_run"
    invalid_run.mkdir(parents=True)
    _touch_json(invalid_run / "manifest.json", {"run_id": "20260429T000002Z-invalid"})
    _write_jsonl(invalid_run / "agent_turns.jsonl", [{"output_status": "request_fail"}])
    _write_jsonl(invalid_run / "debate_messages.jsonl", [{"x": 1}])
    _write_jsonl(invalid_run / "final_predictions.jsonl", [{"method_name": "mad_3a_r1"}])
    _touch_json(invalid_run / "metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(invalid_run / "cost_breakdown.json", {"rows": []})
    _touch_json(invalid_run / "debate_diagnostics.json", {"rows": []})

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
    _touch_json(legacy_run / "manifest.json", {"run_id": "20260429T000003Z-legacy"})
    _touch_json(legacy_run / "run_validation.json", {"passed": True, "missing_files": []})

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
    _touch_json(cue_run / "manifest.json", {"run_id": "20260429T000004Z-cue"})
    _write_jsonl(cue_run / "stage_a_turns.jsonl", [{"output_status": "ok"}])
    _write_jsonl(cue_run / "communication_turns.jsonl", [])
    _write_jsonl(cue_run / "audit_turns.jsonl", [])
    _write_jsonl(
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
    _touch_json(cue_run / "policy_metrics.json", {"summary": [{"dataset": "gsm8k"}]})
    _touch_json(cue_run / "policy_diagnostics.json", {"policy_rows": []})
    _touch_json(cue_run / "oracle_trigger_eval.json", {"summary_rows": []})
    (cue_run / "progress.json").write_text("{}", encoding="utf-8")
    _touch_figure_contract(cue_run)

    statuses = collect_run_statuses(workspace_root=workspace_root, runs_root=runs_root, revalidate_runs=True)
    status = next(item for item in statuses if item.run_id == "20260429T000004Z-cue")
    assert status.reason is None


def _touch_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def _fake_run_status(passed: bool):
    from experiment_core.tools.artifact_cleanup import RunStatus

    return RunStatus(
        package_name="multi_agent",
        run_dir=Path("local/runs/multi_agent/fake"),
        run_id="fake",
        exists=True,
        passed=passed,
        reason=None if passed else "validation_failed",
    )


def _touch_figure_contract(root: Path) -> None:
    figures_dir = root / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for figure_id in ("frontier_overall", "efficiency_rank_overall", "score_by_dataset"):
        (figures_dir / f"{figure_id}.svg").write_text("<svg></svg>\n", encoding="utf-8")
        (figures_dir / f"{figure_id}.csv").write_text("label,value\nx,1\n", encoding="utf-8")
        rows.append(
            {
                "figure_id": figure_id,
                "title": figure_id,
                "caption": "test",
                "svg_path": f"figures/{figure_id}.svg",
                "csv_path": f"figures/{figure_id}.csv",
                "source_kind": "test",
                "dataset_scope": "overall",
                "primary_metric": "Accuracy",
            }
        )
    (root / "figure_manifest.json").write_text(
        json.dumps({"generated_at": "2026-01-01T00:00:00+00:00", "figure_count": len(rows), "figures": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (root / "report.md").write_text("# report\n\n![Frontier](figures/frontier_overall.svg)\n", encoding="utf-8")
    (root / "archive_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": "2026-01-01T00:00:00+00:00",
                "run_id": "cleanup-test",
                "remote_repo": None,
                "remote_prefix": "multi_agent/test/cleanup-test",
                "artifacts_packaged": False,
                "visible_files": [
                    "report.md",
                    "figure_manifest.json",
                    "figures/frontier_overall.svg",
                    "figures/frontier_overall.csv",
                    "figures/efficiency_rank_overall.svg",
                    "figures/efficiency_rank_overall.csv",
                    "figures/score_by_dataset.svg",
                    "figures/score_by_dataset.csv",
                ],
                "archives": [],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

