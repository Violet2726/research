
"""覆盖 JSONL 落盘器与运行收尾逻辑。"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.runtime import RunProgressTracker, finalize_run_outputs


def test_buffered_jsonl_writer_writes_rows(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        writer = BufferedJsonlWriter(handle, flush_every=2, flush_interval_seconds=60.0)
        writer.write_row({"id": 1})
        writer.write_row({"id": 2})
        writer.write_row({"id": 3})
        writer.close()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]

def test_finalize_run_outputs_attaches_hf_publish_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "demo-run"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp_path / "report.md").write_text("# report\n", encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"summary": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "research_experiments.core.execution.runtime.publish_run_if_configured",
        lambda root, validation: {"published": True, "remote_repo": "owner/research-runs"},
    )

    payload = finalize_run_outputs(
        tmp_path,
        validator=lambda _: {"passed": True},
    )

    assert payload["hf_publish"]["published"] is True
    validation_payload = json.loads((tmp_path / "run_validation.json").read_text(encoding="utf-8"))
    assert validation_payload["hf_publish"]["remote_repo"] == "owner/research-runs"


def test_run_progress_tracker_heartbeat_refreshes_snapshot(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    tracker = RunProgressTracker(
        progress_path,
        total_planned_calls=100,
        total_planned_predictions=10,
        write_interval_seconds=0.01,
        heartbeat_interval_seconds=0.05,
    )
    initial = json.loads(progress_path.read_text(encoding="utf-8"))
    time.sleep(0.12)
    updated = json.loads(progress_path.read_text(encoding="utf-8"))
    tracker.mark_completed()

    assert initial["last_write_reason"] == "startup"
    assert updated["last_write_reason"] == "heartbeat"
    assert updated["last_updated_at"] != initial["last_updated_at"]
    assert "seconds_since_last_progress_event" in updated


def test_run_progress_tracker_close_stops_heartbeat(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    tracker = RunProgressTracker(
        progress_path,
        total_planned_calls=10,
        total_planned_predictions=2,
        write_interval_seconds=0.01,
        heartbeat_interval_seconds=0.05,
    )
    time.sleep(0.08)
    tracker.close()
    frozen = json.loads(progress_path.read_text(encoding="utf-8"))
    time.sleep(0.10)
    after_close = json.loads(progress_path.read_text(encoding="utf-8"))

    assert frozen["status"] == "running"
    assert after_close["last_updated_at"] == frozen["last_updated_at"]


def test_run_progress_tracker_separates_rolling_and_lifetime_network_rpm(tmp_path: Path) -> None:
    progress_path = tmp_path / "progress.json"
    tracker = RunProgressTracker(
        progress_path,
        total_planned_calls=3,
        total_planned_predictions=1,
        target_network_rpm=95,
        rate_limit_snapshot_provider=lambda: {
            "effective_network_rpm_limit": 95,
            "rate_limit_429_count": 0,
        },
        network_rpm_window_seconds=0.2,
        write_interval_seconds=0.01,
        heartbeat_interval_seconds=10.0,
    )
    try:
        tracker.record_call({"dataset": "gsm8k", "method_name": "cot", "sample_id": "1", "cache_hit": False})
        tracker.record_call({"dataset": "gsm8k", "method_name": "cot", "sample_id": "2", "cache_hit": True})
        time.sleep(0.02)
        tracker.write(force=True, reason="test-current")
        current = json.loads(progress_path.read_text(encoding="utf-8"))

        assert current["network_calls"] == 1
        assert current["cache_hits"] == 1
        assert current["cache_hit_ratio"] == 0.5
        assert current["observed_network_rpm"] > 0
        assert current["observed_call_rpm"] > current["observed_network_rpm"]
        assert current["lifetime_network_rpm"] > 0
        assert current["target_network_rpm"] == 95
        assert current["effective_network_rpm_limit"] == 95
        assert current["rate_limit_429_count"] == 0

        time.sleep(0.25)
        tracker.write(force=True, reason="test-window-expired")
        expired = json.loads(progress_path.read_text(encoding="utf-8"))

        assert expired["observed_network_rpm"] == 0
        assert expired["observed_call_rpm"] == 0
        assert expired["lifetime_network_rpm"] > 0
    finally:
        tracker.mark_completed()

