
"""覆盖 JSONL 落盘器与运行收尾逻辑。"""

from __future__ import annotations

from pathlib import Path
import json
import time

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

