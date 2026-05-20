"""Shared runtime utilities for long-running experiment families."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
from threading import Event, RLock, Thread
import time
from typing import Any, Callable

from research_experiments.workspace.run_archives import pack_run_artifacts, publish_run_if_configured


class RunProgressTracker:
    """Persist a live run heartbeat and coarse progress snapshot to disk."""

    def __init__(
        self,
        progress_path: Path,
        total_planned_calls: int,
        total_planned_predictions: int,
        *,
        initial_completed_calls: int = 0,
        initial_completed_predictions: int = 0,
        planned_calls_are_upper_bound: bool = False,
        write_interval_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 3.0,
    ) -> None:
        self.progress_path = progress_path
        self.total_planned_calls = total_planned_calls
        self.total_planned_predictions = total_planned_predictions
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.completed_calls = initial_completed_calls
        self.completed_predictions = initial_completed_predictions
        self.planned_calls_are_upper_bound = planned_calls_are_upper_bound
        self.cache_hits = 0
        self.network_calls = 0
        self.last_dataset: str | None = None
        self.last_method: str | None = None
        self.last_sample_id: str | None = None
        self.status = "running"
        self.write_interval_seconds = max(0.1, float(write_interval_seconds))
        self.heartbeat_interval_seconds = max(0.1, float(heartbeat_interval_seconds))
        self.last_write_monotonic = 0.0
        self.last_progress_event_monotonic = self.started_monotonic
        self.last_progress_event_at = self.started_at
        self.last_write_reason = "startup"
        self._lock = RLock()
        self._stop_event = Event()
        self._heartbeat_thread = Thread(
            target=self._heartbeat_loop,
            name=f"progress-heartbeat-{progress_path.stem}",
            daemon=True,
        )
        self.write(force=True, reason="startup")
        self._heartbeat_thread.start()

    def record_call(self, row: dict[str, object], method_key: str = "method_name") -> None:
        """Record one completed low-level model call."""

        with self._lock:
            self.completed_calls += 1
            if row.get("cache_hit"):
                self.cache_hits += 1
            else:
                self.network_calls += 1
            self.last_dataset = str(row.get("dataset") or "")
            self.last_method = str(row.get(method_key) or "")
            self.last_sample_id = str(row.get("sample_id") or "")
            self._note_progress_event_locked()
            force = self.completed_calls % 10 == 0
        self.write(force=force, reason="call")

    def record_predictions(self, count: int, dataset: str, method_name: str) -> None:
        """Record that question-level predictions have been flushed to disk."""

        with self._lock:
            self.completed_predictions += count
            self.last_dataset = dataset
            self.last_method = method_name
            self._note_progress_event_locked()
        self.write(force=True, reason="prediction")

    def mark_completed(self) -> None:
        """Mark the run as completed and stop background heartbeats."""

        with self._lock:
            self.status = "completed"
            self._note_progress_event_locked()
        self.write(force=True, reason="completed")
        self._stop_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.2)

    def write(self, force: bool = False, *, reason: str | None = None) -> None:
        """Refresh the persisted progress snapshot with throttling."""

        with self._lock:
            now = time.monotonic()
            if not force and now - self.last_write_monotonic < self.write_interval_seconds:
                return
            elapsed = now - self.started_monotonic
            calls_per_minute = (self.network_calls / elapsed * 60) if elapsed > 0 else 0.0
            eta_seconds = None
            remaining_network_calls = max(0, self.total_planned_calls - self.completed_calls)
            if self.network_calls > 0 and calls_per_minute > 0:
                eta_seconds = remaining_network_calls / calls_per_minute * 60
            effective_total_calls = (
                max(self.completed_calls, 1)
                if self.status == "completed" and self.planned_calls_are_upper_bound
                else self.total_planned_calls
            )
            self.last_write_reason = reason or self.last_write_reason
            payload = {
                "status": self.status,
                "started_at": self.started_at,
                "last_updated_at": datetime.now(timezone.utc).isoformat(),
                "last_write_reason": self.last_write_reason,
                "last_progress_event_at": self.last_progress_event_at,
                "seconds_since_last_progress_event": round(max(0.0, now - self.last_progress_event_monotonic), 2),
                "elapsed_seconds": round(elapsed, 2),
                "total_planned_calls": self.total_planned_calls,
                "completed_calls": self.completed_calls,
                "planned_calls_are_upper_bound": self.planned_calls_are_upper_bound,
                "completed_call_ratio": round(self.completed_calls / effective_total_calls, 6) if effective_total_calls else 0.0,
                "completed_call_ratio_upper_bound": round(self.completed_calls / self.total_planned_calls, 6) if self.total_planned_calls else 0.0,
                "total_planned_predictions": self.total_planned_predictions,
                "completed_predictions": self.completed_predictions,
                "completed_prediction_ratio": round(self.completed_predictions / self.total_planned_predictions, 6) if self.total_planned_predictions else 0.0,
                "cache_hits": self.cache_hits,
                "network_calls": self.network_calls,
                "observed_network_rpm": round(calls_per_minute, 2),
                "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
                "last_dataset": self.last_dataset,
                "last_method": self.last_method,
                "last_sample_id": self.last_sample_id,
            }
            self.progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
            self.last_write_monotonic = now

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            self.write(force=True, reason="heartbeat")

    def _note_progress_event_locked(self) -> None:
        self.last_progress_event_monotonic = time.monotonic()
        self.last_progress_event_at = datetime.now(timezone.utc).isoformat()


def build_run_id(*parts: str) -> str:
    """Build a stable run id with a UTC timestamp prefix."""

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_parts = [part.replace("/", "-") for part in parts if part]
    return "-".join([timestamp, *safe_parts])


RunValidator = Callable[[str | Path], dict[str, Any]]


def finalize_run_outputs(
    run_dir: str | Path,
    *,
    validator: RunValidator,
    validation_path: str | Path | None = None,
) -> dict[str, Any]:
    """Pack, validate, and optionally publish a finished run."""

    root = Path(run_dir)
    pack_run_artifacts(root)
    validation = validator(root)
    output_path = Path(validation_path) if validation_path is not None else root / "run_validation.json"
    output_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    publish_payload = publish_run_if_configured(root, validation=validation)
    if publish_payload is not None:
        validation["hf_publish"] = publish_payload
        output_path.write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
    return validation
