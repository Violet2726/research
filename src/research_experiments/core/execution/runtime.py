"""长时间实验运行的进度、收尾与归档工具。"""

from __future__ import annotations

import time
from collections import deque
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from threading import Event, RLock, Thread
from typing import Any

from research_experiments.core.io import write_json
from research_experiments.workspace.run_archives import pack_run_artifacts, publish_run_if_configured


class RunProgressTracker:
    """持续写出实时进度快照。"""

    def __init__(
        self,
        progress_path: Path,
        total_planned_calls: int,
        total_planned_predictions: int,
        *,
        initial_completed_calls: int = 0,
        initial_completed_predictions: int = 0,
        planned_calls_are_upper_bound: bool = False,
        target_network_rpm: int | None = None,
        rate_limit_snapshot_provider: Callable[[], dict[str, Any]] | None = None,
        network_rpm_window_seconds: float = 60.0,
        write_interval_seconds: float = 5.0,
        heartbeat_interval_seconds: float = 3.0,
    ) -> None:
        self.progress_path = progress_path
        self.total_planned_calls = total_planned_calls
        self.total_planned_predictions = total_planned_predictions
        self.target_network_rpm = target_network_rpm
        self.rate_limit_snapshot_provider = rate_limit_snapshot_provider
        self.network_rpm_window_seconds = max(0.01, float(network_rpm_window_seconds))
        self.started_at = datetime.now(UTC).isoformat()
        self.started_monotonic = time.monotonic()
        self.completed_calls = initial_completed_calls
        self.completed_predictions = initial_completed_predictions
        self.planned_calls_are_upper_bound = planned_calls_are_upper_bound
        self.cache_hits = 0
        self.network_calls = 0
        self._network_call_events: deque[float] = deque()
        self._call_events: deque[float] = deque()
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
        """记录一次完成的底层模型调用。"""

        with self._lock:
            now = time.monotonic()
            self.completed_calls += 1
            self._call_events.append(now)
            if row.get("cache_hit"):
                self.cache_hits += 1
            else:
                self.network_calls += 1
                self._network_call_events.append(now)
            self.last_dataset = str(row.get("dataset") or "")
            self.last_method = str(row.get(method_key) or "")
            self.last_sample_id = str(row.get("sample_id") or "")
            self._note_progress_event_locked(now)
            force = self.completed_calls % 10 == 0
        self.write(force=force, reason="call")

    def record_predictions(self, count: int, dataset: str, method_name: str) -> None:
        """记录题级预测已经落盘。"""

        with self._lock:
            now = time.monotonic()
            self.completed_predictions += count
            self.last_dataset = dataset
            self.last_method = method_name
            self._note_progress_event_locked(now)
        self.write(force=True, reason="prediction")

    def mark_completed(self) -> None:
        """标记 run 完成并停止后台心跳。"""

        with self._lock:
            self.status = "completed"
            self._note_progress_event_locked(time.monotonic())
        self.write(force=True, reason="completed")
        self.close()

    def close(self) -> None:
        """停止后台心跳线程，避免异常路径持续改写 progress.json。"""

        self._stop_event.set()
        if self._heartbeat_thread.is_alive():
            self._heartbeat_thread.join(timeout=0.2)

    def write(self, force: bool = False, *, reason: str | None = None) -> None:
        """刷新磁盘上的进度快照。"""

        with self._lock:
            now = time.monotonic()
            if not force and now - self.last_write_monotonic < self.write_interval_seconds:
                return
            self._evict_metric_events(now)
            elapsed = now - self.started_monotonic
            lifetime_network_rpm = (self.network_calls / elapsed * 60) if elapsed > 0 else 0.0
            observed_network_rpm = len(self._network_call_events) / self.network_rpm_window_seconds * 60
            observed_call_rpm = len(self._call_events) / self.network_rpm_window_seconds * 60
            eta_rpm = observed_network_rpm or lifetime_network_rpm
            eta_seconds = None
            remaining_calls = max(0, self.total_planned_calls - self.completed_calls)
            if eta_rpm > 0:
                eta_seconds = remaining_calls / eta_rpm * 60
            effective_total_calls = (
                max(self.completed_calls, 1)
                if self.status == "completed" and self.planned_calls_are_upper_bound
                else self.total_planned_calls
            )
            self.last_write_reason = reason or self.last_write_reason
            payload: dict[str, Any] = {
                "status": self.status,
                "started_at": self.started_at,
                "last_updated_at": datetime.now(UTC).isoformat(),
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
                "cache_hit_ratio": round(self.cache_hits / self.completed_calls, 6) if self.completed_calls else 0.0,
                "network_rpm_window_seconds": self.network_rpm_window_seconds,
                "observed_network_rpm": round(observed_network_rpm, 2),
                "lifetime_network_rpm": round(lifetime_network_rpm, 2),
                "observed_call_rpm": round(observed_call_rpm, 2),
                "target_network_rpm": self.target_network_rpm,
                "eta_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
                "last_dataset": self.last_dataset,
                "last_method": self.last_method,
                "last_sample_id": self.last_sample_id,
            }
            if self.rate_limit_snapshot_provider is not None:
                payload.update(self.rate_limit_snapshot_provider())
            write_json(self.progress_path, payload)
            self.last_write_monotonic = now

    def _heartbeat_loop(self) -> None:
        while not self._stop_event.wait(self.heartbeat_interval_seconds):
            self.write(force=True, reason="heartbeat")

    def _note_progress_event_locked(self, now: float) -> None:
        self.last_progress_event_monotonic = now
        self.last_progress_event_at = datetime.now(UTC).isoformat()

    def _evict_metric_events(self, now: float) -> None:
        cutoff = now - self.network_rpm_window_seconds
        while self._network_call_events and self._network_call_events[0] < cutoff:
            self._network_call_events.popleft()
        while self._call_events and self._call_events[0] < cutoff:
            self._call_events.popleft()


def build_run_id(*parts: str) -> str:
    """生成带 UTC 时间戳前缀的稳定 run id。"""

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    safe_parts = [part.replace("/", "-") for part in parts if part]
    return "-".join([timestamp, *safe_parts])


RunValidator = Callable[[str | Path], dict[str, Any]]


def finalize_run_outputs(
    run_dir: str | Path,
    *,
    validator: RunValidator,
    validation_path: str | Path | None = None,
) -> dict[str, Any]:
    """打包、校验并按配置发布已完成 run。"""

    root = Path(run_dir)
    pack_run_artifacts(root)
    validation = validator(root)
    output_path = Path(validation_path) if validation_path is not None else root / "run_validation.json"
    write_json(output_path, validation)
    publish_payload = publish_run_if_configured(root, validation=validation)
    if publish_payload is not None:
        validation["hf_publish"] = publish_payload
        write_json(output_path, validation)
    return validation
