"""运行时辅助工具。

本模块放置各实验线都会复用的轻量运行时原语，例如进度快照、ETA 估计，
以及带 UTC 时间戳的稳定 `run_id` 生成逻辑。
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time
from typing import Any, Callable

from research_experiments.core.foundation.run_archives import pack_run_artifacts, publish_run_if_configured


class RunProgressTracker:
    """把长时间实验的进度持续写入磁盘。"""

    def __init__(
        self,
        progress_path: Path,
        total_planned_calls: int,
        total_planned_predictions: int,
        *,
        initial_completed_calls: int = 0,
        initial_completed_predictions: int = 0,
    ) -> None:
        self.progress_path = progress_path
        self.total_planned_calls = total_planned_calls
        self.total_planned_predictions = total_planned_predictions
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.completed_calls = initial_completed_calls
        self.completed_predictions = initial_completed_predictions
        self.cache_hits = 0
        self.network_calls = 0
        self.last_dataset: str | None = None
        self.last_method: str | None = None
        self.last_sample_id: str | None = None
        self.status = "running"
        self.last_write_monotonic = 0.0
        self.write(force=True)

    def record_call(self, row: dict[str, object], method_key: str = "method_name") -> None:
        """记录一次底层调用完成。"""
        self.completed_calls += 1
        if row.get("cache_hit"):
            self.cache_hits += 1
        else:
            self.network_calls += 1
        self.last_dataset = str(row.get("dataset") or "")
        self.last_method = str(row.get(method_key) or "")
        self.last_sample_id = str(row.get("sample_id") or "")
        self.write(force=self.completed_calls % 10 == 0)

    def record_predictions(self, count: int, dataset: str, method_name: str) -> None:
        """记录题级预测产物已经落盘。"""
        self.completed_predictions += count
        self.last_dataset = dataset
        self.last_method = method_name
        self.write(force=True)

    def mark_completed(self) -> None:
        """将运行状态标记为完成。"""
        self.status = "completed"
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        """按节流策略刷新当前进度快照。"""
        now = time.monotonic()
        if not force and now - self.last_write_monotonic < 5:
            return
        elapsed = now - self.started_monotonic
        calls_per_minute = (self.network_calls / elapsed * 60) if elapsed > 0 else 0.0
        eta_seconds = None
        remaining_network_calls = max(0, self.total_planned_calls - self.completed_calls)
        if self.network_calls > 0 and calls_per_minute > 0:
            eta_seconds = remaining_network_calls / calls_per_minute * 60
        payload = {
            "status": self.status,
            "started_at": self.started_at,
            "last_updated_at": datetime.now(timezone.utc).isoformat(),
            "elapsed_seconds": round(elapsed, 2),
            "total_planned_calls": self.total_planned_calls,
            "completed_calls": self.completed_calls,
            "completed_call_ratio": round(self.completed_calls / self.total_planned_calls, 6) if self.total_planned_calls else 0.0,
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


def build_run_id(*parts: str) -> str:
    """生成包含 UTC 时间戳的稳定运行 ID。"""
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
    """统一完成 run 级归档与校验落盘。"""
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
