"""实验运行期共享工具。"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import json
import time

from api_baselines.config import BenchmarkConfig
from api_baselines.datasets import DatasetSample, load_samples


class RunProgressTracker:
    """通用运行进度跟踪器。

    两条实验线的字段基本一致，统一抽到这里，避免 single-agent 和
    multi-agent 各自维护一份近似相同的进度写盘逻辑。
    """

    def __init__(self, progress_path: Path, total_planned_calls: int, total_planned_predictions: int) -> None:
        self.progress_path = progress_path
        self.total_planned_calls = total_planned_calls
        self.total_planned_predictions = total_planned_predictions
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.started_monotonic = time.monotonic()
        self.completed_calls = 0
        self.completed_predictions = 0
        self.cache_hits = 0
        self.network_calls = 0
        self.last_dataset: str | None = None
        self.last_method: str | None = None
        self.last_sample_id: str | None = None
        self.status = "running"
        self.last_write_monotonic = 0.0
        self.write(force=True)

    def record_call(self, row: dict[str, object], method_key: str = "method") -> None:
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
        """记录一批题级预测已写出。"""
        self.completed_predictions += count
        self.last_dataset = dataset
        self.last_method = method_name
        self.write(force=True)

    def mark_completed(self) -> None:
        """把运行状态标记为完成。"""
        self.status = "completed"
        self.write(force=True)

    def write(self, force: bool = False) -> None:
        """按节流策略刷新进度文件，避免高频磁盘写入。"""
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
    """生成带 UTC 时间戳的运行目录名。"""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    safe_parts = [part.replace("/", "-") for part in parts if part]
    return "-".join([timestamp, *safe_parts])


def load_split_ids(dataset_slug: str, split_name: str, splits_root: str | Path = "configs/benchmarks/splits") -> list[str]:
    """读取冻结后的 split manifest。"""
    payload = json.loads((Path(splits_root) / f"{dataset_slug}-{split_name}.json").read_text(encoding="utf-8"))
    return payload["sample_ids"]


def select_samples(benchmark: BenchmarkConfig, split_name: str, splits_root: str | Path = "configs/benchmarks/splits") -> list[DatasetSample]:
    """按冻结 split 选择当前 benchmark 的样本。"""
    split_ids = load_split_ids(benchmark.slug, split_name, splits_root=splits_root)
    sample_map = {sample.sample_id: sample for sample in load_samples(benchmark)}
    return [sample_map[sample_id] for sample_id in split_ids if sample_id in sample_map]
