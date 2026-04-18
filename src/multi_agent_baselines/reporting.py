"""多智能体实验报告摘要。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取多智能体运行目录下的 `metrics.json`。"""
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总多智能体实验摘要。"""
    metrics = load_metrics(run_dir)
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }
