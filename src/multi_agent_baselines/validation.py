"""多智能体运行结果校验。"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查多智能体运行目录的关键产物是否齐全。"""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "cost_breakdown.json",
        "debate_diagnostics.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = _load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in agent_rows if row.get("parse_status") == "request_fail")
    methods = Counter(row.get("method_name") for row in prediction_rows)
    return {
        "run_dir": str(root),
        "passed": not missing and request_failures == 0 and bool(prediction_rows),
        "missing_files": missing,
        "request_failures": request_failures,
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 编码的 JSONL。"""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
