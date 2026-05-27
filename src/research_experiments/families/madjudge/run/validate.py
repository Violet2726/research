"""MADJudge 运行验证。"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """验证 MADJudge 运行的完整性。"""
    root = Path(run_dir)
    issues: list[str] = []

    required_files = [
        "manifest.json",
        "metrics.json",
        "predictions.jsonl",
        "turns.jsonl",
    ]

    for fname in required_files:
        if not (root / fname).exists():
            issues.append(f"Missing required file: {fname}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
    }
