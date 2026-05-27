"""CONSENSAGENT 实验的验证模块。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """验证一个 CONSENSAGENT 运行的完整性。"""
    run_root = Path(run_dir)
    issues: list[str] = []
    warnings: list[str] = []

    # 检查必要文件是否存在
    required_files = [
        "turns.jsonl",
        "predictions.jsonl",
        "metrics.json",
    ]
    for filename in required_files:
        filepath = run_root / filename
        if not filepath.exists():
            issues.append(f"Missing required file: {filename}")
        elif filepath.stat().st_size == 0:
            issues.append(f"Empty file: {filename}")

    # 检查可选文件
    optional_files = [
        "debate_messages.jsonl",
        "cost_breakdown.json",
        "debate_diagnostics.json",
    ]
    for filename in optional_files:
        filepath = run_root / filename
        if not filepath.exists():
            warnings.append(f"Missing optional file: {filename}")

    # 检查 predictions.jsonl 的内容
    predictions_path = run_root / "predictions.jsonl"
    if predictions_path.exists():
        try:
            predictions = _load_jsonl(predictions_path)
            if not predictions:
                issues.append("predictions.jsonl is empty")
            else:
                # 检查必要字段
                required_fields = [
                    "run_id", "dataset", "sample_id", "method_name",
                    "prediction", "gold", "score",
                ]
                for i, pred in enumerate(predictions[:5]):  # 只检查前5条
                    for field in required_fields:
                        if field not in pred:
                            issues.append(f"Prediction {i} missing field: {field}")

                # 检查触发机制字段
                trigger_fields = ["trigger_type", "trigger_round", "sycophancy_rate"]
                for i, pred in enumerate(predictions[:5]):
                    for field in trigger_fields:
                        if field not in pred:
                            warnings.append(f"Prediction {i} missing trigger field: {field}")
        except Exception as e:
            issues.append(f"Failed to parse predictions.jsonl: {e}")

    # 检查 metrics.json 的内容
    metrics_path = run_root / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if "summary" not in metrics:
                issues.append("metrics.json missing 'summary' field")
            else:
                summary = metrics["summary"]
                if not isinstance(summary, list):
                    issues.append("metrics.summary is not a list")
                elif len(summary) == 0:
                    issues.append("metrics.summary is empty")
        except Exception as e:
            issues.append(f"Failed to parse metrics.json: {e}")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "checked_files": required_files + optional_files,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """加载 JSONL 文件。"""
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records
