"""多智能体运行结果校验。

这里的目标是快速确认一次多智能体运行是否具备继续分析的最小条件：
关键文件齐全、turn 日志没有请求/格式失败、题级预测不为空，
以及配对分析报告是否已经产出。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.run_archives import validate_archive_contract
from experiment_core.reporting.run_figures import validate_figure_contract


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查多智能体运行目录中的关键产物是否齐全且基本可用。"""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "agent_turns.jsonl",
        "debate_messages.jsonl",
        "final_predictions.jsonl",
        "metrics.json",
        "cost_breakdown.json",
        "debate_diagnostics.json",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing = [name for name in required if not (root / name).exists()]
    agent_rows = _load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []
    request_failures = sum(1 for row in agent_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in agent_rows if row.get("output_status") == "schema_fail")
    methods = Counter(row.get("method_name") for row in prediction_rows)
    figure_contract = validate_figure_contract(root)
    archive_contract = validate_archive_contract(root)
    return {
        "run_dir": str(root),
        "passed": not missing and request_failures == 0 and schema_failures == 0 and bool(prediction_rows) and figure_contract["passed"] and archive_contract["passed"],
        "missing_files": missing,
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "paired_analysis_present": (root / "paired_debate_vs_vote.json").exists(),
        "paired_report_present": (root / "report.md").exists(),
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
