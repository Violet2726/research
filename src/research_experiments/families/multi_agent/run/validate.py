"""多智能体运行结果校验。

目标是快速确认一次多智能体运行是否满足继续分析的最低条件：
关键产物齐全、无请求或格式失败、题级预测非空，并且配对分析报告已经生成。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import (
    load_jsonl,
    summarize_turn_statuses,
    validate_shared_contracts,
)


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
    agent_rows = load_jsonl(root / "agent_turns.jsonl") if (root / "agent_turns.jsonl").exists() else []
    prediction_rows = load_jsonl(root / "final_predictions.jsonl") if (root / "final_predictions.jsonl").exists() else []

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": (
            not missing
            and status_summary["request_failures"] == 0
            and status_summary["schema_failures"] == 0
            and bool(prediction_rows)
            and figure_contract["passed"]
            and archive_contract["passed"]
        ),
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "paired_analysis_present": (root / "paired_debate_vs_vote.json").exists(),
        "paired_report_present": (root / "report.md").exists(),
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
    }
