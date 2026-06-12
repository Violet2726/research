"""多智能体运行结果校验。

目标是快速确认一次多智能体运行是否满足继续分析的最低条件：
关键产物齐全、无请求或格式失败、题级预测非空，并且配对分析报告已经生成。
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """检查多智能体运行目录中的关键产物是否齐全且基本可用。"""
    index = resolve_run_artifact_index(run_dir, family_name="multi_agent")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="multi_agent")
    diagnostic_paths = named_diagnostic_paths(root, family_name="multi_agent")
    required_paths = [
        index.manifest_path,
        turn_paths["agent_turns.jsonl"],
        turn_paths["debate_messages.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["cost_breakdown.json"],
        diagnostic_paths["debate_diagnostics.json"],
        diagnostic_paths["answer_contract_diagnostics.json"],
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = missing_relative_paths(root, required_paths)
    manifest = load_json(index.manifest_path)
    agent_rows = load_jsonl(turn_paths["agent_turns.jsonl"]) if turn_paths["agent_turns.jsonl"].exists() else []
    prediction_rows = load_jsonl(index.prediction_records_path) if index.prediction_records_path.exists() else []
    answer_contract_diagnostics = load_json(diagnostic_paths["answer_contract_diagnostics.json"])

    status_summary = summarize_turn_statuses(agent_rows)
    methods = Counter(row.get("method_name") for row in prediction_rows)
    rate_limit_check = validate_rate_limit_check(index.progress_path, agent_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    return {
        "run_dir": str(root),
        "passed": (
            not missing
            and status_summary["request_failures"] == 0
            and status_summary["answer_contract_failures"] == 0
            and bool(prediction_rows)
            and rate_limit_check["passed"]
            and figure_contract["passed"]
            and archive_contract["passed"]
        ),
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "answer_contract_failures": status_summary["answer_contract_failures"],
        "prediction_rows": len(prediction_rows),
        "methods": dict(methods),
        "paired_analysis_present": (root / "paired_debate_vs_vote.json").exists() or (root / "exports" / "paired_debate_vs_vote.json").exists(),
        "paired_report_present": (root / "report.md").exists(),
        "rate_limit_check": rate_limit_check,
        "figure_contract": figure_contract,
        "archive_contract": archive_contract,
        "answer_contract_diagnostics": answer_contract_diagnostics,
    }

