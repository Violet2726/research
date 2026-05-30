"""family 运行校验的共享辅助。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_experiments.reporting.run_figures import validate_figure_contract
from research_experiments.workspace.run_archives import validate_archive_contract


def summarize_turn_statuses(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """统计 turn 记录中的 `output_status` 分布。"""
    request_failures = sum(1 for row in turn_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in turn_rows if row.get("output_status") == "schema_fail")
    ok_count = sum(1 for row in turn_rows if row.get("output_status") == "ok")
    total = len(turn_rows)
    return {
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "ok_count": ok_count,
        "total_turns": total,
        "output_success_rate": round(ok_count / total, 4) if total else 0.0,
    }


def validate_shared_contracts(run_dir: str | Path) -> dict[str, Any]:
    """统一执行 figure 与 archive 合同校验。"""

    root = Path(run_dir)
    return {
        "figure_contract": validate_figure_contract(root),
        "archive_contract": validate_archive_contract(root),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON 文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
