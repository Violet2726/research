"""family 运行校验的共享工具。"""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

from research_experiments.workspace.run_archives import validate_archive_contract
from research_experiments.reporting.run_figures import validate_figure_contract


def validate_shared_contracts(run_dir: str | Path) -> dict[str, Any]:
    """统一校验 figure 与 archive 两类共享合同。"""

    root = Path(run_dir)
    return {
        "figure_contract": validate_figure_contract(root),
        "archive_contract": validate_archive_contract(root),
    }


def load_json(path: str | Path) -> dict[str, Any]:
    """读取一个 UTF-8 JSON 文件。"""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取一个 UTF-8 JSONL 文件。"""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
