"""CLI 测试共享辅助函数。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from typing import Any

from research_experiments.cli import main as research_main


def run_cli_json(argv: list[str]) -> dict[str, Any]:
    """执行统一 CLI 并把标准输出解析为 JSON。"""

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        research_main(argv[1:])
    return json.loads(buffer.getvalue())
