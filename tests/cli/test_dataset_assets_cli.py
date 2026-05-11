"""覆盖 `research_cli tools dataset-assets` 的基础命令路径。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from research_experiments.cli import main as research_main


def _run_cli(argv: list[str]) -> dict[str, object]:
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        research_main(argv[1:])
    return json.loads(buffer.getvalue())


def test_dataset_assets_list_used_cli() -> None:
    payload = _run_cli(["research_cli", "tools", "dataset-assets", "list-used"])
    assert payload["benchmark_count"] == 7
    assert {item["slug"] for item in payload["benchmarks"]} == {
        "gpqa_diamond",
        "gsm8k",
        "gsm_symbolic",
        "hotpotqa",
        "math500",
        "mmlu_pro",
        "strategyqa",
    }
    assert any(item["asset_id"] == "train" for item in payload["supplementary_assets"])
