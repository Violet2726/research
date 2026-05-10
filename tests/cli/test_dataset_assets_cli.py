"""覆盖 dataset_assets_cli 的基础命令路径。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from experiment_core.tools.dataset_assets import main as dataset_assets_main


def _run_cli(argv: list[str]) -> dict[str, object]:
    import sys

    previous = sys.argv
    buffer = io.StringIO()
    try:
        sys.argv = argv
        with redirect_stdout(buffer):
            dataset_assets_main()
    finally:
        sys.argv = previous
    return json.loads(buffer.getvalue())


def test_dataset_assets_list_used_cli() -> None:
    payload = _run_cli(["dataset_assets_cli", "list-used"])
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
