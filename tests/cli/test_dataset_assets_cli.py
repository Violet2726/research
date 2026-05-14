"""覆盖 `research_cli tools dataset-assets` 的基础命令路径。"""

from __future__ import annotations

from testsupport.cli import run_cli_json


def test_dataset_assets_list_used_cli() -> None:
    payload = run_cli_json(["research_cli", "tools", "dataset-assets", "list-used"])
    assert payload["benchmark_count"] == 16
    assert {item["slug"] for item in payload["benchmarks"]} == {
        "dog_cwq",
        "dog_grailqa",
        "dog_metaqa_1hop",
        "dog_metaqa_2hop",
        "dog_metaqa_3hop",
        "dog_webqsp",
        "dog_webquestions",
        "grailqa",
        "gpqa_diamond",
        "gsm8k",
        "gsm_symbolic",
        "hotpotqa",
        "math500",
        "mmlu_pro",
        "strategyqa",
        "webquestions",
    }
    assert any(item["asset_id"] == "train" for item in payload["supplementary_assets"])
