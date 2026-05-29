"""覆盖 `research_cli tools dataset-assets` 的基础命令路径。"""

from __future__ import annotations

from testsupport.cli import run_cli_json


def test_dataset_assets_list_used_cli() -> None:
    payload = run_cli_json(["research_cli", "tools", "dataset-assets", "list-used"])
    assert payload["benchmark_count"] == 15
    assert {item["slug"] for item in payload["benchmarks"]} == {
        "commongen_hard",
        "competition_math",
        "gpqa_diamond",
        "gsm8k",
        "hotpotqa",
        "humaneval",
        "math500",
        "mmlu",
        "mmlu_abstract_algebra",
        "mmlu_pro",
        "realmistake_answerability_classification",
        "realmistake_fine_grained_fact_verification",
        "realmistake_math_problem_generation",
        "strategyqa",
        "webquestions",
    }
    assert any(item["asset_id"] == "train" for item in payload["supplementary_assets"])
