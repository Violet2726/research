"""覆盖各实验 CLI 基本命令路径的轻量 smoke 测试。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from experiment_core.cache import CachedResponse, RequestCacheRouter, json_dump
from experiment_core.cache_inspector import main as cache_inspector_main
from experiment_core.faithful_matrix import main as faithful_matrix_main
from multi_agent.cli import main as multi_agent_main
from budget_comm.cli import main as budget_main
from comm_necessary.cli import main as comm_necessary_main
from free_mad_lite.cli import main as free_mad_main
from sparc.cli import main as sparc_main
from selective_comm.cli import main as selective_main
from sid_lite.cli import main as sid_lite_main
from single_agent.cli import main as single_agent_main


def _run_cli(main_func, argv: list[str]) -> dict[str, object]:
    import sys

    previous = sys.argv
    buffer = io.StringIO()
    try:
        sys.argv = argv
        with redirect_stdout(buffer):
            main_func()
    finally:
        sys.argv = previous
    return json.loads(buffer.getvalue())


def test_single_agent_inspect_cli() -> None:
    payload = _run_cli(
        single_agent_main,
        [
            "single_agent_cli",
            "inspect-experiment",
            "--experiment",
            "configs/single_agent/experiments/same_context_core_benchmarks.toml",
        ],
    )
    assert payload["name"] == "same_context_core_benchmarks"
    assert payload["workspace_defaults"]["experiment_cache_root"].endswith("cache")


def test_faithful_matrix_inspect_cli() -> None:
    payload = _run_cli(
        faithful_matrix_main,
        [
            "faithful_matrix_cli",
            "inspect-matrix",
            "--phase",
            "smoke20",
        ],
    )
    assert payload["overrides"]["phase_name"] == "smoke20"
    assert payload["counts"]["semantic_unique_targets"] == 15


def test_multi_agent_inspect_cli() -> None:
    payload = _run_cli(
        multi_agent_main,
        [
            "multi_agent_cli",
            "inspect-experiment",
            "--experiment",
            "configs/multi_agent/experiments/same_context_controlled_debate.toml",
        ],
    )
    assert payload["name"] == "same_context_controlled_debate"


def test_selective_comm_inspect_cli() -> None:
    payload = _run_cli(
        selective_main,
        [
            "selective_comm_cli",
            "inspect-experiment",
            "--experiment",
            "configs/selective_comm/experiments/trigger_early_exit_main.toml",
        ],
    )
    assert payload["name"] == "trigger_early_exit_main"
    assert payload["workspace_defaults"]["experiment_runs_root"].endswith("selective_comm")


def test_selective_comm_voc_v2_inspect_cli() -> None:
    payload = _run_cli(
        selective_main,
        [
            "selective_comm_cli",
            "inspect-experiment",
            "--experiment",
            "configs/selective_comm/experiments/voc_trigger_main.toml",
        ],
    )
    assert payload["name"] == "voc_trigger_main"
    assert payload["prompt_version"] == "selective_comm_voc_json_v2"
    assert len(payload["policies"]) == 5
    assert payload["policies"][-1]["policy_name"] == "voc_trigger_v2"
    assert payload["policies"][-1]["claim_divergence_threshold"] == 0.55
    assert payload["policies"][-1]["uncertainty_type_diversity_threshold"] == 0.5
    assert payload["model_fit_warnings"] == []


def test_sparc_inspect_cli() -> None:
    payload = _run_cli(
        sparc_main,
        [
            "sparc_cli",
            "inspect-experiment",
            "--experiment",
            "configs/sparc/experiments/content_ablation.toml",
        ],
    )
    assert payload["name"] == "content_ablation"


def test_sparc_local_auditing_inspect_cli() -> None:
    payload = _run_cli(
        sparc_main,
        [
            "sparc_cli",
            "inspect-experiment",
            "--experiment",
            "configs/sparc/experiments/local_auditing_ablation.toml",
        ],
    )
    assert payload["name"] == "local_auditing_ablation"
    assert payload["resolved_model"]["name"] == "deepseek/deepseek-v4-flash"
    assert payload["max_concurrent_requests"] == 5
    assert payload["requests_per_minute_limit"] == 50
    assert payload["tokens_per_minute_limit"] == 1000000
    assert payload["aggregation_methods"] == [
        "majority_vote",
        "weighted_vote_fallback",
        "single_judge",
        "final_round_vote",
        "local_auditing",
    ]


def test_budget_comm_inspect_cli() -> None:
    payload = _run_cli(
        budget_main,
        [
            "budget_comm_cli",
            "inspect-experiment",
            "--experiment",
            "configs/budget_comm/experiments/dala_lite_same_context_main.toml",
        ],
    )
    assert payload["name"] == "dala_lite_same_context_main"
    assert payload["context_view"]["track_name"] == "same_context"
    assert payload["resolved_model"]["name"] == "deepseek/deepseek-v4-flash"


def test_sid_lite_inspect_cli() -> None:
    payload = _run_cli(
        sid_lite_main,
        [
            "sid_lite_cli",
            "inspect-experiment",
            "--experiment",
            "configs/sid_lite/experiments/sid_lite_mechanism_validation.toml",
        ],
    )
    assert payload["name"] == "sid_lite_mechanism_validation"
    assert payload["methods"] == ["mv_3", "always_full", "compression_only", "sid_lite"]
    assert payload["max_concurrent_requests"] == 5
    assert payload["requests_per_minute_limit"] == 50
    assert payload["tokens_per_minute_limit"] == 1000000


def test_free_mad_lite_inspect_cli() -> None:
    payload = _run_cli(
        free_mad_main,
        [
            "free_mad_lite_cli",
            "inspect-experiment",
            "--experiment",
            "configs/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml",
        ],
    )
    assert payload["name"] == "free_mad_lite_mechanism_validation"
    assert payload["methods"] == [
        "mv_3_initial",
        "vanilla_mad_r1_final_vote",
        "anti_conformity_final_vote",
        "free_mad_lite_llm_trajectory",
    ]
    assert payload["protocol"]["debate_rounds"] == 1
    assert payload["anti_conformity_prompt_hash"]


def test_comm_necessary_inspect_cli() -> None:
    payload = _run_cli(
        comm_necessary_main,
        [
            "comm_necessary_cli",
            "inspect-experiment",
            "--experiment",
            "configs/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml",
        ],
    )
    assert payload["name"] == "hotpotqa_split_context_communication_necessity"
    assert payload["methods"] == [
        "full_context_single",
        "split_no_comm_mv3",
        "answer_only_exchange",
        "evidence_exchange",
        "full_packet_exchange",
    ]
    assert payload["max_concurrent_requests"] == 5
    assert payload["requests_per_minute_limit"] == 50
    assert payload["tokens_per_minute_limit"] == 1000000


def test_cache_inspector_summarize_cli(tmp_path) -> None:
    router = RequestCacheRouter(tmp_path)
    cache = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    cache.put(
        CachedResponse(
            cache_key="a",
            payload_json=json_dump({"request": 1}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_a",
        )
    )
    router.close()

    payload = _run_cli(
        cache_inspector_main,
        [
            "cache_inspector_cli",
            "summarize",
            "--cache-root",
            str(tmp_path),
            "--top-shards",
            "5",
            "--json",
        ],
    )
    assert payload["shard_count"] == 1
    assert payload["total_request_count"] == 1
    assert payload["providers"][0]["provider"] == "deepseek"
    assert payload["providers"][0]["model_count"] == 1
    assert payload["providers"][0]["dataset_count"] == 1
