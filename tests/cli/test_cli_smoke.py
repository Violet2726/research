"""覆盖各实验 CLI 基本命令路径的轻量 smoke 测试。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

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
            "configs/single_agent/experiments/main_baselines.toml",
        ],
    )
    assert payload["name"] == "main_baselines"
    assert payload["workspace_defaults"]["experiment_cache_path"].endswith("single_agent_requests.sqlite")


def test_multi_agent_inspect_cli() -> None:
    payload = _run_cli(
        multi_agent_main,
        [
            "multi_agent_cli",
            "inspect-experiment",
            "--experiment",
            "configs/multi_agent/experiments/vanilla_mad_minimal.toml",
        ],
    )
    assert payload["name"] == "vanilla_mad_minimal"


def test_selective_comm_inspect_cli() -> None:
    payload = _run_cli(
        selective_main,
        [
            "selective_comm_cli",
            "inspect-experiment",
            "--experiment",
            "configs/selective_comm/experiments/trigger_early_exit_v1.toml",
        ],
    )
    assert payload["name"] == "trigger_early_exit_v1"
    assert payload["workspace_defaults"]["experiment_runs_root"].endswith("selective_comm")


def test_selective_comm_voc_v2_inspect_cli() -> None:
    payload = _run_cli(
        selective_main,
        [
            "selective_comm_cli",
            "inspect-experiment",
            "--experiment",
            "configs/selective_comm/experiments/trigger_voc_v2.toml",
        ],
    )
    assert payload["name"] == "trigger_voc_v2"
    assert payload["prompt_version"] == "selective_comm_voc_json_v2"
    assert len(payload["policies"]) == 8
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
            "configs/sparc/experiments/content_ablation_v1.toml",
        ],
    )
    assert payload["name"] == "content_ablation_v1"


def test_sparc_aggregation_auditing_inspect_cli() -> None:
    payload = _run_cli(
        sparc_main,
        [
            "sparc_cli",
            "inspect-experiment",
            "--experiment",
            "configs/sparc/experiments/aggregation_auditing_ablation_v1.toml",
        ],
    )
    assert payload["name"] == "aggregation_auditing_ablation_v1"
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
            "configs/budget_comm/experiments/dala_lite_same_context_v1.toml",
        ],
    )
    assert payload["name"] == "dala_lite_same_context_v1"
    assert payload["context_view"]["track_name"] == "same_context"
    assert payload["resolved_model"]["name"] == "deepseek/deepseek-v4-flash"


def test_sid_lite_inspect_cli() -> None:
    payload = _run_cli(
        sid_lite_main,
        [
            "sid_lite_cli",
            "inspect-experiment",
            "--experiment",
            "configs/sid_lite/experiments/sid_lite_v1.toml",
        ],
    )
    assert payload["name"] == "sid_lite_v1"
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
            "configs/free_mad_lite/experiments/free_mad_lite_v1.toml",
        ],
    )
    assert payload["name"] == "free_mad_lite_v1"
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
            "configs/comm_necessary/experiments/hotpotqa_split_evidence_v1.toml",
        ],
    )
    assert payload["name"] == "hotpotqa_split_evidence_v1"
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
