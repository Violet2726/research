from __future__ import annotations

import io
import json
from contextlib import redirect_stdout

from multi_agent_baselines.cli import main as multi_agent_main
from budget_comm.cli import main as budget_main
from sparc.cli import main as sparc_main
from selective_comm.cli import main as selective_main
from single_agent_baselines.cli import main as single_agent_main


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
            "single-agent-cli",
            "inspect-experiment",
            "--experiment",
            "configs/single_agent/experiments/main-baselines.toml",
        ],
    )
    assert payload["name"] == "main-baselines"


def test_multi_agent_inspect_cli() -> None:
    payload = _run_cli(
        multi_agent_main,
        [
            "mad-cli",
            "inspect-experiment",
            "--experiment",
            "configs/multi_agent/experiments/vanilla_mad_minimal.toml",
        ],
    )
    assert payload["name"] == "vanilla-mad-minimal"


def test_selective_comm_inspect_cli() -> None:
    payload = _run_cli(
        selective_main,
        [
            "selective-cli",
            "inspect-experiment",
            "--experiment",
            "configs/selective_comm/experiments/trigger_early_exit_v1.toml",
        ],
    )
    assert payload["name"] == "trigger-early-exit-v1"


def test_sparc_inspect_cli() -> None:
    payload = _run_cli(
        sparc_main,
        [
            "sparc-cli",
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
            "sparc-cli",
            "inspect-experiment",
            "--experiment",
            "configs/sparc/experiments/aggregation_auditing_ablation_v1.toml",
        ],
    )
    assert payload["name"] == "aggregation_auditing_ablation_v1"
    assert payload["resolved_backbone"]["name"] == "dashscope/qwen-turbo-1101"
    assert payload["max_concurrent_requests"] == 8
    assert payload["requests_per_minute_limit"] == 600
    assert payload["tokens_per_minute_limit"] == 1000000
    assert payload["aggregation_methods"] == [
        "majority_vote",
        "single_judge",
        "final_round_vote",
        "local_auditing",
    ]


def test_budget_comm_inspect_cli() -> None:
    payload = _run_cli(
        budget_main,
        [
            "budget-cli",
            "inspect-experiment",
            "--experiment",
            "configs/budget_comm/experiments/dala_lite_same_context_v1.toml",
        ],
    )
    assert payload["name"] == "dala_lite_same_context_v1"
    assert payload["context_view"]["track_name"] == "same_context"
    assert payload["resolved_backbone"]["name"] == "dashscope/qwen-turbo-1101"
