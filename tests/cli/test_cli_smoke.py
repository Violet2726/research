"""覆盖各实验 CLI 基本命令路径的轻量 smoke 测试。"""

from __future__ import annotations

import io
import json
from contextlib import redirect_stdout
from pathlib import Path

import pytest
from testsupport.cli import run_cli_json
from testsupport.filesystem import write_json

from research_experiments.cli import main as research_main
from research_experiments.core.execution.cache import CachedResponse, RequestCacheRouter, json_dump
from research_experiments.core.execution.rate_limits import (
    STANDARD_MAX_CONCURRENT_REQUESTS,
    STANDARD_REQUESTS_PER_MINUTE_LIMIT,
    STANDARD_TOKENS_PER_MINUTE_LIMIT,
)


def test_single_agent_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "single_agent",
            "inspect-experiment",
            "--experiment",
            "configs/families/single_agent/experiments/same_context_core_benchmarks.toml",
        ],
    )
    assert payload["name"] == "same_context_core_benchmarks"
    assert payload["workspace_defaults"]["family_cache_root"].endswith("cache")


def test_faithful_matrix_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "matrix",
            "inspect-matrix",
            "--phase",
            "count20",
        ],
    )
    assert payload["overrides"]["phase_name"] == "count20"
    assert payload["counts"]["semantic_unique_targets"] == 14


def test_reproduction_matrix_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "matrix",
            "inspect-matrix",
            "--matrix",
            "reproduction",
            "--phase",
            "count20",
        ],
    )
    assert payload["matrix_id"] == "reproduction"
    assert payload["matrix_kind"] == "reproduction_matrix"
    assert payload["counts"]["semantic_unique_targets"] == 4


def test_matrix_assert_success_cli(tmp_path: Path) -> None:
    write_json(
        tmp_path / "state.json",
        {
            "matrix_id": "faithful",
            "matrix_kind": "faithful_matrix",
            "overrides": {
                "phase_name": "count20",
                "model_ref": "xiaomimimo/mimo-v2.5",
                "max_concurrent_requests": STANDARD_MAX_CONCURRENT_REQUESTS,
                "requests_per_minute_limit": STANDARD_REQUESTS_PER_MINUTE_LIMIT,
                "tokens_per_minute_limit": STANDARD_TOKENS_PER_MINUTE_LIMIT,
            },
            "counts": {"completed": 0, "semantic_unique_targets": 0},
            "entries": [],
            "semantic_entries": [],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "matrix",
            "assert-success",
            "--state-path",
            str(tmp_path),
            "--json",
        ],
    )
    assert payload["passed"] is True
    assert payload["blocking_entries"] == []


def test_matrix_assert_success_cli_exit_code_for_blockers(tmp_path: Path) -> None:
    write_json(
        tmp_path / "state.json",
        {
            "matrix_id": "faithful",
            "matrix_kind": "faithful_matrix",
            "overrides": {
                "phase_name": "count20",
                "model_ref": "xiaomimimo/mimo-v2.5",
                "max_concurrent_requests": STANDARD_MAX_CONCURRENT_REQUESTS,
                "requests_per_minute_limit": STANDARD_REQUESTS_PER_MINUTE_LIMIT,
                "tokens_per_minute_limit": STANDARD_TOKENS_PER_MINUTE_LIMIT,
            },
            "counts": {"failed": 1, "semantic_unique_targets": 1},
            "entries": [
                {
                    "family": "selective_comm",
                    "config_path": "configs/families/selective_comm/experiments/trigger_early_exit_main.toml",
                    "experiment_name": "trigger_early_exit_main",
                    "description": "demo",
                    "phase_name": "count20",
                    "status": "failed",
                    "review_notes": "runner_error:demo",
                }
            ],
            "semantic_entries": [
                {
                    "family": "selective_comm",
                    "config_path": "configs/families/selective_comm/experiments/trigger_early_exit_main.toml",
                    "experiment_name": "trigger_early_exit_main",
                    "description": "demo",
                    "phase_name": "count20",
                    "status": "failed",
                    "review_notes": "runner_error:demo",
                }
            ],
        },
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer), pytest.raises(SystemExit) as exc_info:
        research_main(
            [
                "matrix",
                "assert-success",
                "--state-path",
                str(tmp_path),
                "--json",
            ]
        )

    assert exc_info.value.code == 1
    payload = json.loads(buffer.getvalue())
    assert payload["passed"] is False
    assert payload["blocking_entries"][0]["family"] == "selective_comm"


def test_imad_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "imad",
            "inspect-experiment",
            "--experiment",
            "configs/families/imad/experiments/imad_same_context_main.toml",
        ],
    )
    assert payload["name"] == "imad_same_context_main"
    assert payload["protocol"]["max_rounds"] == 3
    assert payload["methods"][-1]["name"] == "imad_adaptive"


def test_dmad_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "dmad",
            "inspect-experiment",
            "--experiment",
            "configs/families/dmad/experiments/dmad_reasoning_main.toml",
        ],
    )
    assert payload["name"] == "dmad_reasoning_main"
    assert payload["evaluation_scope"] == "paper_main"
    assert payload["protocol"]["agent_count"] == 3
    assert payload["methods"][-1]["name"] == "dmad_cot_sbp_pot"
    assert payload["methods"][-1]["roster_config"]["diversity_mode"] == "strategy_diverse"


def test_econ_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "econ",
            "inspect-experiment",
            "--experiment",
            "configs/families/econ/experiments/econ_same_context_main.toml",
        ],
    )
    assert payload["name"] == "econ_same_context_main"
    assert payload["protocol"]["agent_count"] == 3
    assert payload["methods"][-1]["name"] == "econ_bne_main"


def test_colmad_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "colmad",
            "inspect-experiment",
            "--experiment",
            "configs/families/colmad/experiments/colmad_realmistake_main.toml",
        ],
    )
    assert payload["name"] == "colmad_realmistake_main"
    assert payload["protocol"]["max_debate_rounds"] == 1
    assert payload["methods"][-1]["name"] == "colmad_collaborative"


def test_macnet_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "macnet",
            "inspect-experiment",
            "--experiment",
            "configs/families/macnet/experiments/macnet_paper_main.toml",
        ],
    )
    assert payload["name"] == "macnet_paper_main"
    assert payload["experiment_kind"] == "paper"
    assert payload["protocol"]["default_direction_mode"] == "divergent"


def test_faithful_matrix_render_family_landscape_cli(tmp_path: Path) -> None:
    write_json(
        tmp_path / "state.json",
        {
            "overrides": {"phase_name": "count100", "model_ref": "xiaomimimo/mimo-v2.5"},
            "counts": {"completed": 1, "semantic_unique_targets": 1},
        },
    )
    write_json(
        tmp_path / "faithful_analysis.json",
        {
            "combined_overall": [
                {
                    "family": "budget_comm",
                    "experiment_name": "dala_lite_same_context_main",
                    "evaluation_track": "same_context",
                    "evidence_tier": "headline",
                    "primary_method_name": "dala_lite",
                    "faithful_score": 0.708082,
                    "delta_vs_best_no_comm": 0.110977,
                    "delta_vs_full_comm": -0.028951,
                    "total_tokens_mean": 1600.0,
                    "communication_tokens_mean": 90.0,
                    "calls_per_question_mean": 6.0,
                    "stage_ceiling_gap": 0.014475,
                }
            ]
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "matrix",
            "render-family-landscape",
            "--state-path",
            str(tmp_path),
        ],
    )

    assert Path(payload["json_path"]).exists()
    assert Path(payload["markdown_path"]).exists()


def test_multi_agent_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "multi_agent",
            "inspect-experiment",
            "--experiment",
            "configs/families/multi_agent/experiments/same_context_controlled_debate.toml",
        ],
    )
    assert payload["name"] == "same_context_controlled_debate"


def test_selective_comm_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "selective_comm",
            "inspect-experiment",
            "--experiment",
            "configs/families/selective_comm/experiments/trigger_early_exit_main.toml",
        ],
    )
    assert payload["name"] == "trigger_early_exit_main"
    assert payload["workspace_defaults"]["family_runs_root"].endswith("selective_comm")


def test_selective_comm_voc_v2_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "selective_comm",
            "inspect-experiment",
            "--experiment",
            "configs/families/selective_comm/experiments/voc_trigger_main.toml",
        ],
    )
    assert payload["name"] == "voc_trigger_main"
    assert payload["prompt_version"] == "selective_comm_voc_json_v2"
    assert len(payload["policies"]) == 5
    assert payload["policies"][-1]["policy_name"] == "voc_trigger_v2"
    assert payload["policies"][-1]["claim_divergence_threshold"] == 0.55
    assert payload["policies"][-1]["uncertainty_type_diversity_threshold"] == 0.5
    assert payload["model_fit_warnings"] == []


def test_budget_comm_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "budget_comm",
            "inspect-experiment",
            "--experiment",
            "configs/families/budget_comm/experiments/dala_lite_same_context_main.toml",
        ],
    )
    assert payload["name"] == "dala_lite_same_context_main"
    assert payload["context_view"]["track_name"] == "same_context"
    assert payload["resolved_model"]["name"] == "deepseek/deepseek-v4-flash"


def test_sid_lite_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "sid_lite",
            "inspect-experiment",
            "--experiment",
            "configs/families/sid_lite/experiments/sid_lite_mechanism_validation.toml",
        ],
    )
    assert payload["name"] == "sid_lite_mechanism_validation"
    assert payload["methods"] == ["mv_3", "always_full", "compression_only", "sid_lite"]


def test_free_mad_lite_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "free_mad_lite",
            "inspect-experiment",
            "--experiment",
            "configs/families/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml",
        ],
    )
    assert payload["name"] == "free_mad_lite_mechanism_validation"
    assert payload["methods"] == [
        "mv_3",
        "vanilla_mad_r1_final_vote",
        "anti_conformity_final_vote",
        "free_mad_lite_llm_trajectory",
    ]
    assert payload["protocol"]["debate_rounds"] == 1
    assert payload["anti_conformity_prompt_hash"]


def test_comm_necessary_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "comm_necessary",
            "inspect-experiment",
            "--experiment",
            "configs/families/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml",
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

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "cache-inspector",
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


def test_cache_inspector_normalize_layout_cli(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    cache = router.for_request_target(
        provider="xiaomimimo",
        request_model="mimo-v2.5",
        dataset="gsm8k/test",
    )
    cache.put(
        CachedResponse(
            cache_key="legacy",
            payload_json=json_dump({"request": "legacy"}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_legacy",
        )
    )
    router.close()

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "cache-inspector",
            "normalize-layout",
            "--cache-root",
            str(tmp_path),
            "--json",
        ],
    )
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["source_dataset"] == "gsm8k/test"
    assert payload["candidates"][0]["target_dataset"] == "gsm8k"


def test_archive_runs_publish_uses_repo_env(monkeypatch, tmp_path) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "test-run"}, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setattr(
        "research_experiments.cli.tools.archive_runs.publish_run_to_hub",
        lambda run_root, repo_id, token, create_repo: {
            "run_root": str(run_root),
            "remote_repo": repo_id,
            "published": True,
            "create_repo": create_repo,
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "archive-runs",
            "publish-run",
            "--run-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-runs"
    assert payload["published"] is True


def test_archive_runs_fetch_accepts_run_prefix(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setattr(
        "research_experiments.cli.tools.archive_runs.fetch_run_from_hub",
        lambda run_id, repo_id, remote_prefix, token, target_root: {
            "run_id": run_id,
            "remote_repo": repo_id,
            "remote_prefix": remote_prefix,
            "target_root": str(target_root),
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "archive-runs",
            "fetch-run",
            "--run-prefix",
            "single_agent/demo/count20/20260510T000000Z-model",
            "--target-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-runs"
    assert payload["remote_prefix"] == "single_agent/demo/count20/20260510T000000Z-model"
    assert payload["run_id"] is None


def test_cache_archive_push_uses_repo_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.cache_archive.push_latest_cache_snapshot",
        lambda cache_root, repo_id, token, create_repo, private, shard_filters=None: {
            "cache_root": str(cache_root),
            "remote_repo": repo_id,
            "published": True,
            "private_repo": private,
            "shard_filters": shard_filters or [],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "cache-archive",
            "push-latest",
            "--cache-root",
            str(tmp_path),
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-cache"
    assert payload["private_repo"] is True


def test_cache_archive_pull_accepts_cache_shard(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.cache_archive.pull_latest_cache_snapshot",
        lambda target, repo_id, token, shard_filters=None: {
            "target_root": str(target),
            "remote_repo": repo_id,
            "shard_filters": shard_filters or [],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "cache-archive",
            "pull-latest",
            "--target",
            str(tmp_path),
            "--cache-shard",
            "providers/xiaomimimo/mimo-v2-5/strategyqa/dev",
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-cache"
    assert payload["shard_filters"] == ["providers/xiaomimimo/mimo-v2-5/strategyqa/dev"]


def test_hf_sync_push_workspace_uses_repo_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf_sync.push_workspace_to_hub",
        lambda **kwargs: {
            "runs_repo": kwargs["runs_repo_id"],
            "cache_repo": kwargs["cache_repo_id"],
            "publish_runs": kwargs["publish_runs"],
            "push_cache": kwargs["push_cache"],
            "selected_run_dirs": kwargs["selected_run_dirs"],
            "cache_shard_filters": kwargs["cache_shard_filters"],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf-sync",
            "push-workspace",
            "--runs-root",
            str(tmp_path / "runs"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--run-dir",
            str(tmp_path / "runs" / "single_agent" / "demo"),
            "--cache-shard",
            "providers/xiaomimimo/mimo-v2-5/strategyqa/dev",
            "--json",
        ],
    )

    assert payload["runs_repo"] == "owner/research-runs"
    assert payload["cache_repo"] == "owner/research-cache"
    assert payload["publish_runs"] is True
    assert payload["push_cache"] is True
    assert payload["selected_run_dirs"] == [str(tmp_path / "runs" / "single_agent" / "demo")]
    assert payload["cache_shard_filters"] == ["providers/xiaomimimo/mimo-v2-5/strategyqa/dev"]


def test_hf_sync_pull_workspace_uses_repo_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf_sync.pull_workspace_from_hub",
        lambda **kwargs: {
            "runs_repo": kwargs["runs_repo_id"],
            "cache_repo": kwargs["cache_repo_id"],
            "fetch_runs": kwargs["fetch_runs"],
            "pull_cache": kwargs["pull_cache"],
            "selected_run_ids": kwargs["selected_run_ids"],
            "selected_run_prefixes": kwargs["selected_run_prefixes"],
            "cache_shard_filters": kwargs["cache_shard_filters"],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf-sync",
            "pull-workspace",
            "--runs-root",
            str(tmp_path / "runs"),
            "--cache-root",
            str(tmp_path / "cache"),
            "--run-id",
            "20260510T000000Z-model",
            "--run-prefix",
            "single_agent/demo/count20/20260510T000000Z-model",
            "--cache-shard",
            "providers/xiaomimimo/mimo-v2-5/strategyqa/dev",
            "--json",
        ],
    )

    assert payload["runs_repo"] == "owner/research-runs"
    assert payload["cache_repo"] == "owner/research-cache"
    assert payload["fetch_runs"] is True
    assert payload["pull_cache"] is True
    assert payload["selected_run_ids"] == ["20260510T000000Z-model"]
    assert payload["selected_run_prefixes"] == ["single_agent/demo/count20/20260510T000000Z-model"]
    assert payload["cache_shard_filters"] == ["providers/xiaomimimo/mimo-v2-5/strategyqa/dev"]
