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
from research_experiments.core.execution.rate_limits import (
    STANDARD_MAX_CONCURRENT_REQUESTS,
    STANDARD_REQUESTS_PER_MINUTE_LIMIT,
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


def test_single_agent_canonical_simple_baselines_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "single_agent",
            "inspect-experiment",
            "--experiment",
            "configs/families/single_agent/experiments/canonical_simple_baselines.toml",
        ],
    )
    assert payload["name"] == "canonical_simple_baselines"
    assert payload["cot_uses_reruns"] is True
    assert payload["phases"]["count100"]["methods"] == ["cot_1", "mv_3", "sc_5"]
    assert payload["phases"]["count100"]["split_overrides"]["competition_math"] == "count100_seed42"
    assert payload["methods"]["cot_1"]["temperature"] == 0.7
    assert payload["methods"]["mv_3"]["temperature"] == 0.7
    assert payload["methods"]["sc_5"]["temperature"] == 0.7


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
    assert payload["counts"]["semantic_unique_targets"] == 15


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
    assert payload["control_output_protocol"] == "free_text_answer_v1"
    assert payload["mad_initial_output_protocol"] == "free_text_answer_v1"
    assert payload["mad_debate_output_protocol"] == "free_text_answer_v1"


def test_baseline_compare_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "baseline_compare",
            "inspect-experiment",
            "--experiment",
            "configs/families/baseline_compare/experiments/core_six_method_baseline.toml",
        ],
    )
    assert payload["name"] == "core_six_method_baseline"
    assert payload["benchmarks"] == [
        "competition_math",
        "gpqa_diamond",
        "gsm8k",
        "hotpotqa",
        "math500",
        "mmlu_pro",
    ]
    assert payload["control_method_names"] == ["cot_1", "sc_3", "sc_5"]
    assert payload["method_order"] == ["cot_1", "sc_3", "sc_5", "mad_3a_r1", "mad_3a_r2", "mad_5a_r1"]
    assert [setup["name"] for setup in payload["setups"]] == ["mad_3a_r1", "mad_3a_r2", "mad_5a_r1"]
    assert payload["phases"]["count20"]["setups"] == ["mad_3a_r1", "mad_3a_r2", "mad_5a_r1"]
    assert payload["phases"]["count100"]["setups"] == ["mad_3a_r1", "mad_3a_r2", "mad_5a_r1"]
    assert payload["phases"]["count300"]["split_overrides"]["gpqa_diamond"] == "full198_seed42"
    assert payload["control_methods"]["cot_1"]["budget_calls"] == 1
    assert payload["control_methods"]["sc_3"]["budget_calls"] == 3
    assert payload["control_methods"]["sc_5"]["budget_calls"] == 5
    assert payload["control_output_protocol"] == "free_text_answer_v1"
    assert payload["mad_initial_output_protocol"] == "free_text_answer_v1"
    assert payload["mad_debate_output_protocol"] == "free_text_answer_v1"


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


def test_adaptive_sparse_mad_main_v5_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "adaptive_sparse_mad",
            "inspect-experiment",
            "--experiment",
            "configs/families/adaptive_sparse_mad/experiments/same_context_main_v5.toml",
        ],
    )
    assert payload["name"] == "same_context_main_v5"
    assert payload["benchmarks"] == ["math500", "hotpotqa", "gpqa_diamond", "strategyqa", "mmlu_pro"]
    assert payload["methods"] == [
        "cot_1",
        "sc_3",
        "sc_5",
        "hetero_vote_3",
        "adaptive_sparse_debate_v1",
    ]
    assert payload["control_prompt_version"] == "single_agent_free_text_v1"
    assert payload["control_output_protocol"] == "free_text_answer_v1"
    assert payload["prompt_version"] == "adaptive_sparse_mad_free_text_debate_v1"
    assert payload["stage_a_response_format_mode"] == "free_text"
    assert payload["adaptive_response_format_mode"] == "free_text"
    assert payload["legacy_json_mode"] is False


def test_adaptive_sparse_mad_main_v6_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "adaptive_sparse_mad",
            "inspect-experiment",
            "--experiment",
            "configs/families/adaptive_sparse_mad/experiments/same_context_main_v6.toml",
        ],
    )
    assert payload["name"] == "same_context_main_v6"
    assert payload["benchmarks"] == ["math500", "hotpotqa", "gpqa_diamond", "strategyqa", "mmlu_pro"]
    assert payload["methods"] == [
        "cot_1",
        "sc_3",
        "sc_5",
        "hetero_vote_3",
        "adaptive_sparse_debate_v1",
        "adaptive_sparse_rescue_only_v1",
        "adaptive_sparse_probe_only_v1",
        "adaptive_sparse_rescue_probe_v1",
    ]
    assert payload["protocol"]["false_consensus_confidence_threshold"] == 0.9
    assert payload["protocol"]["family_promotion_gap_threshold"] == 0.35
    assert payload["protocol"]["post_probe_debate_gap_threshold"] == 0.35
    assert payload["stage_a_response_format_mode"] == "free_text"
    assert payload["adaptive_response_format_mode"] == "free_text"


def test_adaptive_sparse_mad_main_v5_json_legacy_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "adaptive_sparse_mad",
            "inspect-experiment",
            "--experiment",
            "configs/families/adaptive_sparse_mad/experiments/same_context_main_v5_json_legacy.toml",
        ],
    )
    assert payload["name"] == "same_context_main_v5_json_legacy"
    assert payload["methods"] == [
        "cot_1",
        "sc_3",
        "sc_5",
        "hetero_vote_3",
        "adaptive_sparse_debate_v1",
    ]
    assert payload["control_prompt_version"] == "single_agent_free_text_v1"
    assert payload["control_output_protocol"] == "free_text_answer_v1"
    assert payload["stage_a_response_format_mode"] == "json_object"
    assert payload["adaptive_response_format_mode"] == "json_object"
    assert payload["legacy_json_mode"] is True


def test_adaptive_sparse_mad_full_counterfactual_v1_screen_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "adaptive_sparse_mad",
            "inspect-experiment",
            "--experiment",
            "configs/families/adaptive_sparse_mad/experiments/same_context_full_counterfactual_v1_screen.toml",
        ],
    )
    assert payload["name"] == "same_context_full_counterfactual_v1_screen"
    assert payload["benchmarks"] == [
        "hotpotqa",
        "competition_math",
        "mmlu_pro",
        "gpqa_diamond",
        "math500",
        "strategyqa",
        "gsm8k",
    ]
    assert payload["prompt_version"] == "adaptive_sparse_mad_v4_evidence_gate"
    assert payload["methods"] == [
        "cot_1",
        "hetero_vote_3",
        "adaptive_gate_v4",
        "adaptive_counterfactual_v1",
    ]


def test_adaptive_sparse_mad_full_counterfactual_v1_inspect_cli() -> None:
    payload = run_cli_json(
        [
            "research_cli",
            "experiment",
            "--family",
            "adaptive_sparse_mad",
            "inspect-experiment",
            "--experiment",
            "configs/families/adaptive_sparse_mad/experiments/same_context_full_counterfactual_v1.toml",
        ],
    )
    assert payload["name"] == "same_context_full_counterfactual_v1"
    assert payload["benchmarks"] == [
        "hotpotqa",
        "competition_math",
        "mmlu_pro",
        "gpqa_diamond",
        "math500",
        "strategyqa",
        "gsm8k",
    ]
    assert payload["methods"] == [
        "cot_1",
        "mv_3",
        "sc_5",
        "hetero_vote_3",
        "adaptive_gate_v4",
        "adaptive_counterfactual_v1",
    ]
    assert payload["phases"]["count100"]["split_overrides"]["competition_math"] == "count100_seed42"


def test_hf_push_cache_uses_repo_env(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf.push_cache_to_hub",
        lambda: {
            "remote_repo": "owner/research-cache",
            "published": True,
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf",
            "push-cache",
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-cache"
    assert payload["published"] is True


def test_hf_pull_cache_uses_repo_env(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf.pull_cache_from_hub",
        lambda: {
            "remote_repo": "owner/research-cache",
            "fetched_shard_count": 2,
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf",
            "pull-cache",
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-cache"
    assert payload["fetched_shard_count"] == 2


def test_hf_push_runs_uses_new_arguments(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf.push_runs_to_hub",
        lambda **kwargs: {
            "remote_repo": "owner/research-runs",
            "sources": kwargs["sources"],
            "skip_validation": kwargs["skip_validation"],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf",
            "push-runs",
            "--source",
            str(tmp_path / "runs" / "single_agent"),
            "--skip-validation",
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-runs"
    assert payload["sources"] == [str(tmp_path / "runs" / "single_agent")]
    assert payload["skip_validation"] is True


def test_hf_pull_runs_uses_new_arguments(monkeypatch) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setattr(
        "research_experiments.cli.tools.hf.pull_runs_from_hub",
        lambda **kwargs: {
            "remote_repo": "owner/research-runs",
            "selected_prefixes": kwargs["prefixes"],
            "recent_hours": kwargs["recent_hours"],
        },
    )

    payload = run_cli_json(
        [
            "research_cli",
            "tools",
            "hf",
            "pull-runs",
            "--prefix",
            "single_agent/demo",
            "--recent-hours",
            "1",
            "--json",
        ],
    )

    assert payload["remote_repo"] == "owner/research-runs"
    assert payload["selected_prefixes"] == ["single_agent/demo"]
    assert payload["recent_hours"] == 1.0


def test_hf_is_the_only_huggingface_tool_family() -> None:
    from research_experiments.cli.main import TOOL_MAINS

    assert {"artifact-cleanup", "dataset-assets", "hf"} == set(TOOL_MAINS)
