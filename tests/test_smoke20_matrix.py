from __future__ import annotations

from collections import Counter

from budget_comm.config import load_experiment_config as load_budget_experiment_config
from experiment_core.smoke20_matrix import (
    RuntimeOverrides,
    apply_runtime_overrides,
    build_run_matrix,
)
from single_agent.config import load_experiment_config as load_single_agent_experiment_config
from single_agent.config import required_model_tags


def test_build_run_matrix_counts_expected() -> None:
    overrides = RuntimeOverrides()
    matrix = build_run_matrix(overrides)
    semantic_counts = Counter(entry.status for entry in matrix.semantic_entries)
    entry_counts = Counter(entry.status for entry in matrix.entries)

    assert len(matrix.semantic_entries) == 20
    assert semantic_counts["pending"] == 20
    assert entry_counts["excluded"] == 2
    assert matrix.counts["semantic_unique_targets"] == 20
    cue_entry = next(entry for entry in matrix.semantic_entries if entry.experiment_name == "cue_v1")
    assert cue_entry.evaluation_track == "same_context"
    assert cue_entry.primary_method_name == "cue_v1"
    assert cue_entry.best_no_comm_candidates == ["mv_3"]


def test_build_run_matrix_counts_expected_for_pilot100() -> None:
    overrides = RuntimeOverrides(phase_name="pilot100")
    matrix = build_run_matrix(overrides)
    semantic_counts = Counter(entry.status for entry in matrix.semantic_entries)
    entry_counts = Counter(entry.status for entry in matrix.entries)

    assert len(matrix.semantic_entries) == 20
    assert semantic_counts["pending"] == 20
    assert entry_counts["excluded"] == 1
    assert matrix.counts["semantic_unique_targets"] == 20
    split_entry = next(
        entry
        for entry in matrix.semantic_entries
        if entry.experiment_name == "hotpotqa_split_evidence_v1"
    )
    assert split_entry.phase_name == "pilot100"
    assert split_entry.evaluation_track == "split_context"


def test_apply_runtime_overrides_clears_single_agent_smoke20_tag_constraints() -> None:
    overrides = RuntimeOverrides()
    experiment = load_single_agent_experiment_config("configs/single_agent/experiments/robustness.toml")

    overridden = apply_runtime_overrides("single_agent", experiment, overrides)

    assert required_model_tags(experiment, "smoke20") == ["provider_zhipu"]
    assert required_model_tags(overridden, "smoke20") == []
    assert overridden.max_concurrent_requests == 60
    assert overridden.requests_per_minute_limit == 90
    assert overridden.tokens_per_minute_limit == 9000000


def test_apply_runtime_overrides_updates_backbone_without_mutating_source_config() -> None:
    overrides = RuntimeOverrides()
    experiment = load_budget_experiment_config("configs/budget_comm/experiments/dala_lite_same_context_v1.toml")

    overridden = apply_runtime_overrides("budget_comm", experiment, overrides)

    assert experiment.primary_model_ref == "deepseek/deepseek-v4-flash"
    assert overridden.primary_model_ref == "xiaomimimo/mimo-v2.5"
    assert overridden.max_concurrent_requests == 60
    assert overridden.requests_per_minute_limit == 90
    assert overridden.tokens_per_minute_limit == 9000000
