"""覆盖 faithful_matrix 编排与状态汇总逻辑的测试。"""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path

from budget_comm.config import load_experiment_config as load_budget_experiment_config
from experiment_core.faithful_matrix import (
    MATRIX_EXPERIMENT_KIND,
    RuntimeOverrides,
    _prepare_orchestrator_paths,
    apply_runtime_overrides,
    build_run_matrix,
    resume_faithful_matrix,
)
from single_agent.config import load_experiment_config as load_single_agent_experiment_config
from single_agent.config import required_model_tags


def test_build_run_matrix_counts_expected() -> None:
    overrides = RuntimeOverrides()
    matrix = build_run_matrix(overrides)
    semantic_counts = Counter(entry.status for entry in matrix.semantic_entries)
    entry_counts = Counter(entry.status for entry in matrix.entries)

    assert len(matrix.semantic_entries) == 16
    assert semantic_counts["pending"] == 16
    assert entry_counts["excluded"] == 1
    assert matrix.counts["semantic_unique_targets"] == 16
    cue_entry = next(entry for entry in matrix.semantic_entries if entry.experiment_name == "cue_v1")
    assert cue_entry.evaluation_track == "same_context"
    assert cue_entry.primary_method_name == "cue_v1"
    assert cue_entry.best_no_comm_candidates == ["mv_3"]


def test_build_run_matrix_counts_expected_for_pilot100() -> None:
    overrides = RuntimeOverrides(phase_name="pilot100")
    matrix = build_run_matrix(overrides)
    semantic_counts = Counter(entry.status for entry in matrix.semantic_entries)
    entry_counts = Counter(entry.status for entry in matrix.entries)

    assert len(matrix.semantic_entries) == 16
    assert semantic_counts["pending"] == 16
    assert entry_counts["excluded"] == 0
    assert matrix.counts["semantic_unique_targets"] == 16
    split_entry = next(
        entry
        for entry in matrix.semantic_entries
        if entry.experiment_name == "hotpotqa_split_main"
    )
    assert split_entry.phase_name == "pilot100"
    assert split_entry.evaluation_track == "split_context"


def test_apply_runtime_overrides_clears_single_agent_smoke20_tag_constraints() -> None:
    overrides = RuntimeOverrides()
    experiment = load_single_agent_experiment_config("configs/single_agent/experiments/robustness.toml")

    overridden = apply_runtime_overrides("single_agent", experiment, overrides)

    assert required_model_tags(experiment, "smoke20") == ["provider_zhipu"]
    assert required_model_tags(overridden, "smoke20") == []
    assert overridden.max_concurrent_requests == 85
    assert overridden.requests_per_minute_limit == 95
    assert overridden.tokens_per_minute_limit == 9000000


def test_apply_runtime_overrides_updates_backbone_without_mutating_source_config() -> None:
    overrides = RuntimeOverrides()
    experiment = load_budget_experiment_config("configs/budget_comm/experiments/dala_lite_same_context_v1.toml")

    overridden = apply_runtime_overrides("budget_comm", experiment, overrides)

    assert experiment.primary_model_ref == "deepseek/deepseek-v4-flash"
    assert overridden.primary_model_ref == "xiaomimimo/mimo-v2.5"
    assert overridden.max_concurrent_requests == 85
    assert overridden.requests_per_minute_limit == 95
    assert overridden.tokens_per_minute_limit == 9000000


def test_prepare_orchestrator_paths_uses_resolved_model_slug(tmp_path: Path) -> None:
    overrides = RuntimeOverrides(phase_name="pilot100", model_ref="openai/gpt-5.5")

    paths = _prepare_orchestrator_paths(tmp_path, overrides)

    assert MATRIX_EXPERIMENT_KIND == "faithful_matrix"
    assert "pilot100-openai-gpt-5.5" in paths.root.as_posix()


def test_resume_faithful_matrix_continues_rerun_needed_and_pending_entries(
    tmp_path: Path,
    monkeypatch,
) -> None:
    root = tmp_path / "20260506T000000Z-pilot100-xiaomimimo-mimo-v2.5"
    root.mkdir(parents=True)
    state_path = root / "state.json"
    payload = {
        "overrides": {
            "phase_name": "pilot100",
            "model_ref": "xiaomimimo/mimo-v2.5",
            "max_concurrent_requests": 60,
            "requests_per_minute_limit": 90,
            "tokens_per_minute_limit": 9000000,
        },
        "counts": {
            "completed": 0,
            "rerun-needed": 1,
            "pending": 0,
            "semantic_unique_targets": 1,
        },
        "entries": [
            {
                "family": "comm_necessary",
                "config_path": "configs/comm_necessary/experiments/hotpotqa_split_main.toml",
                "experiment_name": "hotpotqa_split_main",
                "description": "",
                "phase_name": "pilot100",
                "evaluation_track": "split_context",
                "primary_method_name": "full_packet_exchange",
                "best_no_comm_candidates": ["split_no_comm_mv3"],
                "full_comm_reference": None,
                "full_context_reference": "full_context_single",
                "status": "rerun-needed",
                "excluded_reason": None,
                "run_dir": None,
                "validation_passed": None,
                "review_passed": None,
                "review_notes": "validation_not_passed",
            },
        ],
    }
    payload["semantic_entries"] = payload["entries"]
    state_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _fake_execute_entry(entry, overrides):
        run_dir = root / entry.experiment_name
        run_dir.mkdir(parents=True, exist_ok=True)
        return run_dir

    monkeypatch.setattr("experiment_core.faithful_matrix._execute_entry", _fake_execute_entry)
    monkeypatch.setattr("experiment_core.faithful_matrix._validate_entry", lambda family, run_dir: {"passed": True})
    monkeypatch.setattr(
        "experiment_core.faithful_matrix.review_run_health",
        lambda run_dir, family: type("Review", (), {"passed": True, "notes": "validation_passed_and_metrics_nonempty"})(),
    )
    monkeypatch.setattr(
        "experiment_core.faithful_matrix.render_faithful_analysis",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        "experiment_core.faithful_matrix.render_acceptance_summary",
        lambda *args, **kwargs: {},
    )

    resumed_root = resume_faithful_matrix(state_path)

    assert resumed_root == root
    resumed = json.loads(state_path.read_text(encoding="utf-8"))
    statuses = {item["experiment_name"]: item["status"] for item in resumed["semantic_entries"]}
    assert statuses["hotpotqa_split_main"] == "completed"
