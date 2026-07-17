from __future__ import annotations

import json

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
)
from research_experiments.families.contrastive_active_testing.run import execute
from research_experiments.family_runtime.config_helpers import resolve_model


class _DummyProvider:
    def __init__(self, _backbone) -> None:
        pass

    def close(self) -> None:
        pass


class _DummyRouter:
    def __init__(self, _root, *, namespace) -> None:
        self.namespace = namespace

    def for_request_target(self, **_kwargs):
        return object()

    def close(self) -> None:
        pass


class _DummyThrottle:
    @classmethod
    def for_model(cls, *_args, **_kwargs):
        return cls()

    def snapshot(self):
        return {}


def test_execute_always_lands_progress_and_validation_without_live_network(tmp_path, monkeypatch) -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_gate.toml"
    )
    backbone = resolve_model(experiment.primary_model_ref)
    sample = DatasetSample("bbeh", "unit", "Question", "A", "", {"task": "unit", "options": []})
    benchmark = load_phase_benchmarks(experiment, "development")[0]

    monkeypatch.setattr(execute, "_require_passing_provider_audit", lambda *args, **kwargs: {"passed": True})
    monkeypatch.setattr(execute, "OpenAICompatibleProvider", _DummyProvider)
    monkeypatch.setattr(execute, "RequestCacheRouter", _DummyRouter)
    monkeypatch.setattr(execute, "RequestThrottle", _DummyThrottle)
    monkeypatch.setattr(execute, "load_phase_benchmarks", lambda *_args: [benchmark])
    monkeypatch.setattr(execute, "_select_phase_samples", lambda *_args: [sample])

    def fake_sample(*_args, **_kwargs):
        turn = {
            "dataset": "bbeh",
            "sample_id": "unit",
            "method_name": "catch_test_designer_shared",
            "role": "test_designer",
            "cache_namespace": "catch-dev-v1",
            "request_source": "catch_confirmation_cache",
            "payload": {"max_completion_tokens": 4096},
            "cache_key": "key",
            "raw_finish_reason": "stop",
            "network_attempt_count": 1,
            "network_request_count": 1,
            "request_count": 1,
            "cache_request_count": 0,
            "cache_hit": False,
            "request_error": None,
            "usage_source": "reported",
            "actual_total_tokens": 2,
            "actual_completion_tokens": 1,
            "reasoning_tokens": 0,
            "protocol_parse_status": "ok",
        }
        base = {
            "dataset": "bbeh",
            "sample_id": "unit",
            "task": "unit",
            "prediction": "A",
            "gold": "A",
            "score": 1.0,
            "candidate_oracle_correct": True,
            "triggered": True,
            "override_accepted": False,
            "corrected_by_debate": False,
            "harmed_by_debate": False,
            "total_tokens_per_question": 2,
            "network_attempts_per_question": 1,
            "logical_calls_per_question": 8,
        }
        predictions = [
            {**base, "method_name": "sc_5", "logical_calls_per_question": 5},
            {**base, "method_name": "adaptive_sc_8"},
            *[
                {**base, "method_name": f"catch_d{d_min}_m{margin}"}
                for d_min in (2, 3, 4)
                for margin in (1, 2)
            ],
            {**base, "method_name": "direct_judge_3"},
        ]
        variants = [
            {"d_min": d_min, "margin": margin, "pair_distances": {"B": d_min}}
            for d_min in (2, 3, 4)
            for margin in (1, 2)
        ]
        router = {"triggered": True, "catch_variants": variants}
        return [turn], router, predictions

    monkeypatch.setattr(execute, "run_catch_sample", fake_sample)
    run_dir = execute.run_experiment(
        experiment,
        "development",
        backbone,
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )

    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "diagnostics" / "gate.json").read_text(encoding="utf-8"))
    assert progress["status"] == "completed"
    assert validation["passed"] is True
    assert validation["performance_gate_passed"] is False
    assert gate["passed"] is False
