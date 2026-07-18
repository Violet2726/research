from __future__ import annotations

import json
import threading
import time

import pytest

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
)
from research_experiments.families.contrastive_active_testing.run import execute
from research_experiments.families.contrastive_active_testing.run.preflight import PreflightGateFailed
from research_experiments.family_runtime.config_helpers import resolve_model
from research_experiments.family_runtime.manifest import finalize_family_manifest


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

    def fake_preflight(_jobs, *, turns_path, output_path, **_kwargs):
        turns_path.write_text("", encoding="utf-8")
        payload = {"status": "passed", "passed": True, "uses_gold": False}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    monkeypatch.setattr(execute, "run_structural_preflight", fake_preflight)

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


def test_bounded_sample_executor_uses_parallel_workers_without_exceeding_cap() -> None:
    jobs = [execute.CatchSampleJob(index, index, "dev", None) for index in range(60)]
    lock = threading.Lock()
    active = 0
    peak = 0

    def worker(job):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return [], {}, []

    class Progress:
        def update_scheduler_state(self, **_kwargs):
            pass

    completed = list(
        execute._execute_jobs_bounded(jobs, max_workers=15, worker=worker, progress=Progress())
    )
    assert len(completed) == 60
    assert 1 < peak <= 15


def test_parallel_completion_order_has_same_stable_predictions_as_serial() -> None:
    jobs = [execute.CatchSampleJob(index, index, "dev", None) for index in range(20)]

    def worker(job):
        time.sleep((20 - job.sequence_index) * 0.0005)
        return [], {}, [{"method_name": "sc_5", "prediction": str(job.sequence_index)}]

    class Progress:
        def update_scheduler_state(self, **_kwargs):
            pass

    parallel = []
    for job, _, _, rows in execute._execute_jobs_bounded(
        jobs,
        max_workers=5,
        worker=worker,
        progress=Progress(),
    ):
        parallel.extend({**row, "sample_sequence_index": job.sequence_index} for row in rows)
    parallel.sort(key=execute._prediction_sort_key)
    serial = [
        {"method_name": "sc_5", "prediction": str(job.sequence_index), "sample_sequence_index": job.sequence_index}
        for job in jobs
    ]
    assert parallel == serial


def test_failed_preflight_stops_main_run_and_lands_terminal_artifacts(tmp_path, monkeypatch) -> None:
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
    main_called = False

    def fail_preflight(_jobs, *, turns_path, output_path, **_kwargs):
        turns_path.write_text("", encoding="utf-8")
        payload = {"status": "designer_failed", "passed": False}
        output_path.write_text(json.dumps(payload), encoding="utf-8")
        raise PreflightGateFailed(payload)

    def main_sample(*_args, **_kwargs):
        nonlocal main_called
        main_called = True
        return [], {}, []

    monkeypatch.setattr(execute, "run_structural_preflight", fail_preflight)
    monkeypatch.setattr(execute, "run_catch_sample", main_sample)
    with pytest.raises(PreflightGateFailed):
        execute.run_experiment(
            experiment,
            "development",
            backbone,
            run_root=tmp_path / "runs",
            cache_root=tmp_path / "cache",
        )
    run_dir = next((tmp_path / "runs" / "catch_gate" / "development").iterdir())
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert main_called is False
    assert progress["status"] == "failed"
    assert progress["failure"]["termination_reason"] == "structural_preflight_failed"
    assert validation["passed"] is False


def test_hard_stopped_v1_run_can_be_finalized_as_failed_futility(tmp_path) -> None:
    run_dir = tmp_path / "historical-v1"
    (run_dir / "turns").mkdir(parents=True)
    manifest = finalize_family_manifest(
        {
            "run_id": "historical-v1",
            "family_name": "contrastive_active_testing",
            "phase_name": "development",
            "sample_count": 100,
            "cache_namespace": "catch-dev-v1",
            "request_source": "fresh_catch_confirmation_cache",
        },
        family_name="contrastive_active_testing",
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (run_dir / "turns" / "router_decisions.jsonl").write_text(
        json.dumps({"sample_id": "x"}) + "\n",
        encoding="utf-8",
    )

    result = execute.finalize_partial_run_directory(run_dir)
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    validation = json.loads((run_dir / "run_validation.json").read_text(encoding="utf-8"))
    assert result["termination_reason"] == "futility_gate_impossible"
    assert progress["status"] == "failed"
    assert progress["completed_samples"] == 1
    assert progress["incomplete_samples"] == 99
    assert validation["passed"] is False
    assert validation["counts"]["completed_samples"] == 1
    assert validation["counts"]["incomplete_samples"] == 99
    assert (run_dir / "diagnostics" / "gate.json").exists()
