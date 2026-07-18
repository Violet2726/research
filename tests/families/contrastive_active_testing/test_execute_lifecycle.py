from __future__ import annotations

import json
import threading
import time
from dataclasses import replace

import pytest

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
)
from research_experiments.families.contrastive_active_testing.run import execute
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
    experiment = replace(experiment, provider_audit_path=tmp_path / "missing-provider-audit.json")
    backbone = resolve_model(experiment.primary_model_ref)
    sample = DatasetSample("bbeh", "unit", "Question", "A", "", {"task": "unit", "options": []})
    benchmark = load_phase_benchmarks(experiment, "development")[0]

    monkeypatch.setattr(execute, "OpenAICompatibleProvider", _DummyProvider)
    monkeypatch.setattr(
        execute,
        "_require_passing_provider_audit",
        lambda *_args, **_kwargs: pytest.fail("provider audit must not be consulted"),
    )
    monkeypatch.setattr(execute, "RequestCacheRouter", _DummyRouter)
    monkeypatch.setattr(execute, "RequestThrottle", _DummyThrottle)
    monkeypatch.setattr(execute, "load_phase_benchmarks", lambda *_args: [benchmark])
    monkeypatch.setattr(execute, "_select_phase_samples", lambda *_args: [sample])

    def fake_sample(*_args, **_kwargs):
        turn = {
            "dataset": "bbeh",
            "sample_id": "unit",
            "method_name": "catch_stage_a_shared",
            "role": "stage_a_solver",
            "cache_namespace": "catch-dev-v3",
            "request_source": "active_cache",
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
            "answer_class_key": "A",
            "normalized_answer": "A",
            "prediction": "A",
            "validated_output": {"reasoning": "r", "final_answer": "A"},
        }
        turns = [{**turn, "agent_id": index} for index in range(1, 6)]
        base = {
            "dataset": "bbeh",
            "sample_id": "unit",
            "task": "unit",
            "prediction": "A",
            "gold": "A",
            "score": 1.0,
            "candidate_oracle_correct": True,
            "target_oracle_correct": True,
            "triggered": False,
            "override_accepted": False,
            "corrected_by_debate": False,
            "harmed_by_debate": False,
            "total_tokens_per_question": 10,
            "network_attempts_per_question": 5,
            "logical_calls_per_question": 5,
        }
        predictions = [
            {**base, "method_name": "sc_5", "logical_calls_per_question": 5},
            {**base, "method_name": "adaptive_sc_8"},
            {**base, "method_name": "catch"},
            {**base, "method_name": "direct_judge_3"},
            {**base, "method_name": "pair_judge_3"},
        ]
        router = {
            "sample_id": "unit",
            "protocol_version": "catch_v3",
            "triggered": False,
            "candidate_oracle_correct": True,
            "target_oracle_correct": True,
            "eligible_challengers": [],
            "validated_contrasts": [],
            "witness_panels": [],
            "direct_judge_selections": [],
            "pair_judge_selections": [],
            "decision": {"override_accepted": False, "resolver": ""},
        }
        return turns, router, predictions

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
    report = (run_dir / "report.md").read_text(encoding="utf-8")
    assert progress["status"] == "completed"
    assert validation["passed"] is True
    assert validation["scientific_gate_applicable"] is False
    assert validation["performance_gate_passed"] is None
    assert not (run_dir / "diagnostics" / "gate.json").exists()
    assert not (run_dir / "diagnostics" / "preflight.json").exists()
    assert not (run_dir / "archive_manifest.json").exists()
    assert "Complete-case" in report
    assert "Wilson 95% CI" in report


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


def test_bounded_executor_records_worker_failure_and_finishes_siblings() -> None:
    class Endpoint:
        def __init__(self) -> None:
            self.stop_event = threading.Event()

    endpoints = [Endpoint() for _ in range(60)]
    jobs = [
        execute.CatchSampleJob(index, index, "dev", endpoints[index])
        for index in range(60)
    ]
    started: list[int] = []
    lock = threading.Lock()

    def worker(job):
        with lock:
            started.append(job.sequence_index)
        if job.sequence_index == 0:
            raise RuntimeError("worker failed")
        time.sleep(0.02)
        return [], {}, []

    class Progress:
        def update_scheduler_state(self, **_kwargs):
            pass

    completed = list(
        execute._execute_jobs_bounded(jobs, max_workers=15, worker=worker, progress=Progress())
    )
    assert len(completed) == 60
    assert len(started) == 60
    failed = [router for _, _, router, _ in completed if router.get("sample_error")]
    assert len(failed) == 1
    assert failed[0]["sample_error"]["error_type"] == "RuntimeError"
    assert not any(endpoint.stop_event.is_set() for endpoint in endpoints)


def test_partial_finalizer_persists_completed_predictions_into_precreated_file(tmp_path) -> None:
    root = tmp_path / "partial"
    (root / "turns").mkdir(parents=True)
    (root / "views").mkdir()
    (root / "diagnostics").mkdir()

    class Layout:
        pass

    layout = Layout()
    layout.root = root
    layout.agent_turns = root / "turns" / "agent_turns.jsonl"
    layout.router_decisions = root / "turns" / "router_decisions.jsonl"
    layout.preflight_turns = root / "turns" / "preflight_turns.jsonl"
    layout.predictions = root / "views" / "predictions.jsonl"
    layout.preflight = root / "diagnostics" / "preflight.json"
    layout.metrics = root / "views" / "metrics.json"
    layout.gate = root / "diagnostics" / "gate.json"
    layout.run_summary = root / "views" / "run_summary.json"
    layout.manifest = root / "manifest.json"
    layout.predictions.write_text("", encoding="utf-8")
    layout.manifest.write_text(
        json.dumps(
            {
                "method_version": "catch_v3",
                "run_mode": "full",
                "planned_sample_count": 2,
            }
        ),
        encoding="utf-8",
    )
    prediction = {"sample_id": "s0", "method_name": "catch", "prediction": "A"}
    execute._write_partial_outputs(
        layout,
        turns=[],
        routers=[{"sample_id": "s0"}],
        predictions=[prediction],
        termination_reason="execution_failure",
        error=RuntimeError("boom"),
    )
    assert json.loads(layout.predictions.read_text(encoding="utf-8")) == prediction
    summary = json.loads(layout.run_summary.read_text(encoding="utf-8"))
    assert summary["execution"]["attempted_sample_count"] == 1
    assert summary["execution"]["incomplete_sample_count"] == 1
    assert not layout.gate.exists()


def test_legacy_preflight_mode_no_longer_blocks_full_run(tmp_path, monkeypatch) -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_gate.toml"
    )
    backbone = resolve_model(experiment.primary_model_ref)
    sample = DatasetSample("bbeh", "unit", "Question", "A", "", {"task": "unit", "options": []})
    benchmark = load_phase_benchmarks(experiment, "development")[0]
    monkeypatch.setattr(execute, "OpenAICompatibleProvider", _DummyProvider)
    monkeypatch.setattr(execute, "RequestCacheRouter", _DummyRouter)
    monkeypatch.setattr(execute, "RequestThrottle", _DummyThrottle)
    monkeypatch.setattr(execute, "load_phase_benchmarks", lambda *_args: [benchmark])
    monkeypatch.setattr(execute, "_select_phase_samples", lambda *_args: [sample])
    main_called = False

    def main_sample(*_args, **_kwargs):
        nonlocal main_called
        main_called = True
        return [], {"dataset": "bbeh", "sample_id": "unit", "triggered": False}, []

    monkeypatch.setattr(
        execute,
        "_run_required_canonicalization_replay",
        lambda *_args, **_kwargs: {"passed": True, "metrics": {}, "hashes": {}},
    )
    monkeypatch.setattr(
        execute,
        "run_icv_structural_preflight",
        lambda *_args, **_kwargs: pytest.fail("preflight must not run"),
    )
    monkeypatch.setattr(execute, "run_catch_sample", main_sample)
    run_dir = execute.run_experiment(
        experiment,
        "development",
        backbone,
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
        run_mode="structural_preflight",
    )
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert main_called is True
    assert progress["status"] == "completed"
    assert manifest["run_mode"] == "full"
    assert "legacy_structural_preflight_request_ignored_running_full_phase" in manifest["execution_warnings"]


def test_heldout_runs_without_frozen_file_or_prior_development(tmp_path, monkeypatch) -> None:
    experiment = load_experiment_config(
        "configs/families/contrastive_active_testing/experiments/catch_gate.toml"
    )
    backbone = resolve_model(experiment.primary_model_ref)
    sample = DatasetSample("bbeh", "unit", "Question", "A", "", {"task": "unit", "options": []})
    experiment = replace(experiment, frozen_decoding_path=tmp_path / "missing.json")
    benchmark = load_phase_benchmarks(experiment, "heldout")[0]
    monkeypatch.setattr(execute, "OpenAICompatibleProvider", _DummyProvider)
    monkeypatch.setattr(execute, "RequestCacheRouter", _DummyRouter)
    monkeypatch.setattr(execute, "RequestThrottle", _DummyThrottle)
    monkeypatch.setattr(execute, "load_phase_benchmarks", lambda *_args: [benchmark])
    monkeypatch.setattr(execute, "_select_phase_samples", lambda *_args: [sample])

    main_called = False

    def main_sample(*_args, **_kwargs):
        nonlocal main_called
        main_called = True
        return [], {"dataset": "bbeh", "sample_id": "unit", "triggered": False}, []

    monkeypatch.setattr(execute, "run_catch_sample", main_sample)
    run_dir = execute.run_experiment(
        experiment,
        "heldout",
        backbone,
        run_root=tmp_path / "runs",
        cache_root=tmp_path / "cache",
    )

    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert main_called is True
    assert progress["status"] == "completed"
    assert manifest["frozen_decoding"]["source"] == "built_in_fixed_v3_decoder"
    assert "confirmatory_evidence_missing_or_not_enforced" in manifest["execution_warnings"]


def test_preflight_human_audit_requires_completed_adjudication(tmp_path) -> None:
    path = tmp_path / "audit.json"
    hashes = {f"hash-{index}" for index in range(40)}
    payload = {
        "audit_version": "catch_v3_icv_blind_coordinate_audit_v1",
        "adjudication_complete": False,
        "source_preflight_run_id": "preflight-run",
        "source_config_sha256": "config-sha",
        "seed": 42,
        "blind_to_gold_votes_and_candidate_answers": True,
        "items": [
            {
                "coordinate_sha256": coordinate_hash,
                "annotator_1": {
                    "decidable": True,
                    "mutually_exclusive": True,
                    "atomic": index >= 4,
                    "answer_leakage": False,
                },
                "annotator_2": {
                    "decidable": True,
                    "mutually_exclusive": True,
                    "atomic": index >= 4,
                    "answer_leakage": False,
                },
                "adjudication": None,
            }
            for index, coordinate_hash in enumerate(sorted(hashes))
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="adjudication_complete"):
        execute._require_passing_preflight_human_audit(
            path,
            expected_run_id="preflight-run",
            expected_config_sha="config-sha",
            expected_coordinate_hashes=hashes,
        )
    payload["adjudication_complete"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    execute._require_passing_preflight_human_audit(
        path,
        expected_run_id="preflight-run",
        expected_config_sha="config-sha",
        expected_coordinate_hashes=hashes,
    )


def test_v3_development_and_heldout_are_one_shot_by_manifest(tmp_path) -> None:
    for phase in ("development", "heldout"):
        run_dir = tmp_path / "catch_gate" / phase / f"{phase}-run"
        run_dir.mkdir(parents=True)
        (run_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "method_version": "catch_v3",
                    "run_mode": "full",
                    "run_status": "failed",
                }
            ),
            encoding="utf-8",
        )
        with pytest.raises(RuntimeError, match="one shot"):
            execute._require_unused_v3_full_phase_attempt(
                tmp_path,
                experiment_name="catch_gate",
                phase_name=phase,
            )


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
    assert validation["passed"] is True
    assert validation["artifact_valid"] is True
    assert validation["counts"]["completed_samples"] == 1
    assert validation["counts"]["incomplete_samples"] == 99
    assert (run_dir / "diagnostics" / "gate.json").exists()


def test_partial_finalizer_preserves_structural_preflight_sample_semantics(tmp_path) -> None:
    run_dir = tmp_path / "structural-preflight"
    (run_dir / "turns").mkdir(parents=True)
    (run_dir / "diagnostics").mkdir(parents=True)
    manifest = finalize_family_manifest(
        {
            "run_id": "structural-preflight",
            "family_name": "contrastive_active_testing",
            "phase_name": "development",
            "method_version": "catch_v3",
            "run_mode": "structural_preflight",
            "sample_count": 100,
            "planned_sample_count": 20,
            "cache_namespace": "catch-dev-v3",
            "request_source": "role_aware_versioned_catch_cache",
        },
        family_name="contrastive_active_testing",
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    (run_dir / "progress.json").write_text(json.dumps({"status": "running"}), encoding="utf-8")
    (run_dir / "diagnostics" / "preflight.json").write_text(
        json.dumps(
            {
                "passed": False,
                "status": "selector_failed",
                "selected_sample_ids": [f"sample-{index}" for index in range(20)],
                "selector_gate": {"evidence": {}},
            }
        ),
        encoding="utf-8",
    )

    execute.finalize_partial_run_directory(
        run_dir,
        termination_reason="structural_preflight_failed",
    )
    progress = json.loads((run_dir / "progress.json").read_text(encoding="utf-8"))
    gate = json.loads((run_dir / "diagnostics" / "gate.json").read_text(encoding="utf-8"))
    assert progress["total_planned_samples"] == 20
    assert progress["completed_samples"] == 20
    assert progress["incomplete_samples"] == 0
    assert gate["gate_name"] == "catch_v3_structural_preflight"
    assert gate["planned_sample_count"] == 20
    assert gate["completed_sample_count"] == 20
