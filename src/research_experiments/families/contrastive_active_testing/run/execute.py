"""带隔离缓存、硬门控和可审计生命周期的 CATCH 执行器。"""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Event
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import load_split_ids, select_samples
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.families.contrastive_active_testing.cache_layers import ReadThroughRequestCache
from research_experiments.families.contrastive_active_testing.config import (
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    CATCH_PROMPT_VERSION,
    CATCH_SCHEMA_VERSION,
)
from research_experiments.families.contrastive_active_testing.run.preflight import (
    PreflightGateFailed,
    PreflightJob,
    run_structural_preflight,
)
from research_experiments.families.contrastive_active_testing.run.report import render_report
from research_experiments.families.contrastive_active_testing.run.sample import (
    NetworkAttemptBudget,
    run_catch_sample,
)
from research_experiments.families.contrastive_active_testing.run.validate import validate_run
from research_experiments.families.contrastive_active_testing.statistics import (
    build_metrics,
    evaluate_gate,
    materialize_development_catch,
)
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.workspace.layout import default_cache_root, default_runs_root


@dataclass(frozen=True)
class CatchEndpoint:
    backbone: object
    provider: OpenAICompatibleProvider
    baseline_cache: object
    intervention_cache: object
    throttle: RequestThrottle
    cache_namespace: str
    baseline_cache_namespace: str | None = None
    stop_event: Event | None = None

    def cache_for_role(self, role: str):
        if role in {"stage_a_solver", "independent_resample", "direct_judge"}:
            return self.baseline_cache
        return self.intervention_cache

    def cache_lookup_namespaces_for_role(self, role: str) -> tuple[str, ...]:
        if role in {"stage_a_solver", "independent_resample", "direct_judge"} and self.baseline_cache_namespace:
            return (self.cache_namespace, self.baseline_cache_namespace)
        return (self.cache_namespace,)


@dataclass(frozen=True)
class CatchSampleJob:
    sequence_index: int
    sample: Any
    split_name: str
    endpoint: CatchEndpoint


def run_experiment(
    experiment,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    if backbone.provider != "xiaomimimo":
        raise ValueError("CATCH is frozen to the audited xiaomimimo provider.")
    phase = phase_metadata(experiment, phase_name)
    protocol = load_protocol_config(experiment.protocol)
    config_sha = _frozen_config_sha(experiment)
    provider_audit = _require_passing_provider_audit(
        experiment.provider_audit_path,
        expected_cache_namespace=experiment.cache_namespaces["provider_audit"],
        expected_provider=backbone.provider,
        expected_model_id=backbone.model_id,
    )
    run_root = Path(run_root or default_runs_root("contrastive_active_testing"))
    cache_root = Path(cache_root or default_cache_root())
    frozen_decoding = None
    if phase_name in {"heldout", "confirmation"}:
        frozen_decoding = _load_frozen_decoding(experiment.frozen_decoding_path, config_sha=config_sha)
        _require_passing_gate(
            run_root,
            experiment_name=experiment.name,
            phase_name="development",
            model_name=backbone.name,
            config_sha=config_sha,
            frozen_sha=str(frozen_decoding["sha256"]),
        )
    if phase_name == "confirmation":
        _require_passing_gate(
            run_root,
            experiment_name=experiment.name,
            phase_name="heldout",
            model_name=backbone.name,
            config_sha=config_sha,
            frozen_sha=str(frozen_decoding["sha256"]),
        )
        _require_passing_human_audit(experiment.human_audit_path)

    cache_namespace = experiment.cache_namespaces[phase_name]
    provider = OpenAICompatibleProvider(backbone)
    active_router = RequestCacheRouter(cache_root, namespace=cache_namespace)
    baseline_namespace = experiment.baseline_cache_namespaces.get(phase_name)
    baseline_router = (
        RequestCacheRouter(cache_root, namespace=baseline_namespace)
        if baseline_namespace is not None
        else None
    )
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    network_budget = NetworkAttemptBudget(protocol.max_network_attempts)
    run_id = build_run_id(backbone.name)
    layout = prepare_registered_run_layout(
        "contrastive_active_testing",
        run_root,
        experiment.name,
        phase_name,
        run_id,
    )
    benchmarks = load_phase_benchmarks(experiment, phase_name)
    selected_by_benchmark = {
        benchmark.slug: _select_phase_samples(benchmark, phase, phase_name)
        for benchmark in benchmarks
    }
    sample_count = sum(len(samples) for samples in selected_by_benchmark.values())
    endpoints: dict[str, CatchEndpoint] = {}
    stop_event = Event()
    for benchmark in benchmarks:
        active_cache = active_router.for_request_target(
            provider=backbone.provider,
            request_model=backbone.model_id,
            dataset=benchmark.slug,
        )
        fallback_cache = (
            baseline_router.for_request_target(
                provider=backbone.provider,
                request_model=backbone.model_id,
                dataset=benchmark.slug,
            )
            if baseline_router is not None
            else None
        )
        baseline_cache = ReadThroughRequestCache(
            active_cache,
            primary_namespace=cache_namespace,
            fallback=fallback_cache,
            fallback_namespace=baseline_namespace,
        )
        endpoints[benchmark.slug] = CatchEndpoint(
            backbone=backbone,
            provider=provider,
            baseline_cache=baseline_cache,
            intervention_cache=active_cache,
            throttle=throttle,
            cache_namespace=cache_namespace,
            baseline_cache_namespace=baseline_namespace,
            stop_event=stop_event,
        )
    jobs: list[CatchSampleJob] = []
    sequence_index = 0
    for benchmark in benchmarks:
        split_name = str(phase["split_overrides"][benchmark.slug])
        for sample in selected_by_benchmark[benchmark.slug]:
            jobs.append(CatchSampleJob(sequence_index, sample, split_name, endpoints[benchmark.slug]))
            sequence_index += 1
    run_direct_judge = bool(phase.get("run_direct_judge", phase_name != "confirmation"))
    calls_per_triggered = 18 if phase_name == "development" else 14 if run_direct_judge else 11
    predictions_per_sample = 9 if phase_name == "development" else 4 if run_direct_judge else 3
    preflight_call_upper_bound = (
        sample_count * protocol.stage_candidates
        + protocol.preflight_sample_count * (1 + protocol.witness_count)
        if phase_name == "development" and protocol.protocol_version == "catch_v2"
        else 0
    )
    progress = RunProgressTracker(
        layout.progress,
        total_planned_calls=sample_count * calls_per_triggered + preflight_call_upper_bound,
        total_planned_predictions=sample_count * predictions_per_sample,
        total_planned_samples=sample_count,
        planned_calls_are_upper_bound=True,
        target_network_rpm=experiment.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": "contrastive_active_testing",
            "paper_method_name": "CATCH",
            "method_version": protocol.protocol_version,
            "protocol_version": protocol.protocol_version,
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "description": experiment.description,
            "resolved_model": asdict(backbone),
            "protocol": asdict(protocol),
            "prompt_version": CATCH_PROMPT_VERSION,
            "schema_version": CATCH_SCHEMA_VERSION,
            "global_seed": experiment.global_seed,
            "cache_namespace": cache_namespace,
            "baseline_read_cache_namespace": baseline_namespace,
            "request_source": "role_aware_catch_v2_cache",
            "provider_audit": provider_audit,
            "frozen_config_sha256": config_sha,
            "frozen_decoding": frozen_decoding,
            "phase_metadata": phase,
            "benchmarks": [asdict(item) for item in benchmarks],
            "sample_count": sample_count,
            "method_order": ["sc_5", "adaptive_sc_8", "catch", "direct_judge_3"],
            "max_network_attempts": protocol.max_network_attempts,
            "calls_per_triggered_question_upper_bound": calls_per_triggered,
            "preflight_call_upper_bound": preflight_call_upper_bound,
            "run_status": "running",
            "dgcr_predecessor_status": "retired_exact_span_reconstruction_channel_failed",
        },
        family_name="contrastive_active_testing",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    if not (phase_name == "development" and protocol.protocol_version == "catch_v2"):
        layout.preflight_turns.write_text("", encoding="utf-8")
        layout.preflight.write_text(
            json.dumps({"status": "not_applicable", "passed": True}, indent=2),
            encoding="utf-8",
        )
    turns: list[dict[str, Any]] = []
    routers: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    try:
        preflight = None
        if phase_name == "development" and protocol.protocol_version == "catch_v2":
            preflight = run_structural_preflight(
                [PreflightJob(job.sequence_index, job.sample, job.split_name, job.endpoint) for job in jobs],
                run_id=run_id,
                experiment=experiment,
                protocol=protocol,
                network_budget=network_budget,
                progress=progress,
                turns_path=layout.root / "turns" / "preflight_turns.jsonl",
                output_path=layout.root / "diagnostics" / "preflight.json",
            )
        with (
            layout.agent_turns.open("w", encoding="utf-8") as turns_handle,
            layout.router_decisions.open("w", encoding="utf-8") as routers_handle,
        ):
            for job, sample_turns, router_row, sample_predictions in _execute_jobs_bounded(
                jobs,
                max_workers=experiment.max_concurrent_requests,
                worker=lambda job: run_catch_sample(
                    job.sample,
                    run_id=run_id,
                    split_name=job.split_name,
                    experiment=experiment,
                    protocol=protocol,
                    endpoint=job.endpoint,
                    network_budget=network_budget,
                    phase_name=phase_name,
                    frozen_decoding=frozen_decoding,
                    run_direct_judge=run_direct_judge,
                ),
                progress=progress,
            ):
                for raw in sample_turns:
                    row = {**raw, "sample_sequence_index": job.sequence_index, "run_stage": "main"}
                    turns.append(row)
                    turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    progress.record_call(row)
                router_row = {**router_row, "sample_sequence_index": job.sequence_index}
                routers.append(router_row)
                routers_handle.write(json.dumps(router_row, ensure_ascii=False) + "\n")
                indexed_predictions = [
                    {**row, "sample_sequence_index": job.sequence_index} for row in sample_predictions
                ]
                predictions.extend(indexed_predictions)
                turns_handle.flush()
                routers_handle.flush()
                progress.record_predictions(
                    len(indexed_predictions),
                    job.sample.dataset,
                    "catch_sample",
                    sample_completed=True,
                )

        turns.sort(key=_turn_sort_key)
        routers.sort(key=lambda row: int(row.get("sample_sequence_index") or 0))
        predictions.sort(key=_prediction_sort_key)
        development_selection = None
        if phase_name == "development":
            predictions, development_selection = materialize_development_catch(predictions, routers)
        with layout.predictions.open("w", encoding="utf-8") as predictions_handle:
            for row in predictions:
                predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = build_metrics(predictions)
        preflight_turn_rows = _read_jsonl(layout.preflight_turns)
        gate = evaluate_gate(
            phase_name=phase_name,
            predictions=predictions,
            turns=[*preflight_turn_rows, *turns],
            routers=routers,
            development_selection=development_selection,
        )
        gate["actual_network_attempts"] = network_budget.actual
        gate["network_attempt_cap"] = protocol.max_network_attempts
        layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.gate.write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        if development_selection is not None:
            frozen_candidate = _build_frozen_decoding_candidate(
                run_id=run_id,
                config_sha=config_sha,
                selection=development_selection,
            )
            layout.frozen_decoding_candidate.write_text(
                json.dumps(frozen_candidate, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        layout.run_summary.write_text(
            json.dumps(
                {
                    "metrics": metrics,
                    "gate": gate,
                    "development_selection": development_selection,
                    "preflight": preflight,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        _update_manifest_status(layout.manifest, "completed")
        render_report(layout.root)
        finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_completed()
        return layout.root
    except BaseException as exc:
        termination_reason = (
            "structural_preflight_failed"
            if isinstance(exc, PreflightGateFailed)
            else "futility_gate_impossible"
            if isinstance(exc, KeyboardInterrupt)
            else "execution_failure"
        )
        _write_partial_outputs(
            layout,
            turns=turns,
            routers=routers,
            predictions=predictions,
            termination_reason=termination_reason,
            error=exc,
        )
        _update_manifest_status(layout.manifest, "failed", termination_reason=termination_reason)
        if not layout.validation.exists():
            with suppress(BaseException):
                finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_failed(
            type(exc).__name__,
            str(exc),
            last_sample_id=progress.last_sample_id,
            termination_reason=termination_reason,
        )
        raise
    finally:
        progress.close()
        provider.close()
        active_router.close()
        if baseline_router is not None:
            baseline_router.close()


def _execute_jobs_bounded(
    jobs: list[CatchSampleJob],
    *,
    max_workers: int,
    worker,
    progress: RunProgressTracker,
):
    """Keep at most max_workers sample futures alive and yield completed sample blocks."""

    executor = ThreadPoolExecutor(max_workers=max(1, max_workers), thread_name_prefix="catch-sample")
    iterator = iter(jobs)
    pending: dict[Future, CatchSampleJob] = {}

    def submit_one() -> bool:
        try:
            job = next(iterator)
        except StopIteration:
            return False
        pending[executor.submit(worker, job)] = job
        return True

    try:
        for _ in range(min(max(1, max_workers), len(jobs))):
            submit_one()
        progress.update_scheduler_state(
            in_flight_samples=len(pending),
            queued_samples=max(0, len(jobs) - len(pending)),
        )
        completed = 0
        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                job = pending.pop(future)
                result = future.result()
                completed += 1
                submit_one()
                progress.update_scheduler_state(
                    in_flight_samples=len(pending),
                    queued_samples=max(0, len(jobs) - completed - len(pending)),
                )
                yield (job, *result)
    except BaseException:
        for job in jobs:
            if job.endpoint.stop_event is not None:
                job.endpoint.stop_event.set()
        for future in pending:
            future.cancel()
        raise
    finally:
        executor.shutdown(wait=True, cancel_futures=True)
        progress.update_scheduler_state(in_flight_samples=0, queued_samples=0)


def _turn_sort_key(row: dict[str, Any]) -> tuple[int, str, int]:
    return (
        int(row.get("sample_sequence_index") or 0),
        str(row.get("method_name") or ""),
        int(row.get("agent_id") or 0),
    )


def _prediction_sort_key(row: dict[str, Any]) -> tuple[int, str]:
    method_order = {
        "sc_5": "00",
        "adaptive_sc_8": "01",
        "catch": "02",
        "direct_judge_3": "03",
    }
    method = str(row.get("method_name") or "")
    return int(row.get("sample_sequence_index") or 0), method_order.get(method, f"10:{method}")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_partial_outputs(
    layout,
    *,
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    termination_reason: str,
    error: BaseException,
) -> None:
    """Land a self-describing failed run without fabricating missing scientific outputs."""

    if not layout.agent_turns.exists():
        layout.agent_turns.write_text("", encoding="utf-8")
    if turns and layout.agent_turns.stat().st_size == 0:
        layout.agent_turns.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in turns),
            encoding="utf-8",
        )
    if not layout.router_decisions.exists():
        layout.router_decisions.write_text("", encoding="utf-8")
    if not layout.preflight_turns.exists():
        layout.preflight_turns.write_text("", encoding="utf-8")
    if not layout.preflight.exists():
        layout.preflight.write_text(
            json.dumps({"status": "not_completed", "passed": False}, indent=2),
            encoding="utf-8",
        )
    if routers and layout.router_decisions.stat().st_size == 0:
        layout.router_decisions.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in routers),
            encoding="utf-8",
        )
    if not layout.predictions.exists():
        layout.predictions.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
            encoding="utf-8",
        )
    failure_gate = {
        "gate_name": "catch_failed_partial_v2",
        "passed": False,
        "termination_reason": termination_reason,
        "completed_sample_count": len(routers),
        "completed_prediction_count": len(predictions),
        "error": {"error_type": type(error).__name__, "message": str(error)},
    }
    if not layout.metrics.exists():
        layout.metrics.write_text(json.dumps({"summary": []}, indent=2), encoding="utf-8")
    if not layout.gate.exists():
        layout.gate.write_text(json.dumps(failure_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    if not layout.run_summary.exists():
        layout.run_summary.write_text(
            json.dumps(
                {"metrics": {"summary": []}, "gate": failure_gate, "termination_reason": termination_reason},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
    with suppress(BaseException):
        render_report(layout.root)


def _update_manifest_status(path: Path, status: str, *, termination_reason: str | None = None) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["run_status"] = status
    if termination_reason is not None:
        payload["termination_reason"] = termination_reason
    payload["status_updated_at"] = datetime.now(UTC).isoformat()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def finalize_partial_run_directory(
    run_dir: str | Path,
    *,
    termination_reason: str = "futility_gate_impossible",
) -> dict[str, Any]:
    """Recover a hard-stopped CATCH run into an auditable failed terminal state."""

    root = Path(run_dir)
    manifest_path = root / "manifest.json"
    progress_path = root / "progress.json"
    if not manifest_path.exists() or not progress_path.exists():
        raise FileNotFoundError("partial finalization requires manifest.json and progress.json")
    _update_manifest_status(manifest_path, "failed", termination_reason=termination_reason)
    manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    planned_sample_count = int(manifest_payload.get("sample_count") or 0)
    router_path = root / "turns" / "router_decisions.jsonl"
    completed_sample_ids: set[str] = set()
    if router_path.exists():
        for line in router_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                sample_id = json.loads(line).get("sample_id")
            except (AttributeError, json.JSONDecodeError):
                continue
            if sample_id:
                completed_sample_ids.add(str(sample_id))
    completed_sample_count = len(completed_sample_ids)
    incomplete_sample_count = max(0, planned_sample_count - completed_sample_count)
    progress_payload = json.loads(progress_path.read_text(encoding="utf-8"))
    progress_payload.update(
        {
            "status": "failed",
            "last_write_reason": "failed",
            "last_updated_at": datetime.now(UTC).isoformat(),
            "failure": {
                "error_type": "FutilityStop",
                "message": "run stopped after its frozen gate became unreachable",
                "termination_reason": termination_reason,
            },
            "total_planned_samples": planned_sample_count,
            "completed_samples": completed_sample_count,
            "incomplete_samples": incomplete_sample_count,
        }
    )
    progress_path.write_text(json.dumps(progress_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    required_empty = (
        root / "turns" / "agent_turns.jsonl",
        root / "turns" / "router_decisions.jsonl",
        root / "turns" / "preflight_turns.jsonl",
        root / "views" / "predictions.jsonl",
    )
    for path in required_empty:
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            path.write_text("", encoding="utf-8")
    metrics_path = root / "views" / "metrics.json"
    gate_path = root / "diagnostics" / "gate.json"
    preflight_path = root / "diagnostics" / "preflight.json"
    summary_path = root / "views" / "run_summary.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    if not metrics_path.exists():
        metrics_path.write_text(json.dumps({"summary": []}, indent=2), encoding="utf-8")
    failure_gate = {
        "gate_name": "catch_failed_partial",
        "passed": False,
        "termination_reason": termination_reason,
        "planned_sample_count": planned_sample_count,
        "completed_sample_count": completed_sample_count,
        "incomplete_sample_count": incomplete_sample_count,
    }
    gate_path.write_text(json.dumps(failure_gate, ensure_ascii=False, indent=2), encoding="utf-8")
    if not preflight_path.exists():
        preflight_path.write_text(
            json.dumps({"status": "not_applicable_historical_v1", "passed": False}, indent=2),
            encoding="utf-8",
        )
    if not summary_path.exists():
        summary_path.write_text(
            json.dumps({"metrics": {"summary": []}, "gate": failure_gate}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    render_report(root)
    validation = finalize_run_outputs(root, validator=validate_run, validation_path=root / "run_validation.json")
    return {"run_dir": root.as_posix(), "termination_reason": termination_reason, "validation": validation}


def _select_phase_samples(benchmark, phase: dict[str, Any], phase_name: str):
    split_name = str(phase["split_overrides"][benchmark.slug])
    samples = select_samples(benchmark, split_name)
    if phase_name != "confirmation":
        return samples
    excluded_names = dict(phase.get("exclude_splits") or {}).get(benchmark.slug, [])
    excluded: set[str] = set()
    for excluded_name in excluded_names:
        excluded.update(load_split_ids(benchmark.cache_namespace or benchmark.slug, str(excluded_name)))
    return [sample for sample in samples if sample.sample_id not in excluded]


def _frozen_config_sha(experiment) -> str:
    payload = {
        "experiment": experiment.raw,
        "protocol": Path(experiment.protocol).read_text(encoding="utf-8"),
        "prompt_version": CATCH_PROMPT_VERSION,
        "schema_version": CATCH_SCHEMA_VERSION,
        "decoder_version": "catch_ecoc_decoder_v2",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _build_frozen_decoding_candidate(*, run_id: str, config_sha: str, selection: dict[str, Any]) -> dict[str, Any]:
    winner = dict(selection["selected"])
    payload = {
        "freeze_kind": "catch_decoding_v2",
        "source_development_run_id": run_id,
        "source_config_sha256": config_sha,
        "d_min": int(winner["d_min"]),
        "margin": int(winner["margin"]),
        "selection_constraints_passed": bool(selection.get("positive_constraints_satisfied")),
        "prompt_version": CATCH_PROMPT_VERSION,
        "schema_version": CATCH_SCHEMA_VERSION,
        "decoder_version": "catch_ecoc_decoder_v2",
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    return payload


def _load_frozen_decoding(path: Path, *, config_sha: str) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"CATCH held-out/confirmation is blocked: frozen decoding file is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    expected_sha = str(payload.get("sha256") or "")
    unsigned = dict(payload)
    unsigned.pop("sha256", None)
    actual_sha = hashlib.sha256(json.dumps(unsigned, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
    if expected_sha != actual_sha:
        raise RuntimeError("CATCH frozen decoding hash is invalid.")
    if payload.get("freeze_kind") != "catch_decoding_v2" or payload.get("decoder_version") != "catch_ecoc_decoder_v2":
        raise RuntimeError("CATCH frozen decoding uses a retired protocol version.")
    if payload.get("source_config_sha256") != config_sha or not payload.get("selection_constraints_passed"):
        raise RuntimeError("CATCH frozen decoding does not match the active config or failed development constraints.")
    if int(payload.get("d_min") or 0) not in {2, 3, 4} or int(payload.get("margin") or 0) not in {1, 2}:
        raise RuntimeError("CATCH frozen decoding contains an out-of-grid threshold.")
    return payload


def _require_passing_provider_audit(
    path: Path,
    *,
    expected_cache_namespace: str,
    expected_provider: str,
    expected_model_id: str,
) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"CATCH gate is blocked: required provider audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_kind") != "mimo_live_provider_contract_v1" or payload.get("cache_mode") != "bypassed":
        raise RuntimeError("CATCH gate is blocked: provider audit has the wrong contract or cache mode.")
    if payload.get("provider") != expected_provider or payload.get("model_id") != expected_model_id:
        raise RuntimeError("CATCH gate is blocked: provider audit used a different provider or model.")
    evaluated = evaluate_mimo_provider_audit(
        payload.get("records") or [],
        expected_cache_namespace=expected_cache_namespace,
    )
    if not payload.get("passed") or not evaluated.get("passed"):
        raise RuntimeError(f"CATCH gate is blocked: provider audit failed: {evaluated.get('conditions', {})}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": True,
        "conditions": evaluated["conditions"],
    }


def _require_passing_gate(
    run_root: Path,
    *,
    experiment_name: str,
    phase_name: str,
    model_name: str,
    config_sha: str,
    frozen_sha: str,
) -> None:
    phase_root = run_root / experiment_name / phase_name
    candidates = (
        sorted(phase_root.glob("*/diagnostics/gate.json"), key=lambda path: path.stat().st_mtime, reverse=True)
        if phase_root.exists()
        else []
    )
    for gate_path in candidates:
        run_dir = gate_path.parents[1]
        manifest_path = run_dir / "manifest.json"
        validation_path = run_dir / "run_validation.json"
        if not manifest_path.exists() or not validation_path.exists():
            continue
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        validation = json.loads(validation_path.read_text(encoding="utf-8"))
        candidate_path = run_dir / "diagnostics" / "frozen_decoding_candidate.json"
        candidate_sha = ""
        if candidate_path.exists():
            candidate_sha = str(json.loads(candidate_path.read_text(encoding="utf-8")).get("sha256") or "")
        manifest_frozen_sha = str((manifest.get("frozen_decoding") or {}).get("sha256") or "")
        if (
            gate.get("passed")
            and validation.get("passed")
            and manifest.get("resolved_model", {}).get("name") == model_name
            and manifest.get("frozen_config_sha256") == config_sha
            and (candidate_sha == frozen_sha if phase_name == "development" else manifest_frozen_sha == frozen_sha)
        ):
            return
    raise RuntimeError(
        f"CATCH {phase_name} gate is required for this exact model, config, and frozen decoder."
    )


def _require_passing_human_audit(path: Path) -> None:
    if not path.exists():
        raise RuntimeError(f"CATCH confirmation is blocked: human validity audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    conditions = {
        "passed": bool(payload.get("passed")),
        "decidable_rate": float(payload.get("decidable_rate") or 0) >= 0.90,
        "entailment_rate": float(payload.get("entailment_rate") or 0) >= 0.90,
        "leakage_rate": float(payload.get("answer_leakage_rate") or 1) <= 0.05,
        "sample_count": int(payload.get("sample_count") or 0) >= 100,
        "annotator_count": int(payload.get("annotator_count") or 0) >= 2,
    }
    if not all(conditions.values()):
        raise RuntimeError(f"CATCH confirmation is blocked: human validity audit failed {conditions}.")
