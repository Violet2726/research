"""带隔离缓存、硬门控和可审计生命周期的 CATCH 执行器。"""

from __future__ import annotations

import hashlib
import json
from contextlib import suppress
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import load_split_ids, select_samples
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.families.contrastive_active_testing.config import (
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.contrastive_active_testing.prompts import (
    CATCH_PROMPT_VERSION,
    CATCH_SCHEMA_VERSION,
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
    cache: object
    throttle: RequestThrottle
    cache_namespace: str


def run_experiment(
    experiment,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    if backbone.provider != "xiaomimimo":
        raise ValueError("CATCH v1 is frozen to the audited xiaomimimo provider.")
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
    router = RequestCacheRouter(cache_root, namespace=cache_namespace)
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
    run_direct_judge = bool(phase.get("run_direct_judge", phase_name != "confirmation"))
    calls_per_triggered = 18 if phase_name == "development" else 14 if run_direct_judge else 11
    predictions_per_sample = 9 if phase_name == "development" else 4 if run_direct_judge else 3
    progress = RunProgressTracker(
        layout.progress,
        total_planned_calls=sample_count * calls_per_triggered,
        total_planned_predictions=sample_count * predictions_per_sample,
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
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "description": experiment.description,
            "resolved_model": asdict(backbone),
            "protocol": asdict(protocol),
            "prompt_version": CATCH_PROMPT_VERSION,
            "schema_version": CATCH_SCHEMA_VERSION,
            "global_seed": experiment.global_seed,
            "cache_namespace": cache_namespace,
            "request_source": "fresh_catch_confirmation_cache",
            "provider_audit": provider_audit,
            "frozen_config_sha256": config_sha,
            "frozen_decoding": frozen_decoding,
            "phase_metadata": phase,
            "benchmarks": [asdict(item) for item in benchmarks],
            "sample_count": sample_count,
            "method_order": ["sc_5", "adaptive_sc_8", "catch", "direct_judge_3"],
            "max_network_attempts": protocol.max_network_attempts,
            "calls_per_triggered_question_upper_bound": calls_per_triggered,
            "dgcr_predecessor_status": "retired_exact_span_reconstruction_channel_failed",
        },
        family_name="contrastive_active_testing",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    turns: list[dict[str, Any]] = []
    routers: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    try:
        with (
            layout.agent_turns.open("w", encoding="utf-8") as turns_handle,
            layout.router_decisions.open("w", encoding="utf-8") as routers_handle,
        ):
            for benchmark in benchmarks:
                split_name = str(phase["split_overrides"][benchmark.slug])
                cache = router.for_request_target(
                    provider=backbone.provider,
                    request_model=backbone.model_id,
                    dataset=benchmark.slug,
                )
                endpoint = CatchEndpoint(
                    backbone=backbone,
                    provider=provider,
                    cache=cache,
                    throttle=throttle,
                    cache_namespace=cache_namespace,
                )
                for sample in selected_by_benchmark[benchmark.slug]:
                    sample_turns, router_row, sample_predictions = run_catch_sample(
                        sample,
                        run_id=run_id,
                        split_name=split_name,
                        experiment=experiment,
                        protocol=protocol,
                        endpoint=endpoint,
                        network_budget=network_budget,
                        phase_name=phase_name,
                        frozen_decoding=frozen_decoding,
                        run_direct_judge=run_direct_judge,
                    )
                    for row in sample_turns:
                        turns.append(row)
                        turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        progress.record_call(row)
                    routers.append(router_row)
                    routers_handle.write(json.dumps(router_row, ensure_ascii=False) + "\n")
                    predictions.extend(sample_predictions)
                    progress.record_predictions(len(sample_predictions), sample.dataset, "catch_sample")

        development_selection = None
        if phase_name == "development":
            predictions, development_selection = materialize_development_catch(predictions, routers)
        with layout.predictions.open("w", encoding="utf-8") as predictions_handle:
            for row in predictions:
                predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = build_metrics(predictions)
        gate = evaluate_gate(
            phase_name=phase_name,
            predictions=predictions,
            turns=turns,
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
                {"metrics": metrics, "gate": gate, "development_selection": development_selection},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        render_report(layout.root)
        finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_completed()
        return layout.root
    except BaseException as exc:
        if not layout.validation.exists():
            with suppress(BaseException):
                finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        progress.mark_failed(type(exc).__name__, str(exc), last_sample_id=progress.last_sample_id)
        raise
    finally:
        progress.close()
        provider.close()
        router.close()


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
        "decoder_version": "catch_ecoc_decoder_v1",
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _build_frozen_decoding_candidate(*, run_id: str, config_sha: str, selection: dict[str, Any]) -> dict[str, Any]:
    winner = dict(selection["selected"])
    payload = {
        "freeze_kind": "catch_decoding_v1",
        "source_development_run_id": run_id,
        "source_config_sha256": config_sha,
        "d_min": int(winner["d_min"]),
        "margin": int(winner["margin"]),
        "selection_constraints_passed": bool(selection.get("positive_constraints_satisfied")),
        "prompt_version": CATCH_PROMPT_VERSION,
        "schema_version": CATCH_SCHEMA_VERSION,
        "decoder_version": "catch_ecoc_decoder_v1",
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
