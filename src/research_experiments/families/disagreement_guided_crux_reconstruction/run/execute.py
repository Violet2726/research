"""DGCR 实验编排；真实运行受门控并使用隔离缓存。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.provider_audit import evaluate_mimo_provider_audit
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import build_run_id, finalize_run_outputs
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.families.disagreement_guided_crux_reconstruction.config import (
    load_phase_benchmarks,
    load_protocol_config,
    phase_metadata,
)
from research_experiments.families.disagreement_guided_crux_reconstruction.run.sample import run_dgcr_sample
from research_experiments.families.disagreement_guided_crux_reconstruction.run.validate import validate_run
from research_experiments.families.disagreement_guided_crux_reconstruction.statistics import build_metrics, evaluate_gate
from research_experiments.core.data.datasets import select_samples
from research_experiments.workspace.layout import default_cache_root, default_runs_root


@dataclass(frozen=True)
class DgcrEndpoint:
    backbone: object
    provider: OpenAICompatibleProvider
    cache: object
    throttle: RequestThrottle
    cache_namespace: str


def run_experiment(experiment, phase_name: str, backbone, run_root: str | Path | None = None, cache_root: str | Path | None = None) -> Path:
    load_dotenv(".env.local", override=False)
    phase = phase_metadata(experiment, phase_name)
    run_root = Path(run_root or default_runs_root("disagreement_guided_crux_reconstruction"))
    cache_root = Path(cache_root or default_cache_root())
    protocol = load_protocol_config(experiment.protocol)
    if backbone.provider != "xiaomimimo":
        raise ValueError("DGCR is frozen to the audited xiaomimimo provider.")
    provider_audit = _require_passing_provider_audit(
        experiment.provider_audit_path,
        expected_cache_namespace=experiment.cache_namespaces["provider_audit"],
        expected_provider=backbone.provider,
        expected_model_id=backbone.model_id,
    )
    if phase_name == "heldout":
        _require_passing_development_gate(run_root, experiment.name, backbone.name, _frozen_config_sha(experiment))
    cache_namespace = experiment.cache_namespaces["development" if phase_name == "development" else "heldout"]
    provider = OpenAICompatibleProvider(backbone)
    router = RequestCacheRouter(cache_root, namespace=cache_namespace)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=experiment.max_concurrent_requests,
        requests_per_minute=experiment.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    layout = prepare_registered_run_layout("disagreement_guided_crux_reconstruction", run_root, experiment.name, phase_name, run_id)
    benchmarks = load_phase_benchmarks(experiment, phase_name)
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": "disagreement_guided_crux_reconstruction",
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "description": experiment.description,
            "resolved_model": asdict(backbone),
            "protocol": asdict(protocol),
            "global_seed": experiment.global_seed,
            "cache_namespace": cache_namespace,
            "request_source": "fresh_dgcr_confirmation_cache",
            "provider_audit": provider_audit,
            "frozen_config_sha256": _frozen_config_sha(experiment),
            "phase_metadata": phase,
            "benchmarks": [asdict(item) for item in benchmarks],
            "method_order": ["sc_5", "adaptive_sc_8", "dgcr"],
        },
        family_name="disagreement_guided_crux_reconstruction",
    )
    layout.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    turns: list[dict] = []
    routers: list[dict] = []
    predictions: list[dict] = []
    try:
        with layout.agent_turns.open("w", encoding="utf-8") as turns_handle, layout.router_decisions.open("w", encoding="utf-8") as routers_handle, layout.predictions.open("w", encoding="utf-8") as predictions_handle:
            for benchmark in benchmarks:
                split_name = str(phase["split_overrides"][benchmark.slug])
                cache = router.for_request_target(provider=backbone.provider, request_model=backbone.model_id, dataset=benchmark.slug)
                endpoint = DgcrEndpoint(backbone=backbone, provider=provider, cache=cache, throttle=throttle, cache_namespace=cache_namespace)
                for sample in select_samples(benchmark, split_name):
                    sample_turns, router_row, sample_predictions = run_dgcr_sample(
                        sample, run_id=run_id, split_name=split_name, experiment=experiment, protocol=protocol, endpoint=endpoint
                    )
                    for row in sample_turns:
                        turns.append(row)
                        turns_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    routers.append(router_row)
                    routers_handle.write(json.dumps(router_row, ensure_ascii=False) + "\n")
                    for row in sample_predictions:
                        predictions.append(row)
                        predictions_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        metrics = build_metrics(predictions)
        gate = evaluate_gate(phase_name=phase_name, predictions=predictions, turns=turns, routers=routers)
        layout.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.diagnostic_path("gate.json").write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.run_summary.write_text(json.dumps({"metrics": metrics, "gate": gate}, ensure_ascii=False, indent=2), encoding="utf-8")
        layout.report.write_text(_render_markdown(metrics, gate), encoding="utf-8")
        finalize_run_outputs(layout.root, validator=validate_run, validation_path=layout.validation)
        return layout.root
    finally:
        provider.close()
        router.close()


def _frozen_config_sha(experiment) -> str:
    payload = {"experiment": experiment.raw, "protocol": Path(experiment.protocol).read_text(encoding="utf-8")}
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def _require_passing_provider_audit(
    path: Path,
    *,
    expected_cache_namespace: str,
    expected_provider: str = "xiaomimimo",
    expected_model_id: str | None = None,
) -> dict:
    """Reject both phases unless the recorded live MiMo contract audit passes."""

    if not path.exists():
        raise RuntimeError(f"DGCR gate is blocked: required provider audit is missing at {path}.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("audit_kind") != "mimo_live_provider_contract_v1" or payload.get("cache_mode") != "bypassed":
        raise RuntimeError("DGCR gate is blocked: provider audit has the wrong contract or cache mode.")
    if payload.get("provider") != expected_provider or (expected_model_id and payload.get("model_id") != expected_model_id):
        raise RuntimeError("DGCR gate is blocked: provider audit was recorded for a different provider or model.")
    evaluated = evaluate_mimo_provider_audit(
        payload.get("records") or [],
        expected_cache_namespace=expected_cache_namespace,
    )
    if not payload.get("passed") or not evaluated.get("passed"):
        raise RuntimeError(f"DGCR gate is blocked: provider audit failed: {evaluated.get('conditions', {})}")
    return {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "passed": True,
        "conditions": evaluated["conditions"],
    }


def _require_passing_development_gate(run_root: Path, experiment_name: str, model_name: str, config_sha: str) -> None:
    root = run_root / experiment_name / "development"
    candidates = sorted(root.glob("*/diagnostics/gate.json"), key=lambda path: path.stat().st_mtime, reverse=True) if root.exists() else []
    for gate_path in candidates:
        manifest_path = gate_path.parents[2] / "manifest.json"
        gate = json.loads(gate_path.read_text(encoding="utf-8"))
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
        if gate.get("passed") and manifest.get("resolved_model", {}).get("name") == model_name and manifest.get("frozen_config_sha256") == config_sha:
            return
    raise RuntimeError("Held-out DGCR is blocked until this exact model and frozen config have a passing development gate.")


def _render_markdown(metrics: dict, gate: dict) -> str:
    lines = ["# DGCR run", "", f"Gate passed: `{gate.get('passed')}`", "", "| Method | Task harmonic | Mean tokens |", "|---|---:|---:|"]
    for row in metrics.get("summary", []):
        lines.append(f"| {row['method_name']} | {row['task_harmonic_accuracy']:.4f} | {row['mean_total_tokens']:.1f} |")
    return "\n".join(lines) + "\n"
