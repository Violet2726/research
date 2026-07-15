"""唯一活跃 H-SGSA v5 协议的流式实验执行器。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

from research_experiments.core.data.datasets import load_split_ids
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.risk_controlled_trace_mad.config import (
    HsgsaProtocolConfig,
    MadInnovationExperimentConfig,
    load_version_registry,
    phase_methods,
    require_active_version,
    runtime_for_provider,
)
from research_experiments.families.risk_controlled_trace_mad.prompts import (
    HSGSA_PROMPT_VERSION,
    HSGSA_REVIEW_SCHEMA_VERSION,
)
from research_experiments.families.risk_controlled_trace_mad.run.hsgsa_sample import (
    NetworkAttemptBudget,
    run_hsgsa_batch,
)
from research_experiments.families.risk_controlled_trace_mad.run.metrics import (
    build_diagnostics,
    build_metrics,
    evaluate_gate,
    write_paper_summary,
)
from research_experiments.families.risk_controlled_trace_mad.run.report import render_report, summarize_run
from research_experiments.families.risk_controlled_trace_mad.run.sample import (
    ModelEndpoint,
    load_selected_samples,
    resolve_split_name,
)
from research_experiments.families.risk_controlled_trace_mad.run.validate import validate_run
from research_experiments.families.risk_controlled_trace_mad.statistics import paired_statistics
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata, resolve_model
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.output_protocols import build_shared_output_protocol_diagnostics
from research_experiments.workspace.layout import default_cache_root, default_runs_root

FAMILY_NAME = "risk_controlled_trace_mad"


def run_hsgsa_experiment(
    experiment: MadInnovationExperimentConfig,
    phase_name: str,
    protocol: HsgsaProtocolConfig,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    resume_run_dir: str | Path | None = None,
    version: str | None = None,
) -> Path:
    load_dotenv(".env.local", override=False)
    phase = phase_metadata(experiment, phase_name)
    if bool(phase.get("replay_only")):
        raise ValueError(
            f"Phase {phase_name!r} is cache-replay-only; use the replay audit command and do not make live calls."
        )
    if resume_run_dir is not None:
        raise ValueError("H-SGSA v5 does not silently truncate or append a partial run; resume is not yet supported.")
    development_audit = _require_development_gate(phase)
    registry = load_version_registry(experiment.version_registry)
    version_record = require_active_version(registry, version)
    active_methods = phase_methods(experiment, phase_name)
    benchmarks = load_benchmarks(experiment)
    requested = set(map(str, phase.get("benchmark_slugs") or []))
    benchmarks = [benchmark for benchmark in benchmarks if not requested or benchmark.slug in requested]
    missing = requested - {benchmark.slug for benchmark in benchmarks}
    if missing:
        raise ValueError(f"Unknown benchmark(s): {sorted(missing)}")

    selected_by_dataset, split_audit = _select_confirmation_samples(experiment, phase_name, phase, benchmarks)
    sample_total = sum(len(rows) for rows in selected_by_dataset.values())
    planned_calls = sample_total * protocol.max_logical_calls
    if planned_calls > protocol.max_network_attempts:
        raise ValueError(
            f"Frozen confirmation plan requires {planned_calls} logical calls, above the "
            f"{protocol.max_network_attempts} hard network-attempt cap."
        )

    model = resolve_model(experiment.primary_model_ref)
    runtime = runtime_for_provider(experiment, model.provider)
    provider = OpenAICompatibleProvider(model)
    throttle = RequestThrottle.for_model(
        model,
        max_concurrent_requests=runtime.max_concurrent_requests,
        requests_per_minute=runtime.requests_per_minute_limit,
    )
    cache_router = RequestCacheRouter(cache_root or default_cache_root())
    network_budget = NetworkAttemptBudget(protocol.max_network_attempts)
    run_root = Path(run_root or default_runs_root(FAMILY_NAME))
    run_id = build_run_id("xiaomimimo-mimo-v2.5")
    paths = prepare_registered_run_layout(FAMILY_NAME, run_root, experiment.name, phase_name, run_id)
    progress = RunProgressTracker(
        paths.progress,
        planned_calls,
        sample_total * len(active_methods),
        planned_calls_are_upper_bound=True,
        target_network_rpm=runtime.requests_per_minute_limit,
    )
    method_descriptions = {
        method: str((experiment.raw.get("method_descriptions") or {}).get(method) or method)
        for method in active_methods
    }
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": FAMILY_NAME,
            "family_display_name": "Homogeneous Support-Blind SGSA",
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "experiment": experiment.name,
            "phase": phase_name,
            "description": experiment.description,
            "active_version": version_record.version_id,
            "version_record": asdict(version_record),
            "phase_metadata": phase,
            "development_gate_audit": development_audit,
            "protocol": asdict(protocol),
            "prompt_version": HSGSA_PROMPT_VERSION,
            "schema_version": HSGSA_REVIEW_SCHEMA_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "seed_schedule": {
                "stage_a": [experiment.global_seed + index for index in range(protocol.stage_candidates)],
                "resample": [experiment.global_seed + 10_000 + index for index in range(protocol.resample_candidates)],
                "blind_review": [experiment.global_seed + 20_000 + index for index in range(protocol.reviewer_count)],
            },
            "resolved_models": {"primary": asdict(model)},
            "runtime_profiles": {"primary": asdict(runtime)},
            "max_concurrent_requests": runtime.max_concurrent_requests,
            "requests_per_minute_limit": runtime.requests_per_minute_limit,
            "benchmarks": [asdict(item) for item in benchmarks],
            "dataset_order": [item.slug for item in benchmarks],
            "method_order": active_methods,
            "methods": active_methods,
            "method_descriptions": method_descriptions,
            "max_logical_calls_per_question": protocol.max_logical_calls,
            "max_network_attempts": protocol.max_network_attempts,
            "claim_scope": (
                "fixed MiMo-v2.5, fixed prompt, homogeneous sampling, and matched test-time budget; "
                "at most a held-out BBEH accuracy-cost Pareto claim"
            ),
            "total_planned_calls": planned_calls,
            "network_retry_reserve": protocol.max_network_attempts - planned_calls,
            "total_planned_predictions": sample_total * len(active_methods),
            "split_audit": split_audit,
            "streaming_jsonl": True,
            "completion_archive_format": "tar.zst",
            "no_interim_method_accuracy_inspection": bool(
                phase.get("no_interim_method_accuracy_inspection", False)
            ),
        },
        family_name=FAMILY_NAME,
    )
    paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2, default=str), encoding="utf-8")

    analysis_rows: list[dict[str, Any]] = []
    diagnostic_turn_rows: list[dict[str, Any]] = []
    try:
        with (
            paths.agent_turns.open("w", encoding="utf-8") as turn_handle,
            paths.debate_messages.open("w", encoding="utf-8") as message_handle,
            paths.router_decisions.open("w", encoding="utf-8") as decision_handle,
            paths.predictions.open("w", encoding="utf-8") as prediction_handle,
        ):
            turn_writer = BufferedJsonlWriter(turn_handle)
            message_writer = BufferedJsonlWriter(message_handle)
            decision_writer = BufferedJsonlWriter(decision_handle)
            prediction_writer = BufferedJsonlWriter(prediction_handle)
            for benchmark in benchmarks:
                endpoint = ModelEndpoint(
                    "mimo",
                    model,
                    provider,
                    cache_router.for_request_target(
                        provider=model.provider, request_model=model.model_id, dataset=benchmark.slug
                    ),
                    throttle,
                )
                split_name = resolve_split_name(experiment, phase_name, benchmark.slug)
                batch = run_hsgsa_batch(
                    run_id=run_id,
                    dataset=benchmark.slug,
                    split_name=split_name,
                    samples=selected_by_dataset[benchmark.slug],
                    experiment=experiment,
                    protocol=protocol,
                    active_methods=active_methods,
                    endpoint=endpoint,
                    max_concurrent_samples=max(1, runtime.max_concurrent_requests // protocol.max_logical_calls),
                    network_budget=network_budget,
                )
                for _, turns, messages, decisions, predictions in batch:
                    for row in turns:
                        turn_writer.write_row(row)
                        progress.record_call(row, method_key="method_name")
                        diagnostic_turn_rows.append(_compact_turn(row))
                    for row in messages:
                        message_writer.write_row(row)
                    for row in decisions:
                        decision_writer.write_row(row)
                    for row in predictions:
                        prediction_writer.write_row(row)
                        progress.record_predictions(1, str(row["dataset"]), str(row["method_name"]))
                        analysis_rows.append(_compact_prediction(row))

        metrics = build_metrics(
            analysis_rows,
            dataset_order=[item.slug for item in benchmarks],
            method_order=active_methods,
            bbeh_harmonic=True,
        )
        metrics["analysis_mask"] = split_audit
        diagnostics = build_diagnostics(analysis_rows)
        paired = paired_statistics(
            analysis_rows,
            reference="hsgsa_unanimous_3",
            competitors=[name for name in active_methods if name != "hsgsa_unanimous_3"],
            seed=experiment.global_seed,
            bbeh_harmonic=True,
        )
        protocol_diagnostics = build_shared_output_protocol_diagnostics(
            diagnostic_turn_rows,
            dataset_order=[item.slug for item in benchmarks],
            method_order=["hsgsa_stage_a_shared", "hsgsa_resample_shared", "hsgsa_blind_reviewer_shared"],
        )
        gate = evaluate_gate(phase_name, analysis_rows, paired, protocol.provider_abstention_limit)
        gate["actual_network_attempts"] = network_budget.actual
        gate["network_attempt_cap"] = protocol.max_network_attempts
        metrics["progression_gate"] = gate
        paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.hsgsa_diagnostics.write_text(
            json.dumps({**diagnostics, "progression_gate": gate}, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.paired_statistics.write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        paths.output_protocol_diagnostics.write_text(
            json.dumps(protocol_diagnostics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        paths.hsgsa_comparison.write_text(
            json.dumps(
                {
                    "reference_method": "hsgsa_unanimous_3",
                    "primary_comparison": "adaptive_sc_8",
                    "paired_statistics": paired,
                    "progression_gate": gate,
                    "claim_scope": manifest["claim_scope"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        write_paper_summary(paths.paper_summary, metrics)
        render_report(paths.root)
        paths.run_summary.write_text(
            json.dumps(summarize_run(paths.root), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        finalize_run_outputs(paths.root, validator=validate_run, validation_path=paths.validation)
        progress.mark_completed()
        return paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()


def _select_confirmation_samples(experiment, phase_name, phase, benchmarks):
    excluded_specs = dict(phase.get("exclude_splits") or {})
    selected: dict[str, list[Any]] = {}
    audit: dict[str, Any] = {"kind": "pre_registered_exclusion", "datasets": {}}
    for benchmark in benchmarks:
        split_name = resolve_split_name(experiment, phase_name, benchmark.slug)
        rows = load_selected_samples(benchmark, split_name)
        requested_ids = {sample.sample_id for sample in rows}
        exclusion_name = str(excluded_specs.get(benchmark.slug) or "")
        excluded_ids = set()
        if exclusion_name:
            excluded_ids = set(
                load_split_ids(
                    benchmark.cache_namespace or benchmark.slug,
                    exclusion_name,
                    random_seed=benchmark.random_seed,
                )
            )
            rows = [sample for sample in rows if sample.sample_id not in excluded_ids]
        included_ids = {sample.sample_id for sample in rows}
        if included_ids & excluded_ids:
            raise ValueError(f"Development/confirmation overlap detected for {benchmark.slug}.")
        tasks = sorted({str(sample.metadata.get("task") or "") for sample in rows if sample.metadata.get("task")})
        if benchmark.slug == "bbeh" and exclusion_name and len(tasks) != 23:
            raise ValueError(f"Held-out BBEH must retain all 23 tasks; found {len(tasks)}.")
        selected[benchmark.slug] = rows
        audit["datasets"][benchmark.slug] = {
            "source_split": split_name,
            "requested_count": len(requested_ids),
            "excluded_split": exclusion_name or None,
            "excluded_count": len(requested_ids & excluded_ids),
            "confirmation_count": len(included_ids),
            "overlap_count": len(included_ids & excluded_ids),
            "task_count": len(tasks),
            "task_names": tasks,
        }
    return selected, audit


def _require_development_gate(phase: dict[str, Any]) -> dict[str, Any] | None:
    audit_path = str(phase.get("required_development_audit") or "")
    if not audit_path:
        return None
    path = Path(audit_path)
    if not path.exists():
        raise ValueError(f"Required development audit does not exist: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not payload.get("passed"):
        raise ValueError(
            "H-SGSA confirmation is locked by the pre-registered stop rule; "
            f"development audit failed: {payload.get('failures', [])}"
        )
    return {
        "path": str(path),
        "passed": True,
        "audit_kind": payload.get("audit_kind"),
        "source_run": payload.get("source_run"),
    }


def _compact_prediction(row: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "dataset", "sample_id", "task", "method_name", "score", "total_tokens_per_question",
        "prompt_tokens_per_question", "completion_tokens_per_question", "latency_ms_per_question",
        "logical_calls_per_question", "network_attempts_per_question", "provider_abstentions_per_question",
        "protocol_failures_per_question", "request_failures_per_question", "triggered", "override_accepted",
        "corrected_by_debate", "harmed_by_debate", "vote_flipped", "candidate_oracle_correct", "novel_answer",
        "initial_vote_score", "shared_physical_network_attempts_per_question", "reviewer_calls_per_question",
        "reviewer_valid_picks_per_question", "reviewer_protocol_failures_per_question",
        "shared_physical_request_failures_per_question", "shared_physical_protocol_failures_per_question",
    }
    return {key: value for key, value in row.items() if key in keys}


def _compact_turn(row: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "dataset", "method_name", "output_status", "protocol_parse_status", "reason_present",
        "provider_abstention", "network_attempt_count", "total_tokens", "model_lineage",
    }
    return {key: value for key, value in row.items() if key in keys}
