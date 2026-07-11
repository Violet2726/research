"""BRD-MAD 的实验编排。"""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from dotenv import load_dotenv

from research_experiments.core.controls.no_comm_controls import run_unified_control_batch
from research_experiments.core.execution.artifacts import BufferedJsonlWriter
from research_experiments.core.execution.cache import RequestCacheRouter
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runtime import RunProgressTracker, build_run_id, finalize_run_outputs
from research_experiments.core.structured_outputs import ARTIFACT_VERSION
from research_experiments.families.blind_reconstructive_mad.config import (
    BrdMadExperimentConfig,
    load_control_catalog,
    load_protocol_config,
    phase_brd_methods,
    runtime_for_provider,
)
from research_experiments.families.blind_reconstructive_mad.pilot_gate import (
    evaluate_pilot_gate,
    find_passing_pilot_gate,
)
from research_experiments.families.blind_reconstructive_mad.prompts import (
    BRD_PROMPT_VERSION,
    SGSA_PROMPT_VERSION,
)
from research_experiments.families.blind_reconstructive_mad.run.report import render_report, summarize_run
from research_experiments.families.blind_reconstructive_mad.run.sample import (
    _execute_turn,
    append_outputs,
    apply_bbeh_harmonic_primary,
    build_brd_diagnostics,
    build_control_prediction_row,
    build_metrics,
    control_results_as_brd_shape,
    estimate_work,
    load_selected_samples,
    resolve_split_name,
    run_brd_batch,
)
from research_experiments.families.blind_reconstructive_mad.run.validate import validate_run
from research_experiments.families.blind_reconstructive_mad.statistics import paired_method_statistics
from research_experiments.family_runtime.config_helpers import load_benchmarks, phase_metadata
from research_experiments.family_runtime.layout import prepare_registered_run_layout
from research_experiments.family_runtime.manifest import finalize_family_manifest
from research_experiments.family_runtime.output_protocols import build_shared_output_protocol_diagnostics
from research_experiments.workspace.layout import default_cache_root, default_runs_root


def run_experiment(
    experiment: BrdMadExperimentConfig,
    phase_name: str,
    backbone,
    run_root: str | Path | None = None,
    cache_root: str | Path | None = None,
    *,
    family_name: str = "blind_reconstructive_mad",
    display_name: str = "BRD-MAD",
) -> Path:
    load_dotenv(".env.local", override=False)
    run_root = run_root or default_runs_root(family_name)
    cache_root = cache_root or default_cache_root()
    benchmarks = load_benchmarks(experiment)
    requested_benchmarks = phase_metadata(experiment, phase_name).get("benchmark_slugs")
    if requested_benchmarks:
        requested = {str(item) for item in requested_benchmarks}
        benchmarks = [benchmark for benchmark in benchmarks if benchmark.slug in requested]
        missing = requested - {benchmark.slug for benchmark in benchmarks}
        if missing:
            raise ValueError(f"Unknown phase benchmark slug(s): {sorted(missing)}")
    protocol = load_protocol_config(experiment.protocol)
    controls = load_control_catalog(experiment.control_catalog)
    _validate_controls(experiment, controls)
    active_methods = phase_brd_methods(experiment, phase_name)
    runtime = runtime_for_provider(experiment, backbone.provider)
    if family_name == "selective_gsa_mad":
        _assert_full_after_count100_gate(experiment, phase_name, backbone.name, run_root)
    else:
        _assert_locked_pilot_gate(experiment, phase_name, backbone.name, run_root)
    provider = OpenAICompatibleProvider(backbone)
    cache_router = RequestCacheRouter(cache_root)
    throttle = RequestThrottle.for_model(
        backbone,
        max_concurrent_requests=runtime.max_concurrent_requests,
        requests_per_minute=runtime.requests_per_minute_limit,
    )
    run_id = build_run_id(backbone.name)
    run_paths = prepare_registered_run_layout(family_name, run_root, experiment.name, phase_name, run_id)
    total_calls, total_predictions = estimate_work(experiment, phase_name, benchmarks, controls, protocol, active_methods)
    progress = RunProgressTracker(
        run_paths.progress,
        total_calls,
        total_predictions,
        planned_calls_are_upper_bound=True,
        target_network_rpm=runtime.requests_per_minute_limit,
        rate_limit_snapshot_provider=throttle.snapshot,
    )
    manifest = finalize_family_manifest(
        {
            "run_id": run_id,
            "created_at": datetime.now(UTC).isoformat(),
            "family_name": family_name,
            "family_display_name": display_name,
            "experiment_name": experiment.name,
            "phase_name": phase_name,
            "primary_model_ref": experiment.primary_model_ref,
            "resolved_model": asdict(backbone),
            "experiment": experiment.name,
            "description": experiment.description,
            "phase": phase_name,
            "phase_metadata": phase_metadata(experiment, phase_name),
            "protocol": asdict(protocol),
            "control_prompt_version": experiment.control_prompt_version,
            "review_prompt_version": (
                SGSA_PROMPT_VERSION if family_name == "selective_gsa_mad" else BRD_PROMPT_VERSION
            ),
            "candidate_board_version": "balanced_all_candidates_v1",
            "output_protocol": experiment.output_protocol,
            "artifact_version": ARTIFACT_VERSION,
            "global_seed": experiment.global_seed,
            "max_concurrent_requests": runtime.max_concurrent_requests,
            "requests_per_minute_limit": runtime.requests_per_minute_limit,
            "runtime_profile_provider": backbone.provider,
            "benchmarks": [asdict(item) for item in benchmarks],
            "dataset_order": [item.slug for item in benchmarks],
            "control_method_names": experiment.control_methods,
            "control_methods": {name: asdict(controls[name]) for name in experiment.control_methods},
            "brd_methods": active_methods,
            "method_order": [*experiment.control_methods, *active_methods],
            "total_planned_calls": total_calls,
            "total_planned_predictions": total_predictions,
        },
        family_name=family_name,
    )
    run_paths.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    all_turns: list[dict[str, object]] = []
    all_debate_rows: list[dict[str, object]] = []
    all_router_rows: list[dict[str, object]] = []
    all_predictions: list[dict[str, object]] = []
    try:
        with (
            run_paths.agent_turns.open("w", encoding="utf-8") as turns_handle,
            run_paths.debate_messages.open("w", encoding="utf-8") as messages_handle,
            run_paths.router_decisions.open("w", encoding="utf-8") as routers_handle,
            run_paths.predictions.open("w", encoding="utf-8") as predictions_handle,
        ):
            turn_writer = BufferedJsonlWriter(turns_handle)
            debate_writer = BufferedJsonlWriter(messages_handle)
            router_writer = BufferedJsonlWriter(routers_handle)
            prediction_writer = BufferedJsonlWriter(predictions_handle)
            for benchmark in benchmarks:
                cache = cache_router.for_request_target(provider=backbone.provider, request_model=backbone.model_id, dataset=benchmark.slug)
                split_name = resolve_split_name(experiment, phase_name, benchmark.slug)
                samples = load_selected_samples(benchmark, split_name)
                append_outputs(
                    sample_results=run_brd_batch(
                        run_id=run_id,
                        benchmark_slug=benchmark.slug,
                        split_name=split_name,
                        samples=samples,
                        experiment=experiment,
                        protocol=protocol,
                        active_methods=active_methods,
                        backbone=backbone,
                        provider=provider,
                        cache=cache,
                        throttle=throttle,
                    ),
                    dataset_slug=benchmark.slug,
                    progress=progress,
                    turn_writer=turn_writer,
                    debate_writer=debate_writer,
                    router_writer=router_writer,
                    prediction_writer=prediction_writer,
                    all_turns=all_turns,
                    all_debate_rows=all_debate_rows,
                    all_router_rows=all_router_rows,
                    all_predictions=all_predictions,
                )
                for control_name in experiment.control_methods:
                    append_outputs(
                        sample_results=control_results_as_brd_shape(
                            run_unified_control_batch(
                                run_id=run_id,
                                samples=samples,
                                control_name=control_name,
                                method=controls[control_name],
                                benchmark_slug=benchmark.slug,
                                split_name=split_name,
                                backbone=backbone,
                                provider=provider,
                                cache=cache,
                                throttle=throttle,
                                global_seed=experiment.global_seed,
                                max_concurrent_requests=runtime.max_concurrent_requests,
                                execute_turn=_execute_turn,
                                build_prediction_row=build_control_prediction_row,
                                prompt_version=experiment.control_prompt_version,
                            )
                        ),
                        dataset_slug=benchmark.slug,
                        progress=progress,
                        turn_writer=turn_writer,
                        debate_writer=debate_writer,
                        router_writer=router_writer,
                        prediction_writer=prediction_writer,
                        all_turns=all_turns,
                        all_debate_rows=all_debate_rows,
                        all_router_rows=all_router_rows,
                        all_predictions=all_predictions,
                    )
        method_order = [*experiment.control_methods, *active_methods]
        use_bbeh_harmonic = _use_bbeh_harmonic(experiment, phase_name)
        metrics = apply_bbeh_harmonic_primary(
            build_metrics(
                all_predictions,
                dataset_order=[item.slug for item in benchmarks],
                method_order=method_order,
                control_names=experiment.control_methods,
            ),
            all_predictions,
            control_names=experiment.control_methods,
            use_harmonic=use_bbeh_harmonic,
        )
        diagnostics = build_brd_diagnostics(all_predictions, dataset_order=[item.slug for item in benchmarks], method_order=method_order)
        paired = paired_method_statistics(
            all_predictions,
            reference_method="sgsa_unanimous_3" if family_name == "selective_gsa_mad" else "brd_quorum_3",
            seed=experiment.global_seed,
            bbeh_harmonic=use_bbeh_harmonic,
        )
        output_protocol = build_shared_output_protocol_diagnostics(all_turns, dataset_order=[item.slug for item in benchmarks], method_order=["brd_stage_a_shared", *method_order])
        run_paths.metrics.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("brd_diagnostics.json").write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("paired_statistics.json").write_text(json.dumps(paired, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.diagnostic_path("output_protocol_diagnostics.json").write_text(json.dumps(output_protocol, ensure_ascii=False, indent=2), encoding="utf-8")
        gate_evaluator = evaluate_pilot_gate
        gate_filename = "pilot_gate.json"
        if family_name == "selective_gsa_mad":
            from research_experiments.family_runtime.sgsa_bridge import evaluate_count100_gate as gate_evaluator

            gate_filename = "count100_gate.json"

        gate = gate_evaluator(
            prediction_rows=all_predictions,
            turn_rows=all_turns,
            diagnostics=diagnostics,
            model_name=backbone.name,
        )
        run_paths.diagnostic_path(gate_filename).write_text(json.dumps(gate, ensure_ascii=False, indent=2), encoding="utf-8")
        run_paths.run_summary.write_text(json.dumps(summarize_run(run_paths.root, family_name=family_name), ensure_ascii=False, indent=2), encoding="utf-8")
        render_report(run_paths.root, family_name=family_name, display_name=display_name)
        finalize_run_outputs(run_paths.root, validator=lambda path: validate_run(path, family_name=family_name), validation_path=run_paths.validation)
        progress.mark_completed()
        return run_paths.root
    finally:
        progress.close()
        provider.close()
        cache_router.close()


def _validate_controls(experiment: BrdMadExperimentConfig, controls) -> None:
    missing = [name for name in experiment.control_methods if name not in controls]
    if missing:
        raise RuntimeError("Unknown BRD-MAD control methods: " + ", ".join(missing))
    if "sc_5" not in experiment.control_methods:
        raise RuntimeError("BRD-MAD requires sc_5 as the matched Stage-A control.")


def _use_bbeh_harmonic(experiment: BrdMadExperimentConfig, phase_name: str) -> bool:
    """Use task harmonic only for the canonical BBEH full split."""

    configured = str(phase_metadata(experiment, phase_name).get("bbeh_metric") or "").strip().lower()
    if configured in {"micro", "micro_accuracy", "official_mini_micro"}:
        return False
    if configured in {"task_harmonic", "task_harmonic_accuracy"}:
        return True
    return resolve_split_name(experiment, phase_name, "bbeh").startswith("full")


def _assert_locked_pilot_gate(experiment: BrdMadExperimentConfig, phase_name: str, model_name: str, run_root: str | Path) -> None:
    phase = phase_metadata(experiment, phase_name)
    if not phase.get("locked_after_pilot_gate"):
        return
    gate = find_passing_pilot_gate(
        family_run_root=run_root,
        model_name=model_name,
        experiment_name=str(phase.get("pilot_gate_experiment") or "brd_mad_pilot"),
    )
    if gate is None:
        raise RuntimeError(
            "Locked SGSA/BRD run blocked: no passing pilot gate was found for this exact backbone. "
            "Run and validate the configured pilot experiment first; do not tune the locked configuration."
        )


def _assert_full_after_count100_gate(
    experiment: BrdMadExperimentConfig,
    phase_name: str,
    model_name: str,
    run_root: str | Path,
) -> None:
    """Require a passing count100_seed42 gate before an SGSA full run."""

    phase = phase_metadata(experiment, phase_name)
    if not phase.get("full_after_count100_gate"):
        return
    experiment_name = str(phase.get("count100_gate_experiment") or experiment.name)
    root = Path(run_root) / experiment_name / "count100_seed42"
    candidates = sorted(root.glob("*/diagnostics/count100_gate.json"), key=lambda path: path.stat().st_mtime, reverse=True)
    for path in candidates:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("model_name") == model_name and payload.get("passed") is True:
            return
    raise RuntimeError(
        "Full SGSA run blocked: no passing count100_seed42 gate was found for this exact backbone. "
        "Run and validate count100_seed42 first; do not tune the full configuration."
    )
