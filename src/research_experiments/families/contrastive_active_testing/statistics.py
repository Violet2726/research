"""CATCH 指标、开发集全局冻结选择与预注册门控。"""

from __future__ import annotations

import hashlib
import math
from collections import defaultdict
from typing import Any

from scipy.stats import beta

from research_experiments.reporting.paired_inference import paired_statistics

BASE_METHODS = (
    "sc_5",
    "adaptive_sc_8",
    "catch",
    "catch_cert",
    "catch_cert_v2",
    "catch_kernel",
    "direct_judge_3",
    "pair_judge_3",
)


def materialize_development_catch(
    predictions: list[dict[str, Any]],
    routers: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Select one global grid cell and materialize it as the primary CATCH method."""

    variants = sorted(
        {str(row.get("method_name")) for row in predictions if str(row.get("method_name", "")).startswith("catch_d")}
    )
    candidates = []
    triggered_count = sum(bool(row.get("triggered")) for row in routers)
    for method_name in variants:
        rows = [row for row in predictions if row.get("method_name") == method_name]
        d_min, margin = _parse_variant(method_name)
        overrides = [row for row in rows if row.get("override_accepted")]
        corrected = sum(bool(row.get("corrected_by_debate")) for row in rows)
        harmed = sum(bool(row.get("harmed_by_debate")) for row in rows)
        precision = _ratio(sum(float(row.get("score") or 0) == 1.0 for row in overrides), len(overrides))
        eligible = 0
        for router in routers:
            variant = _router_variant(router, d_min=d_min, margin=margin)
            if variant and any(int(value) >= d_min for value in dict(variant.get("pair_distances") or {}).values()):
                eligible += 1
        coverage = _ratio(eligible, triggered_count)
        constraints = {
            "override_precision_at_least_65_percent": precision >= 0.65,
            "net_corrections_at_least_three": corrected - harmed >= 3,
            "code_packet_coverage_at_least_40_percent": coverage >= 0.40,
        }
        candidates.append(
            {
                "method_name": method_name,
                "d_min": d_min,
                "margin": margin,
                "corrected": corrected,
                "harmed": harmed,
                "net_corrections": corrected - harmed,
                "override_count": len(overrides),
                "override_precision": precision,
                "code_packet_coverage": coverage,
                "selection_constraints": constraints,
                "selection_constraints_passed": all(constraints.values()),
                "micro_accuracy": _ratio(sum(float(row.get("score") or 0) for row in rows), len(rows)),
            }
        )
    eligible_candidates = [row for row in candidates if row["selection_constraints_passed"]]
    pool = eligible_candidates or candidates
    if not pool:
        raise ValueError("Development predictions contain no CATCH grid variants.")
    winner = max(
        pool,
        key=lambda row: (
            int(row["net_corrections"]),
            float(row["override_precision"]),
            int(row["d_min"]),
            int(row["margin"]),
            _stable_tie_rank(str(row["method_name"])),
        ),
    )
    selected_rows = []
    for row in predictions:
        if row.get("method_name") != winner["method_name"]:
            selected_rows.append(row)
            continue
        primary = dict(row)
        primary["selected_grid_method_name"] = winner["method_name"]
        primary["method_name"] = "catch"
        selected_rows.append(primary)
    selection = {
        "selection_rule": "max_net_subject_to_precision_net_coverage_then_precision_dmin_margin_hash",
        "selected": winner,
        "candidates": candidates,
        "positive_constraints_satisfied": bool(eligible_candidates),
    }
    return selected_rows, selection


def build_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    available = {str(row.get("method_name")) for row in predictions}
    ordered = [method for method in BASE_METHODS if method in available]
    variants = sorted(method for method in available if method.startswith("catch_d"))
    summaries = [
        _summary(method, [row for row in predictions if row.get("method_name") == method])
        for method in [*ordered, *variants]
    ]
    reference = (
        "catch_kernel"
        if "catch_kernel" in available
        else "catch_cert_v2"
        if "catch_cert_v2" in available
        else "catch_cert"
        if "catch_cert" in available
        else "catch"
    )
    paired_competitors = [
        method
        for method in (
            "adaptive_sc_8",
            "pair_judge_3",
            "direct_judge_3",
            "sc_5",
            "catch",
            "catch_cert",
            "catch_cert_v2",
            "catch_kernel",
        )
        if method in available and method != reference
    ]
    paired = (
        paired_statistics(
            predictions,
            reference=reference,
            competitors=paired_competitors,
            seed=42,
            bootstrap_samples=10_000,
            bbeh_harmonic=False,
        )
        if reference in available and paired_competitors
        else {"reference_method": reference, "tests": []}
    )
    return {"summary": summaries, "paired_statistics": paired}


def build_best_effort_diagnostics(
    *,
    predictions: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    planned_by_dataset: dict[str, int],
) -> dict[str, Any]:
    """Build denominator-aware result and failure diagnostics without gates."""

    datasets: dict[str, Any] = {}
    for dataset in sorted(
        set(planned_by_dataset)
        | {str(row.get("dataset") or "") for row in predictions if row.get("dataset")}
        | {str(row.get("dataset") or "") for row in routers if row.get("dataset")}
    ):
        planned = int(planned_by_dataset.get(dataset, 0))
        dataset_predictions = [row for row in predictions if row.get("dataset") == dataset]
        dataset_routers = [row for row in routers if row.get("dataset") == dataset]
        methods: dict[str, Any] = {}
        by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in dataset_predictions:
            by_method[str(row.get("method_name") or "unknown")].append(row)
        for method, rows in sorted(by_method.items()):
            evaluable = [row for row in rows if row.get("prediction") not in {None, ""}]
            score_sum = sum(float(row.get("score") or 0) for row in evaluable)
            overrides = [row for row in evaluable if row.get("override_accepted")]
            correct_overrides = sum(float(row.get("score") or 0) == 1.0 for row in overrides)
            methods[method] = {
                **_summary(method, evaluable),
                "planned": planned,
                "available_prediction_rows": len(rows),
                "evaluable": len(evaluable),
                "missing": max(0, planned - len(evaluable)),
                "complete_case_accuracy": _ratio(score_sum, len(evaluable)),
                "conservative_accuracy_missing_as_wrong": _ratio(score_sum, planned),
                "override_count": len(overrides),
                "override_precision": _ratio(correct_overrides, len(overrides)),
                "corrected": sum(bool(row.get("corrected_by_debate")) for row in evaluable),
                "harmed": sum(bool(row.get("harmed_by_debate")) for row in evaluable),
                "candidate_oracle": _ratio(
                    sum(bool(row.get("candidate_oracle_correct")) for row in evaluable),
                    len(evaluable),
                ),
                "target_oracle": _ratio(
                    sum(bool(row.get("target_oracle_correct")) for row in evaluable),
                    len(evaluable),
                ),
                "certificate_coverage": _ratio(
                    sum(float(row.get("certificate_coverage") or 0) for row in evaluable),
                    len(evaluable),
                ),
            }
        primary_method = (
            "catch_kernel"
            if "catch_kernel" in by_method
            else "catch_cert_v2"
            if "catch_cert_v2" in by_method
            else "catch_cert"
            if "catch_cert" in by_method
            else "catch"
        )
        catch_by_id = {
            str(row.get("sample_id")): row
            for row in by_method.get(primary_method, [])
            if row.get("prediction") not in {None, ""}
        }
        paired: dict[str, Any] = {}
        for competitor in ("adaptive_sc_8", "pair_judge_3", "direct_judge_3", "sc_5"):
            other = {
                str(row.get("sample_id")): row
                for row in by_method.get(competitor, [])
                if row.get("prediction") not in {None, ""}
            }
            common = sorted(set(catch_by_id) & set(other))
            if common:
                paired[competitor] = {
                    "paired_sample_count": len(common),
                    "catch_accuracy": _ratio(
                        sum(float(catch_by_id[key].get("score") or 0) for key in common), len(common)
                    ),
                    "competitor_accuracy": _ratio(
                        sum(float(other[key].get("score") or 0) for key in common), len(common)
                    ),
                }
        datasets[dataset] = {
            "planned": planned,
            "attempted": len(dataset_routers),
            "sample_errors": sum(bool(row.get("sample_error")) for row in dataset_routers),
            "methods": methods,
            "paired_complete_cases": paired,
        }

    failure_counts: dict[tuple[str, str, str], int] = defaultdict(int)
    failure_examples: dict[tuple[str, str, str], str] = {}
    for row in turns:
        if not row.get("request_error") and row.get("protocol_parse_status") != "failed":
            continue
        error = str(row.get("request_error") or row.get("protocol_parse_error") or "unknown_failure")
        key = (
            str(row.get("dataset") or "unknown"),
            str(row.get("role") or "unknown"),
            _failure_category(error, request_error=bool(row.get("request_error"))),
        )
        failure_counts[key] += 1
        failure_examples.setdefault(key, error)
    total_calls = len(turns)
    request_failures = sum(bool(row.get("request_error")) for row in turns)
    parse_failures = sum(row.get("protocol_parse_status") == "failed" for row in turns)
    failure_rate = _ratio(request_failures, total_calls)
    lower, upper = _wilson_interval(request_failures, total_calls)
    triggered = [row for row in routers if row.get("triggered")]
    eligible = [row for row in triggered if row.get("eligible_challengers")]
    return {
        "datasets": datasets,
        "failures": {
            "logical_call_count": total_calls,
            "request_failure_count": request_failures,
            "request_failure_rate": failure_rate,
            "request_failure_rate_wilson_95": [lower, upper],
            "parse_failure_count": parse_failures,
            "by_dataset_role_error": [
                {
                    "dataset": key[0],
                    "role": key[1],
                    "error_type": key[2],
                    "example": failure_examples[key],
                    "count": count,
                }
                for key, count in sorted(failure_counts.items())
            ],
        },
        "mechanism": {
            "triggered_sample_count": len(triggered),
            "eligible_sample_count": len(eligible),
            "eligible_rate": _ratio(len(eligible), len(triggered)),
            "panel_false_pass_dependence": _v3_panel_dependence(routers),
            "witness_position_and_agreement": _v3_observation_diagnostics(routers, turns),
            "certificate_v2": _certificate_v2_diagnostics(routers),
            "kernel": _kernel_diagnostics(routers),
        },
        "costs": {
            "logical_calls": total_calls,
            "cache_hits": sum(bool(row.get("cache_hit")) for row in turns),
            "physical_network_attempts": sum(int(row.get("network_attempt_count") or 0) for row in turns),
            "retry_attempts": sum(
                max(0, int(row.get("network_attempt_count") or 0) - 1)
                for row in turns
                if int(row.get("network_attempt_count") or 0) > 0
            ),
            "actual_total_tokens": sum(float(row.get("actual_total_tokens") or 0) for row in turns),
            "mean_latency_ms": _ratio(sum(float(row.get("latency_ms") or 0) for row in turns), len(turns)),
        },
    }


def _failure_category(message: str, *, request_error: bool) -> str:
    text = message.casefold()
    if not request_error:
        return f"parse:{message[:80]}"
    if "timeout" in text or "timed out" in text:
        return "request:timeout"
    if "429" in text or "rate limit" in text:
        return "request:http_429"
    if any(f"http {code}" in text for code in range(500, 600)):
        return "request:http_5xx"
    if any(f"http {code}" in text for code in range(400, 500)):
        return "request:http_4xx"
    if "connection" in text or "transport" in text or "network" in text:
        return "request:connection"
    if "content filter" in text or "content_filter" in text:
        return "request:content_filter"
    return "request:other"


def evaluate_gate(
    *,
    phase_name: str,
    predictions: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    development_selection: dict[str, Any] | None = None,
    protocol_version: str = "catch_v2",
) -> dict[str, Any]:
    summaries = {row["method_name"]: row for row in build_metrics(predictions)["summary"]}
    sc5 = summaries.get("sc_5", {})
    adaptive = summaries.get("adaptive_sc_8", {})
    primary_method = (
        "catch_kernel"
        if protocol_version == "catch_kernel_v1"
        else "catch_cert_v2"
        if protocol_version == "catch_cert_v2"
        else "catch_cert"
        if protocol_version == "catch_cert_v1"
        else "catch"
    )
    catch = summaries.get(primary_method, {})
    judge = summaries.get("direct_judge_3", {})
    pair_judge = summaries.get("pair_judge_3", {})
    catch_rows = [row for row in predictions if row.get("method_name") == primary_method]
    triggered = [row for row in routers if row.get("triggered")]
    selected_d_min = int(catch_rows[0].get("d_min") or 0) if catch_rows else 0
    selected_margin = int(catch_rows[0].get("margin") or 0) if catch_rows else 0
    if protocol_version in {"catch_v3", "catch_cert_v1", "catch_cert_v2", "catch_kernel_v1"}:
        eligible = sum(bool(router.get("eligible_challengers")) for router in triggered)
    else:
        eligible = 0
        for router in triggered:
            variant = _router_variant(router, d_min=selected_d_min, margin=selected_margin)
            if variant and any(
                int(distance) >= selected_d_min for distance in dict(variant.get("pair_distances") or {}).values()
            ):
                eligible += 1
    code_coverage = _ratio(eligible, len(triggered))
    overrides = [row for row in catch_rows if row.get("override_accepted")]
    correct_overrides = [row for row in overrides if float(row.get("score") or 0) == 1.0]
    precision = _ratio(len(correct_overrides), len(overrides))
    corrected = sum(bool(row.get("corrected_by_debate")) for row in catch_rows)
    harmed = sum(bool(row.get("harmed_by_debate")) for row in catch_rows)
    panel_dependence = _v3_panel_dependence(routers)
    observation_diagnostics = _v3_observation_diagnostics(routers, turns)
    structured_turns = [
        row
        for row in turns
        if row.get("role")
        in {
            "test_designer",
            "blinded_witness",
            "direct_judge",
            "icv_selector",
            "icv_witness",
            "certificate_designer",
            "certificate_verifier",
            "certificate_designer_v2",
            "certificate_verifier_v2",
            "kernel_obligation_filler",
            "kernel_atomic_verifier",
            "pair_judge",
        }
    ]
    parse_rate = _ratio(
        sum(row.get("protocol_parse_status") == "ok" for row in structured_turns),
        len(structured_turns),
    )
    all_request_success = all(not row.get("request_error") for row in turns)
    all_actual_tokens_reported = bool(turns) and all(
        row.get("usage_source") == "reported"
        and row.get("actual_total_tokens") is not None
        and row.get("reasoning_tokens") is not None
        for row in turns
    )
    common = {
        "zero_terminal_request_failures": all_request_success,
        "all_turns_report_actual_tokens": all_actual_tokens_reported,
        "structured_parse_rate_at_least_99_5_percent": parse_rate >= 0.995,
        "actual_tokens_not_above_adaptive": float(catch.get("mean_total_tokens") or 0)
        <= float(adaptive.get("mean_total_tokens") or 0),
    }
    if phase_name == "development" and protocol_version == "catch_v3":
        strongest_judge = max(
            float(judge.get("micro_accuracy") or 0),
            float(pair_judge.get("micro_accuracy") or 0),
        )
        conditions = {
            **common,
            "code_packet_on_40_percent_of_disagreements": code_coverage >= 0.40,
            "candidate_oracle_micro_at_least_5pp_over_sc5": float(catch.get("candidate_oracle_micro") or 0)
            - float(sc5.get("micro_accuracy") or 0)
            >= 0.05,
            "target_oracle_micro_at_least_8pp_over_sc5": float(catch.get("target_oracle_micro") or 0)
            - float(sc5.get("micro_accuracy") or 0)
            >= 0.08,
            "target_oracle_micro_at_least_5pp_over_adaptive": float(catch.get("target_oracle_micro") or 0)
            - float(adaptive.get("micro_accuracy") or 0)
            >= 0.05,
            "catch_micro_at_least_3pp_over_adaptive": float(catch.get("micro_accuracy") or 0)
            - float(adaptive.get("micro_accuracy") or 0)
            >= 0.03,
            "catch_micro_at_least_2pp_over_strongest_judge": float(catch.get("micro_accuracy") or 0) - strongest_judge
            >= 0.02,
            "net_corrections_at_least_three": corrected - harmed >= 3,
            "override_precision_at_least_65_percent": precision >= 0.65,
            "fixed_decoder_no_dev_search": development_selection is None,
        }
    elif phase_name == "development" and protocol_version == "catch_kernel_v1":
        conditions = {
            "kernel_records_present": bool(catch_rows),
            "task_semantics_recorded": all(bool(row.get("task_family")) for row in catch_rows),
            "jurisdiction_telemetry_recorded": all(
                row.get("verifier_jurisdiction_coverage") is not None for row in catch_rows
            ),
            "proof_telemetry_recorded": all(row.get("proof_completeness") is not None for row in catch_rows),
            "fixed_kernel_no_development_threshold_search": development_selection is None,
        }
    elif phase_name == "development" and protocol_version == "catch_cert_v2":
        answer_link_coverage = _ratio(
            sum(float(row.get("answer_link_coverage") or 0) for row in catch_rows),
            len(catch_rows),
        )
        obligation_coverage = _ratio(
            sum(float(row.get("obligation_coverage") or 0) for row in catch_rows),
            len(catch_rows),
        )
        correct_per_1000 = float(catch.get("correct_per_1000_tokens") or 0)
        sc_correct_per_1000 = float(sc5.get("correct_per_1000_tokens") or 0)
        headroom_utilization = float(catch.get("headroom_utilization") or 0)
        conditions = {
            **common,
            "answer_link_coverage_is_100_percent": answer_link_coverage >= 1.0,
            "certificate_packet_coverage_at_least_80_percent": code_coverage >= 0.80,
            "mandatory_obligation_coverage_at_least_80_percent": obligation_coverage >= 0.80,
            "headroom_utilization_at_least_15_percent": headroom_utilization >= 0.15,
            "corrected_exceeds_harmed": corrected > harmed,
            "correct_per_1000_tokens_at_least_80_percent_of_sc5": (
                sc_correct_per_1000 == 0 or correct_per_1000 >= 0.80 * sc_correct_per_1000
            ),
            "fixed_certificate_decoder_no_dev_search": development_selection is None,
        }
    elif phase_name == "development" and protocol_version == "catch_cert_v1":
        conditions = {
            **common,
            "certificate_packet_coverage_is_nonzero": code_coverage > 0.0,
            "candidate_oracle_micro_at_least_5pp_over_sc5": float(catch.get("candidate_oracle_micro") or 0)
            - float(sc5.get("micro_accuracy") or 0)
            >= 0.05,
            "fixed_certificate_decoder_no_dev_search": development_selection is None,
            "corrected_not_below_harmed": corrected >= harmed,
        }
    elif phase_name == "development":
        conditions = {
            **common,
            "code_packet_on_40_percent_of_disagreements": code_coverage >= 0.40,
            "candidate_oracle_micro_at_least_5pp_over_sc5": float(catch.get("candidate_oracle_micro") or 0)
            - float(sc5.get("micro_accuracy") or 0)
            >= 0.05,
            "catch_micro_at_least_3pp_over_adaptive": float(catch.get("micro_accuracy") or 0)
            - float(adaptive.get("micro_accuracy") or 0)
            >= 0.03,
            "catch_micro_at_least_2pp_over_direct_judge": float(catch.get("micro_accuracy") or 0)
            - float(judge.get("micro_accuracy") or 0)
            >= 0.02,
            "net_corrections_at_least_three": corrected - harmed >= 3,
            "override_precision_at_least_65_percent": precision >= 0.65,
            "grid_selection_constraints_passed": bool(
                (development_selection or {}).get("positive_constraints_satisfied")
            ),
        }
    elif phase_name == "heldout" and protocol_version == "catch_v3":
        conditions = {
            **common,
            "code_packet_on_40_percent_of_disagreements": code_coverage >= 0.40,
            "catch_task_harmonic_at_least_2pp_over_adaptive": float(catch.get("task_harmonic_accuracy") or 0)
            - float(adaptive.get("task_harmonic_accuracy") or 0)
            >= 0.02,
            "catch_micro_not_below_pair_judge": float(catch.get("micro_accuracy") or 0)
            >= float(pair_judge.get("micro_accuracy") or 0),
            "catch_micro_not_below_direct_judge": float(catch.get("micro_accuracy") or 0)
            >= float(judge.get("micro_accuracy") or 0),
            "corrected_exceeds_harmed": corrected > harmed,
            "override_precision_exact_one_sided_95_lower_above_half": _clopper_pearson_lower(
                len(correct_overrides), len(overrides), alpha=0.05
            )
            > 0.5,
        }
    elif phase_name == "heldout":
        conditions = {
            **common,
            "catch_task_harmonic_at_least_2pp_over_adaptive": float(catch.get("task_harmonic_accuracy") or 0)
            - float(adaptive.get("task_harmonic_accuracy") or 0)
            >= 0.02,
            "catch_micro_not_below_direct_judge": float(catch.get("micro_accuracy") or 0)
            >= float(judge.get("micro_accuracy") or 0),
            "corrected_exceeds_harmed": corrected > harmed,
            "override_precision_exact_one_sided_95_lower_above_half": _clopper_pearson_lower(
                len(correct_overrides), len(overrides), alpha=0.05
            )
            > 0.5,
        }
    elif phase_name == "confirmation":
        conditions = common
    else:
        raise ValueError(f"Unsupported CATCH gate phase {phase_name!r}.")
    all_conditions_met = all(conditions.values())
    return {
        "gate_name": f"catch_{phase_name}_{protocol_version}",
        "passed": all_conditions_met,
        "enforcement": (
            "descriptive_only"
            if protocol_version == "catch_kernel_v1"
            else "advisory_only"
            if protocol_version == "catch_cert_v2"
            else "legacy_reporting_only"
        ),
        "blocks_execution": False,
        "all_recommended_conditions_met": all_conditions_met,
        "unmet_conditions": [name for name, met in conditions.items() if not met],
        "conditions": conditions,
        "evidence": {
            "summary": summaries,
            "triggered_count": len(triggered),
            **(
                {
                    "decoder": "fixed_two_of_three_dual_panel_unique_challenger",
                    "coordinates_per_pair": 3,
                    "development_threshold_search": False,
                }
                if protocol_version == "catch_v3"
                else {
                    "decoder": "question_conditioned_certificate_dual_panel_refutation",
                    "development_threshold_search": False,
                }
                if protocol_version == "catch_cert_v1"
                else {
                    "decoder": "answer_linked_global_obligation_adapter_or_dual_panel",
                    "all_stage_candidates_targeted": True,
                    "development_threshold_search": False,
                }
                if protocol_version == "catch_cert_v2"
                else {
                    "decoder": "typed_proof_kernel_no_cross_jurisdiction_fallback",
                    "all_stage_candidates_targeted": True,
                    "development_threshold_search": False,
                }
                if protocol_version == "catch_kernel_v1"
                else {"selected_d_min": selected_d_min, "selected_margin": selected_margin}
            ),
            "code_eligible_count": eligible,
            "code_packet_coverage": code_coverage,
            "structured_turn_count": len(structured_turns),
            "structured_parse_rate": parse_rate,
            "all_actual_tokens_reported": all_actual_tokens_reported,
            "override_count": len(overrides),
            "correct_override_count": len(correct_overrides),
            "override_precision": precision,
            "override_precision_exact_one_sided_95_lower": _clopper_pearson_lower(
                len(correct_overrides), len(overrides), alpha=0.05
            ),
            "override_precision_wilson_one_sided_95_lower": _wilson_lower(
                len(correct_overrides), len(overrides), 1.644854
            ),
            "corrected": corrected,
            "harmed": harmed,
            "development_selection": development_selection,
            "protocol_version": protocol_version,
            "panel_false_pass_dependence": panel_dependence,
            "witness_position_and_agreement": observation_diagnostics,
        },
    }


def _summary(method: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row.get("score") or 0) for row in rows]
    per_task: dict[str, list[float]] = defaultdict(list)
    oracle_per_task: dict[str, list[float]] = defaultdict(list)
    target_oracle_per_task: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        task = str(row.get("task") or "unknown")
        per_task[task].append(float(row.get("score") or 0))
        oracle_per_task[task].append(float(bool(row.get("candidate_oracle_correct"))))
        target_oracle_per_task[task].append(float(bool(row.get("target_oracle_correct"))))
    task_accuracies = {task: sum(values) / len(values) for task, values in per_task.items() if values}
    oracle_task_accuracies = {task: sum(values) / len(values) for task, values in oracle_per_task.items() if values}
    target_oracle_task_accuracies = {
        task: sum(values) / len(values) for task, values in target_oracle_per_task.items() if values
    }
    token_values = [float(row.get("total_tokens_per_question") or 0) for row in rows]
    total_tokens = sum(token_values)
    scores_sum = sum(scores)
    accuracy_wilson = _wilson_interval(int(scores_sum), len(scores))
    transitions = {
        "wrong_to_correct": sum(bool(row.get("corrected_by_debate")) for row in rows),
        "correct_to_wrong": sum(bool(row.get("harmed_by_debate")) for row in rows),
        "wrong_to_wrong": sum(
            float(row.get("initial_vote_score") or 0) < 1.0 and float(row.get("score") or 0) < 1.0 for row in rows
        ),
        "correct_to_correct": sum(
            float(row.get("initial_vote_score") or 0) == 1.0 and float(row.get("score") or 0) == 1.0 for row in rows
        ),
    }
    certificate_rows = [
        row
        for row in rows
        if method in {"catch_cert", "catch_cert_v2", "catch_kernel"} or row.get("certificate_count") is not None
    ]
    verifier_false_pass = sum(
        bool(row.get("override_accepted")) and float(row.get("score") or 0) < 1.0 for row in certificate_rows
    )
    verifier_false_reject = sum(
        bool(row.get("target_oracle_correct"))
        and float(row.get("initial_vote_score") or 0) < 1.0
        and not bool(row.get("override_accepted"))
        for row in certificate_rows
    )
    abstentions = sum(
        bool(row.get("certificate_abstained"))
        or str(row.get("resolver") or "")
        in {
            "abstention",
            "verifier_ambiguous",
            "certificate_invalid",
            "no_certificate",
            "adapter_conflict",
            "jurisdiction_unsupported",
            "proof_incomplete",
            "multiple_complete_proofs",
        }
        for row in certificate_rows
    )
    sequence_metrics = {
        key: _ratio(
            sum(float(row.get(key) or 0) for row in rows if row.get(key) is not None),
            sum(row.get(key) is not None for row in rows),
        )
        for key in (
            "seqbench_exact_match",
            "seqbench_progress_ratio",
            "seqbench_precision",
            "seqbench_recall",
            "seqbench_valid_action_rate",
            "seqbench_execution_prefix_ratio",
            "seqbench_completion_validity",
        )
        if any(row.get(key) is not None for row in rows)
    }
    return {
        "method_name": method,
        "sample_count": len(rows),
        "micro_accuracy": _ratio(sum(scores), len(scores)),
        "accuracy_wilson_95": [accuracy_wilson[0], accuracy_wilson[1]],
        "macro_task_accuracy": _ratio(sum(task_accuracies.values()), len(task_accuracies)),
        "task_harmonic_accuracy": _harmonic_mean(task_accuracies.values()),
        "candidate_oracle_micro": _ratio(sum(bool(row.get("candidate_oracle_correct")) for row in rows), len(rows)),
        "candidate_oracle_task_harmonic": _harmonic_mean(oracle_task_accuracies.values()),
        "target_oracle_micro": _ratio(sum(bool(row.get("target_oracle_correct")) for row in rows), len(rows)),
        "target_oracle_task_harmonic": _harmonic_mean(target_oracle_task_accuracies.values()),
        "mean_total_tokens": _ratio(total_tokens, len(rows)),
        "median_total_tokens": _percentile(token_values, 0.5),
        "p90_total_tokens": _percentile(token_values, 0.9),
        "mean_calls_per_question": _ratio(
            sum(float(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0) for row in rows),
            len(rows),
        ),
        "correct_per_1000_tokens": _ratio(scores_sum * 1000.0, total_tokens),
        "tokens_per_correct": _ratio(total_tokens, scores_sum),
        "transitions": transitions,
        "certificate_coverage": _ratio(
            sum(float(row.get("certificate_coverage") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "certificate_utilization": _ratio(
            sum(bool(row.get("override_accepted")) for row in certificate_rows),
            sum(float(row.get("certificate_coverage") or 0) > 0 for row in certificate_rows),
        ),
        "verifier_false_pass": verifier_false_pass,
        "verifier_false_reject": verifier_false_reject,
        "abstention_rate": _ratio(abstentions, len(certificate_rows)),
        "headroom_utilization": _ratio(
            transitions["wrong_to_correct"] - transitions["correct_to_wrong"],
            sum(
                bool(row.get("target_oracle_correct")) and float(row.get("initial_vote_score") or 0) < 1.0
                for row in rows
            ),
        ),
        "answer_link_coverage": _ratio(
            sum(float(row.get("answer_link_coverage") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "obligation_coverage": _ratio(
            sum(float(row.get("obligation_coverage") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "syntax_validity": _ratio(
            sum(float(row.get("syntax_validity") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "schema_validity": _ratio(
            sum(float(row.get("schema_validity") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "typed_compilation_validity": _mean_present(certificate_rows, "typed_compilation_validity"),
        "semantic_validity": _mean_present(certificate_rows, "semantic_validity"),
        "contract_accuracy": _mean_present(certificate_rows, "contract_accuracy"),
        "verifier_jurisdiction_coverage": _ratio(
            sum(float(row.get("verifier_jurisdiction_coverage") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "proof_completeness": _ratio(
            sum(float(row.get("proof_completeness") or 0) for row in certificate_rows),
            len(certificate_rows),
        ),
        "structural_obligation_completeness": _mean_present(certificate_rows, "structural_obligation_completeness"),
        "provenance_validity": _mean_present(certificate_rows, "provenance_validity"),
        "entailment_validity": _mean_present(certificate_rows, "entailment_validity"),
        "proof_pass_count": sum(int(row.get("proof_pass_count") or 0) for row in certificate_rows),
        "proof_conflict_count": sum(int(row.get("proof_conflict_count") or 0) for row in certificate_rows),
        "proof_unsupported_count": sum(int(row.get("proof_unsupported_count") or 0) for row in certificate_rows),
        "proof_unknown_count": sum(int(row.get("proof_unknown_count") or 0) for row in certificate_rows),
        "adapter_executed_test_count": sum(
            int(row.get("adapter_executed_test_count") or 0) for row in certificate_rows
        ),
        "adapter_conflict_test_count": sum(
            int(row.get("adapter_conflict_test_count") or 0) for row in certificate_rows
        ),
        "adapter_unsupported_test_count": sum(
            int(row.get("adapter_unsupported_test_count") or 0) for row in certificate_rows
        ),
        "adapter_invalid_test_count": sum(int(row.get("adapter_invalid_test_count") or 0) for row in certificate_rows),
        "panel_disagreement_count": sum(int(row.get("panel_disagreement_count") or 0) for row in certificate_rows),
        "verifier_format_repair_count": sum(
            int(row.get("verifier_format_repair_count") or 0) for row in certificate_rows
        ),
        **sequence_metrics,
        "mean_network_attempts": _ratio(
            sum(float(row.get("network_attempts_per_question") or 0) for row in rows), len(rows)
        ),
        "per_task_accuracy": task_accuracies,
    }


def _router_variant(router: dict[str, Any], *, d_min: int, margin: int) -> dict[str, Any] | None:
    for variant in router.get("catch_variants") or []:
        if int(variant.get("d_min") or -1) == d_min and int(variant.get("margin") or -1) == margin:
            return variant
    return None


def _v3_panel_dependence(routers: list[dict[str, Any]]) -> dict[str, Any]:
    false_passes: list[tuple[int, int]] = []
    for router in routers:
        if router.get("protocol_version") != "catch_v3":
            continue
        gold_key = router.get("gold_candidate_key")
        grouped: dict[str, dict[int, bool]] = defaultdict(dict)
        for row in (router.get("decision") or {}).get("panel_diagnostics") or []:
            challenger = str(row.get("challenger_key") or "")
            panel_index = int(row.get("panel_index") or 0)
            if challenger and panel_index in {1, 2}:
                grouped[challenger][panel_index] = bool(row.get("passed"))
        for challenger, panels in grouped.items():
            if challenger == gold_key or set(panels) != {1, 2}:
                continue
            false_passes.append((int(panels[1]), int(panels[2])))
    first_rate = _ratio(sum(first for first, _ in false_passes), len(false_passes))
    second_rate = _ratio(sum(second for _, second in false_passes), len(false_passes))
    joint_rate = _ratio(
        sum(first and second for first, second in false_passes),
        len(false_passes),
    )
    denominator = math.sqrt(first_rate * (1 - first_rate) * second_rate * (1 - second_rate))
    correlation = (joint_rate - first_rate * second_rate) / denominator if denominator > 0 else None
    return {
        "false_challenger_panel_pair_count": len(false_passes),
        "panel_1_false_pass_rate": first_rate,
        "panel_2_false_pass_rate": second_rate,
        "joint_false_pass_rate": joint_rate,
        "bernoulli_correlation": correlation,
        "correlation_defined": correlation is not None,
    }


def _v3_observation_diagnostics(
    routers: list[dict[str, Any]],
    turns: list[dict[str, Any]],
) -> dict[str, Any]:
    comparable = 0
    agreements = 0
    for router in routers:
        panels = router.get("witness_panels") or []
        if len(panels) != 2:
            continue
        first = dict(panels[0].get("observations") or {})
        second = dict(panels[1].get("observations") or {})
        for coordinate_id in set(first) & set(second):
            left = first[coordinate_id]
            right = second[coordinate_id]
            if "ERASURE" in {left, right}:
                continue
            comparable += 1
            agreements += int(left == right)

    verdict_counts = {verdict: 0 for verdict in ("LEFT_ONLY", "RIGHT_ONLY", "BOTH", "NEITHER")}
    for turn in turns:
        if turn.get("role") != "icv_witness":
            continue
        known_ids = set((turn.get("witness_packet") or {}).get("public_to_internal") or {})
        by_id: dict[str, list[str]] = defaultdict(list)
        for row in (turn.get("validated_output") or {}).get("answers") or []:
            if not isinstance(row, dict):
                continue
            public_id = str(row.get("contrast_id") or "")
            verdict = str(row.get("verdict") or "")
            if public_id in known_ids and verdict in verdict_counts:
                by_id[public_id].append(verdict)
        for verdicts in by_id.values():
            if len(verdicts) == 1:
                verdict_counts[verdicts[0]] += 1
    decisive = verdict_counts["LEFT_ONLY"] + verdict_counts["RIGHT_ONLY"]
    return {
        "inverse_mapped_decisive_comparison_count": comparable,
        "inverse_mapped_panel_agreement_count": agreements,
        "inverse_mapped_panel_agreement_rate": _ratio(agreements, comparable),
        "raw_verdict_counts": verdict_counts,
        "left_only_share_among_decisive": _ratio(verdict_counts["LEFT_ONLY"], decisive),
    }


def _certificate_v2_diagnostics(routers: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in routers if row.get("protocol_version") == "catch_cert_v2" and row.get("triggered")]
    comparable = 0
    agreements = 0
    format_repairs = 0
    dropped_reasons: dict[str, int] = defaultdict(int)
    for row in rows:
        panels = row.get("verifier_panels") or []
        format_repairs += sum(int(panel.get("format_repair_count") or 0) for panel in panels)
        if len(panels) == 2:
            first = dict(panels[0].get("results") or {})
            second = dict(panels[1].get("results") or {})
            for test_id in set(first) & set(second):
                comparable += 1
                agreements += int(
                    first[test_id].get("observed_outcome") == second[test_id].get("observed_outcome")
                    and first[test_id].get("support_status") == second[test_id].get("support_status")
                )
        for item in row.get("dropped_certificate_items") or []:
            dropped_reasons[str(item.get("reason") or "unknown")] += 1
    return {
        "sample_count": len(rows),
        "mean_answer_link_coverage": _ratio(
            sum(float(row.get("answer_link_coverage") or 0) for row in rows), len(rows)
        ),
        "mean_obligation_coverage": _ratio(sum(float(row.get("obligation_coverage") or 0) for row in rows), len(rows)),
        "certificate_test_count": sum(len(row.get("certificate_tests") or []) for row in rows),
        "certificate_count": sum(len(row.get("certificates") or []) for row in rows),
        "override_count": sum(bool((row.get("decision") or {}).get("override_accepted")) for row in rows),
        "adapter_executed_test_count": sum(
            sum(item.get("execution_status") == "EXECUTED" for item in dict(row.get("adapter_results") or {}).values())
            for row in rows
        ),
        "panel_comparable_test_count": comparable,
        "panel_exact_agreement_rate": _ratio(agreements, comparable),
        "format_repair_count": format_repairs,
        "dropped_reason_counts": dict(sorted(dropped_reasons.items())),
    }


def _kernel_diagnostics(routers: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for row in routers if row.get("protocol_version") == "catch_kernel_v1" and row.get("triggered")]
    bindings = [
        item for row in rows for item in dict(row.get("verifier_bindings") or {}).values() if isinstance(item, dict)
    ]
    proofs = [item for row in rows for item in row.get("proof_results") or [] if isinstance(item, dict)]
    route_counts: dict[str, int] = defaultdict(int)
    route_outcomes: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    status_counts: dict[str, int] = defaultdict(int)
    adapter_status_counts: dict[str, int] = defaultdict(int)
    unary_status_counts: dict[str, int] = defaultdict(int)
    failure_layers: dict[str, int] = defaultdict(int)
    for item in bindings:
        route_counts[str(item.get("verifier_kind") or "unknown")] += 1
    for item in proofs:
        status_counts[str(item.get("status") or "unknown")] += 1
    for row in rows:
        decision = row.get("kernel_decision") or {}
        failure_layers[str(decision.get("failure_layer") or "none")] += 1
        for item in dict(row.get("adapter_results") or {}).values():
            if isinstance(item, dict):
                adapter_status_counts[str(item.get("execution_status") or "unknown")] += 1
        for item in dict(row.get("unary_adapter_results") or {}).values():
            if isinstance(item, dict):
                unary_status_counts[str(item.get("status") or "unknown")] += 1
        if decision.get("decision") != "OVERRIDE":
            continue
        accepted = set(decision.get("accepted_proofs") or [])
        routes = {
            str(item.get("verifier_kind") or "unknown")
            for item in row.get("proof_results") or []
            if isinstance(item, dict) and item.get("test_id") in accepted
        }
        if any(str(item).startswith("unary:") for item in accepted):
            routes.add("deterministic.unary_exact")
        public_to_key = dict(row.get("candidate_public_to_answer_class_key") or {})
        challenger_key = public_to_key.get(decision.get("challenger_id"))
        anchor_key = row.get("anchor_key")
        gold_keys = set(row.get("gold_candidate_keys") or ())
        if not gold_keys and row.get("gold_candidate_key") is not None:
            gold_keys.add(row.get("gold_candidate_key"))
        initial_correct = anchor_key in gold_keys
        final_correct = challenger_key in gold_keys
        outcome = (
            "wrong_to_correct"
            if not initial_correct and final_correct
            else "correct_to_wrong"
            if initial_correct and not final_correct
            else "correct_to_correct"
            if initial_correct and final_correct
            else "wrong_to_wrong"
        )
        for route in routes:
            route_outcomes[route]["overrides"] += 1
            route_outcomes[route][outcome] += 1
    route_quality = {
        route: {
            **dict(counts),
            "correction_precision": _ratio(counts.get("wrong_to_correct", 0), counts.get("overrides", 0)),
            "harm_rate": _ratio(counts.get("correct_to_wrong", 0), counts.get("overrides", 0)),
        }
        for route, counts in sorted(route_outcomes.items())
    }
    return {
        "sample_count": len(rows),
        "binding_count": len(bindings),
        "jurisdiction_coverage": _ratio(sum(item.get("binding_status") == "BOUND" for item in bindings), len(bindings)),
        "proof_count": len(proofs),
        "proof_completeness": _ratio(
            sum(
                item.get("status") == "PASS"
                and bool(item.get("provenance_valid"))
                and bool(item.get("entailment_valid"))
                and bool(item.get("obligation_valid"))
                and bool(item.get("sufficiency_valid"))
                for item in proofs
            ),
            len(proofs),
        ),
        "verifier_route_counts": dict(sorted(route_counts.items())),
        "verifier_route_quality": route_quality,
        "proof_status_counts": dict(sorted(status_counts.items())),
        "adapter_status_counts": dict(sorted(adapter_status_counts.items())),
        "unary_candidate_status_counts": dict(sorted(unary_status_counts.items())),
        "panel_disagreement_count": sum(item.get("detail") == "panel_disagreement" for item in proofs),
        "failure_layer_counts": dict(sorted(failure_layers.items())),
        "cross_jurisdiction_fallback_count": 0,
    }


def _parse_variant(method_name: str) -> tuple[int, int]:
    pieces = method_name.removeprefix("catch_d").split("_m", 1)
    if len(pieces) != 2:
        raise ValueError(f"Invalid CATCH grid method {method_name!r}.")
    return int(pieces[0]), int(pieces[1])


def _stable_tie_rank(value: str) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return -int(digest, 16)


def _harmonic_mean(values) -> float:
    materialized = [float(value) for value in values]
    if not materialized or any(value <= 0 for value in materialized):
        return 0.0
    return len(materialized) / sum(1 / value for value in materialized)


def _mean_present(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [float(row[field]) for row in rows if row.get(field) is not None]
    return _ratio(sum(values), len(values)) if values else None


def _percentile(values: list[float], quantile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil(quantile * len(ordered)) - 1))
    return ordered[index]


def _ratio(numerator: float, denominator: int) -> float:
    return float(numerator) / denominator if denominator else 0.0


def _clopper_pearson_lower(successes: int, total: int, *, alpha: float) -> float:
    if total <= 0 or successes <= 0:
        return 0.0
    return float(beta.ppf(alpha, successes, total - successes + 1))


def _wilson_lower(successes: int, total: int, z: float) -> float:
    if total <= 0:
        return 0.0
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = proportion + z**2 / (2 * total)
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total)
    return (centre - margin) / denominator


def _wilson_interval(successes: int, total: int, z: float = 1.959964) -> tuple[float, float]:
    if total <= 0:
        return (0.0, 0.0)
    proportion = successes / total
    denominator = 1 + z**2 / total
    centre = (proportion + z**2 / (2 * total)) / denominator
    margin = z * math.sqrt((proportion * (1 - proportion) + z**2 / (4 * total)) / total) / denominator
    return max(0.0, centre - margin), min(1.0, centre + margin)
