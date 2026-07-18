"""Auditable artifacts and researcher-facing report for the four-domain study.

生成四域审计的机器可读指标、人工效度复算与研究者向失败报告。
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.contrastive_active_testing.statistics import (
    _v3_observation_diagnostics,
    _v3_panel_dependence,
    build_metrics,
)


def materialize_boundary_artifacts(
    root: Path,
    *,
    screening_rows: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    routers: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    source_manifest: dict[str, Any],
    checkpoints: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    source_manifest = {
        **source_manifest,
        "turn_contract_hashes": _turn_contract_hashes(turns),
    }
    screening = _screening_metrics(screening_rows)
    conditional = _conditional_metrics(predictions)
    selector = _selector_funnel(turns, routers)
    witness = _witness_analysis(turns, routers)
    costs = _cost_analysis(turns)
    sample_outcomes = _sample_outcomes(predictions, routers)
    mechanism = _mechanism_assessment(screening, conditional, selector, witness)
    metrics = {
        "study_type": "post_failure_cross_domain_boundary_audit",
        "confirmatory": False,
        "screening": screening,
        "conditional_disagreement_audit": conditional,
        "descriptive_unweighted_macro": _descriptive_macro(conditional),
        "costs": costs,
        "mechanism_assessment": mechanism,
    }
    _write_json(root / "metrics.json", metrics)
    _write_json(root / "views" / "metrics.json", metrics)
    _write_json(root / "selector_funnel.json", selector)
    _write_json(root / "witness_analysis.json", witness)
    _write_json(root / "reproducibility_manifest.json", source_manifest)
    _write_jsonl(root / "sample_outcomes.jsonl", sample_outcomes)
    _write_json(root / "diagnostics" / "dataset_checkpoints.json", checkpoints)
    _write_human_audit_sample(root, routers)
    (root / "failure_cases.md").write_text(_failure_cases(routers), encoding="utf-8")
    (root / "index.md").write_text(_index_markdown(root), encoding="utf-8")
    report = _render_report(metrics, selector, witness, source_manifest, checkpoints)
    (root / "report.md").write_text(report, encoding="utf-8")
    return {
        "metrics": metrics,
        "selector_funnel": selector,
        "witness_analysis": witness,
        "sample_outcomes": sample_outcomes,
        "mechanism_assessment": mechanism,
    }


def _turn_contract_hashes(turns: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset, rows in _group(turns, "dataset").items():
        cache_rows = sorted(
            (
                str(row.get("role") or ""),
                str(row.get("cache_key") or ""),
                str(row.get("cache_namespace") or ""),
                str(row.get("request_source") or ""),
            )
            for row in rows
        )
        prompt_hashes = sorted(str(row.get("prompt_hash") or "") for row in rows)
        result[dataset] = {
            "logical_turn_count": len(rows),
            "unique_cache_key_count": len({row[1] for row in cache_rows if row[1]}),
            "cache_contract_sha256": _json_sha256(cache_rows),
            "prompt_hash_multiset_sha256": _json_sha256(prompt_hashes),
        }
    return result


def _json_sha256(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _screening_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    grouped = _group(rows, "dataset")
    for dataset, items in grouped.items():
        candidate_counts = Counter(int(row.get("candidate_count") or 0) for row in items)
        result[dataset] = {
            "sample_count": len(items),
            "sc5_micro_accuracy": _mean(float(row.get("sc5_score") or 0) for row in items),
            "adaptive_sc8_micro_accuracy": None,
            "adaptive_note": "Not run on the full screening pool; it is reported on the selected disagreement subset.",
            "candidate_oracle_micro": _mean(float(bool(row.get("candidate_oracle_correct"))) for row in items),
            "target_oracle_micro": _mean(float(bool(row.get("target_oracle_correct"))) for row in items),
            "disagreement_count": sum(bool(row.get("triggered")) for row in items),
            "disagreement_rate": _mean(float(bool(row.get("triggered"))) for row in items),
            "invalid_stage_answer_count": sum(int(row.get("invalid_stage_answer_count") or 0) for row in items),
            "candidate_count_distribution": {str(key): value for key, value in sorted(candidate_counts.items())},
        }
    return result


def _conditional_metrics(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset, dataset_rows in _group(predictions, "dataset").items():
        summaries = build_metrics(dataset_rows).get("summary") or []
        methods: dict[str, Any] = {}
        for summary in summaries:
            method = str(summary.get("method_name"))
            method_rows = [row for row in dataset_rows if row.get("method_name") == method]
            overrides = [row for row in method_rows if row.get("override_accepted")]
            correct_overrides = sum(float(row.get("score") or 0) == 1.0 for row in overrides)
            methods[method] = {
                **summary,
                "override_count": len(overrides),
                "corrected": sum(bool(row.get("corrected_by_debate")) for row in method_rows),
                "harmed": sum(bool(row.get("harmed_by_debate")) for row in method_rows),
                "override_precision": correct_overrides / len(overrides) if overrides else None,
            }
        result[dataset] = {
            "selected_disagreement_count": len({row.get("sample_id") for row in dataset_rows}),
            "conditional_on_stage_a_disagreement": True,
            "methods": methods,
        }
    return result


def _selector_funnel(turns: list[dict[str, Any]], routers: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    selector_rows = [row for row in turns if row.get("role") == "icv_selector"]
    for dataset, rows in _group(selector_rows, "dataset").items():
        drops = Counter()
        raw_count = accepted_count = leakage = raw_three_coordinate_samples = 0
        for row in rows:
            raw = (row.get("validated_output") or {}).get("contrasts")
            raw = raw if isinstance(raw, list) else []
            raw_count += len(raw)
            per_pair = Counter(str(item.get("pair_id") or "") for item in raw if isinstance(item, dict))
            raw_three_coordinate_samples += int(any(value >= 3 for value in per_pair.values()))
            accepted_count += len(row.get("validated_contrasts") or [])
            leakage += int(row.get("leakage_count") or 0)
            for drop in row.get("dropped_contrasts") or []:
                drops[str(drop.get("reason") or "unknown")] += 1
        dataset_routers = [row for row in routers if row.get("dataset") == dataset]
        eligible = sum(bool(row.get("eligible_challengers")) for row in dataset_routers)
        dropped_count = sum(drops.values())
        result[dataset] = {
            "selector_sample_count": len(rows),
            "top_level_parse_rate": _mean(float(row.get("protocol_parse_status") == "ok") for row in rows),
            "raw_coordinate_count": raw_count,
            "accepted_coordinate_count": accepted_count,
            "dropped_coordinate_count": dropped_count,
            "coordinate_reference_validity_rate": accepted_count / (accepted_count + dropped_count) if accepted_count + dropped_count else 0.0,
            "answer_leakage_count": leakage,
            "eligible_sample_count": eligible,
            "eligible_sample_rate": eligible / len(dataset_routers) if dataset_routers else 0.0,
            "raw_three_coordinate_sample_count": raw_three_coordinate_samples,
            "raw_three_coordinate_coverage_upper_bound": raw_three_coordinate_samples / len(rows) if rows else 0.0,
            "drop_reasons": dict(sorted(drops.items())),
        }
    return result


def _witness_analysis(turns: list[dict[str, Any]], routers: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    witness_rows = [row for row in turns if row.get("role") == "icv_witness"]
    datasets = sorted({str(row.get("dataset")) for row in routers})
    for dataset in datasets:
        rows = [row for row in witness_rows if row.get("dataset") == dataset]
        expected = sum(int((row.get("witness_parse_diagnostics") or {}).get("expected_coordinate_count") or 0) for row in rows)
        valid = sum(int((row.get("witness_parse_diagnostics") or {}).get("valid_coordinate_count") or 0) for row in rows)
        decisive = sum(int((row.get("witness_parse_diagnostics") or {}).get("decisive_coordinate_count") or 0) for row in rows)
        agreements = comparisons = 0
        for router in [row for row in routers if row.get("dataset") == dataset]:
            panels = router.get("witness_panels") or []
            if len(panels) != 2:
                continue
            first = dict(panels[0].get("observations") or {})
            second = dict(panels[1].get("observations") or {})
            for coordinate in set(first) & set(second):
                if first[coordinate] in {"ERASURE", "BOTH", "NEITHER"} or second[coordinate] in {"ERASURE", "BOTH", "NEITHER"}:
                    continue
                comparisons += 1
                agreements += int(first[coordinate] == second[coordinate])
        result[dataset] = {
            "witness_call_count": len(rows),
            "top_level_parse_rate": _mean(float(row.get("protocol_parse_status") == "ok") for row in rows),
            "expected_coordinate_rows": expected,
            "valid_coordinate_rows": valid,
            "valid_coordinate_rate": valid / expected if expected else None,
            "decisive_coordinate_rows": decisive,
            "decisive_verdict_rate": decisive / valid if valid else None,
            "inverse_mapped_panel_comparisons": comparisons,
            "inverse_mapped_panel_agreement_rate": agreements / comparisons if comparisons else None,
            "false_pass_dependence": _v3_panel_dependence(
                [router for router in routers if router.get("dataset") == dataset]
            ),
            "position_and_agreement": _v3_observation_diagnostics(
                [router for router in routers if router.get("dataset") == dataset], rows
            ),
        }
    return result


def _cost_analysis(turns: list[dict[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for dataset, rows in _group(turns, "dataset").items():
        cache_sources = Counter(str(row.get("request_source") or "unknown") for row in rows)
        attempts = [attempt for row in rows for attempt in row.get("attempt_timeline") or []]
        cache_lookups = [
            float((row.get("cache_lookup_timeline") or {}).get("duration_ms") or 0)
            for row in rows
            if row.get("cache_lookup_timeline")
        ]
        result[dataset] = {
            "logical_calls": len(rows),
            "cached_logical_calls": sum(bool(row.get("cache_hit")) for row in rows),
            "physical_network_attempts": sum(int(row.get("network_attempt_count") or 0) for row in rows),
            "retry_attempts": sum(max(0, int(row.get("network_attempt_count") or 0) - int(int(row.get("network_attempt_count") or 0) > 0)) for row in rows),
            "actual_prompt_tokens": sum(float(row.get("actual_prompt_tokens") or 0) for row in rows),
            "actual_completion_tokens": sum(float(row.get("actual_completion_tokens") or 0) for row in rows),
            "actual_total_tokens": sum(float(row.get("actual_total_tokens") or 0) for row in rows),
            "reported_reasoning_tokens": sum(float(row.get("reasoning_tokens") or 0) for row in rows),
            "provider_network_latency_ms_total": sum(float(item.get("latency_ms") or 0) for item in attempts),
            "mean_provider_latency_ms": _mean(float(item.get("latency_ms") or 0) for item in attempts),
            "throttle_wait_ms_total": sum(float(item.get("throttle_wait_ms") or 0) for item in attempts),
            "retry_backoff_seconds_total": sum(float(item.get("retry_delay_seconds") or 0) for item in attempts),
            "cache_lookup_ms_total": sum(cache_lookups),
            "mean_cache_lookup_ms": _mean(cache_lookups),
            "cache_source_counts": dict(sorted(cache_sources.items())),
        }
    return result


def _sample_outcomes(predictions: list[dict[str, Any]], routers: list[dict[str, Any]]) -> list[dict[str, Any]]:
    router_by_sample = {(row.get("dataset"), row.get("sample_id")): row for row in routers}
    grouped: dict[tuple[Any, Any], list[dict[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[(row.get("dataset"), row.get("sample_id"))].append(row)
    output = []
    for key in sorted(grouped, key=lambda value: (str(value[0]), str(value[1]))):
        rows = grouped[key]
        router = router_by_sample.get(key, {})
        output.append(
            {
                "dataset": key[0],
                "sample_id": key[1],
                "task": rows[0].get("task"),
                "candidate_oracle_correct": rows[0].get("candidate_oracle_correct"),
                "target_oracle_correct": rows[0].get("target_oracle_correct"),
                "eligible_challengers": router.get("eligible_challengers") or [],
                "catch_reason_code": (router.get("decision") or {}).get("resolver"),
                "methods": {
                    str(row.get("method_name")): {
                        "prediction": row.get("prediction"),
                        "score": row.get("score"),
                        "override": row.get("override_accepted"),
                        "corrected": row.get("corrected_by_debate"),
                        "harmed": row.get("harmed_by_debate"),
                        "tokens": row.get("total_tokens_per_question"),
                    }
                    for row in rows
                },
            }
        )
    return output


def _mechanism_assessment(screening, conditional, selector, witness) -> dict[str, Any]:
    by_dataset: dict[str, str] = {}
    for dataset in sorted(screening):
        selector_row = selector.get(dataset, {})
        witness_row = witness.get(dataset, {})
        methods = conditional.get(dataset, {}).get("methods", {})
        catch = methods.get("catch", {})
        pair = methods.get("pair_judge_3", {})
        if float(selector_row.get("eligible_sample_rate") or 0) < 0.4:
            conclusion = "indexed_trace_contrast_formation_failed"
        elif witness_row.get("decisive_verdict_rate") is not None and float(witness_row["decisive_verdict_rate"]) < 0.8:
            conclusion = "homogeneous_witness_measurement_failed"
        elif int(catch.get("corrected") or 0) <= int(catch.get("harmed") or 0):
            conclusion = "measurement_not_aligned_with_candidate_correctness"
        elif float(pair.get("micro_accuracy") or 0) >= float(catch.get("micro_accuracy") or 0):
            conclusion = "pair_judge_not_outperformed"
        else:
            conclusion = "exploratory_positive_signal_only"
        by_dataset[dataset] = conclusion
    return {
        "by_dataset": by_dataset,
        "positive_claim_authorized": False,
        "heldout_authorized": False,
        "interpretation": "This audit maps a failed mechanism's domain boundary; it cannot rescue the failed preregistered v3 gate.",
    }


def _descriptive_macro(conditional: dict[str, Any]) -> dict[str, float]:
    values: dict[str, list[float]] = defaultdict(list)
    for dataset in conditional.values():
        for method, row in dataset.get("methods", {}).items():
            values[method].append(float(row.get("micro_accuracy") or 0))
    return {method: sum(rows) / len(rows) for method, rows in sorted(values.items()) if rows}


def _write_human_audit_sample(root: Path, routers: list[dict[str, Any]]) -> None:
    items: list[dict[str, Any]] = []
    for dataset in sorted({str(row.get("dataset")) for row in routers}):
        coordinates = []
        for router in routers:
            if router.get("dataset") != dataset:
                continue
            for coordinate in router.get("validated_contrasts") or []:
                coordinates.append((str(coordinate.get("sha256") or ""), router, coordinate))
        for _, router, coordinate in sorted(coordinates, key=lambda item: item[0])[:10]:
            items.append(
                {
                    "dataset": dataset,
                    "sample_id": router.get("sample_id"),
                    "coordinate_sha256": coordinate.get("sha256"),
                    "source_question_without_answer_contract": router.get("audit_source_question"),
                    "left_text": coordinate.get("left_text"),
                    "right_text": coordinate.get("right_text"),
                    "blind_to_gold_votes_and_candidate_answers": True,
                    "annotator_1": None,
                    "annotator_2": None,
                    "adjudication": None,
                }
            )
    _write_json(
        root / "diagnostics" / "human_audit_sample.json",
        {
            "audit_version": "catch_v3_cross_domain_blind_coordinate_audit_v1",
            "seed": 42,
            "maximum_per_dataset": 10,
            "quota_transfer_between_datasets": False,
            "criteria": [
                "decidable_from_source",
                "mutually_exclusive",
                "atomic",
                "context_sufficient",
                "answer_leakage",
            ],
            "items": items,
        },
    )


_BOUNDARY_HUMAN_CRITERIA = (
    "decidable_from_source",
    "mutually_exclusive",
    "atomic",
    "context_sufficient",
    "answer_leakage",
)


def evaluate_boundary_human_audit(
    payload: dict[str, Any],
    *,
    expected_sample: dict[str, Any],
) -> dict[str, Any]:
    """Recompute the optional two-annotator boundary audit without using gold."""

    expected_items = expected_sample.get("items")
    items = payload.get("items")
    expected_items = expected_items if isinstance(expected_items, list) else []
    items = items if isinstance(items, list) else []
    expected_keys = {
        (str(item.get("dataset") or ""), str(item.get("coordinate_sha256") or ""))
        for item in expected_items
        if isinstance(item, dict)
    }
    observed_keys = [
        (str(item.get("dataset") or ""), str(item.get("coordinate_sha256") or ""))
        for item in items
        if isinstance(item, dict)
    ]
    complete = True
    disagreement_count = 0
    adjudicated_disagreement_count = 0
    final_values: dict[str, dict[str, list[bool]]] = defaultdict(
        lambda: {criterion: [] for criterion in _BOUNDARY_HUMAN_CRITERIA}
    )
    first_values: dict[str, list[bool]] = defaultdict(list)
    second_values: dict[str, list[bool]] = defaultdict(list)

    for item in items:
        if not isinstance(item, dict):
            complete = False
            continue
        dataset = str(item.get("dataset") or "unknown")
        first = item.get("annotator_1")
        second = item.get("annotator_2")
        adjudication = item.get("adjudication")
        if not isinstance(first, dict) or not isinstance(second, dict):
            complete = False
            continue
        for criterion in _BOUNDARY_HUMAN_CRITERIA:
            first_value = first.get(criterion)
            second_value = second.get(criterion)
            if not isinstance(first_value, bool) or not isinstance(second_value, bool):
                complete = False
                continue
            if criterion != "answer_leakage":
                first_values[dataset].append(first_value)
                second_values[dataset].append(second_value)
            if first_value == second_value:
                final_values[dataset][criterion].append(first_value)
                continue
            disagreement_count += 1
            if not isinstance(adjudication, dict) or not isinstance(adjudication.get(criterion), bool):
                complete = False
                continue
            adjudicated_disagreement_count += 1
            final_values[dataset][criterion].append(bool(adjudication[criterion]))

    by_dataset: dict[str, Any] = {}
    pooled_first: list[bool] = []
    pooled_second: list[bool] = []
    pooled_final = {criterion: [] for criterion in _BOUNDARY_HUMAN_CRITERIA}
    for dataset in sorted({key[0] for key in expected_keys}):
        dataset_items = [key for key in observed_keys if key[0] == dataset]
        values = final_values[dataset]
        rates = {criterion: _mean(float(value) for value in rows) for criterion, rows in values.items()}
        by_dataset[dataset] = {
            "item_count": len(dataset_items),
            "rates": rates,
            "cohen_kappa_pooled_non_leakage": _cohen_kappa(
                first_values[dataset], second_values[dataset]
            ),
        }
        pooled_first.extend(first_values[dataset])
        pooled_second.extend(second_values[dataset])
        for criterion in _BOUNDARY_HUMAN_CRITERIA:
            pooled_final[criterion].extend(values[criterion])

    overall_rates = {
        criterion: _mean(float(value) for value in rows)
        for criterion, rows in pooled_final.items()
    }
    pooled_kappa = _cohen_kappa(pooled_first, pooled_second)
    sample_contract_matches = (
        len(observed_keys) == len(set(observed_keys))
        and set(observed_keys) == expected_keys
        and all(dataset and coordinate_hash for dataset, coordinate_hash in observed_keys)
    )
    audit_complete = (
        complete
        and sample_contract_matches
        and disagreement_count == adjudicated_disagreement_count
        and all(
            len(pooled_final[criterion]) == len(expected_items)
            for criterion in _BOUNDARY_HUMAN_CRITERIA
        )
    )
    thresholds = {
        "decidable_at_least_90_percent": overall_rates["decidable_from_source"] >= 0.90,
        "exclusive_at_least_90_percent": overall_rates["mutually_exclusive"] >= 0.90,
        "atomic_at_least_90_percent": overall_rates["atomic"] >= 0.90,
        "context_sufficient_at_least_90_percent": overall_rates["context_sufficient"] >= 0.90,
        "answer_leakage_is_zero": overall_rates["answer_leakage"] == 0.0,
        "pooled_non_leakage_kappa_at_least_0_6": pooled_kappa is not None and pooled_kappa >= 0.60,
    }
    return {
        "audit_version": "catch_v3_cross_domain_blind_coordinate_audit_v1",
        "confirmatory": False,
        "sample_contract_matches": sample_contract_matches,
        "audit_complete": audit_complete,
        "mechanism_validity_thresholds_met": audit_complete and all(thresholds.values()),
        "thresholds": thresholds,
        "item_count": len(items),
        "maximum_per_dataset": 10,
        "quota_transfer_between_datasets": False,
        "overall_rates": overall_rates,
        "cohen_kappa_pooled_non_leakage": pooled_kappa,
        "by_dataset": by_dataset,
        "disagreement_count": disagreement_count,
        "adjudicated_disagreement_count": adjudicated_disagreement_count,
    }


def install_boundary_human_audit(run_root: Path, completed_payload: dict[str, Any]) -> dict[str, Any]:
    """Validate, archive, and expose a completed audit without altering predictions."""

    sample_path = run_root / "diagnostics" / "human_audit_sample.json"
    if not sample_path.exists():
        raise FileNotFoundError(f"Boundary human-audit sample is missing: {sample_path}")
    sample = json.loads(sample_path.read_text(encoding="utf-8"))
    evaluation = evaluate_boundary_human_audit(completed_payload, expected_sample=sample)
    _write_json(run_root / "diagnostics" / "human_audit_completed.json", completed_payload)
    _write_json(run_root / "diagnostics" / "human_audit_evaluation.json", evaluation)
    _rerender_boundary_report(run_root, human_evaluation=evaluation)
    return evaluation


def _cohen_kappa(first: list[bool], second: list[bool]) -> float | None:
    if not first or len(first) != len(second):
        return None
    observed = _mean(float(left == right) for left, right in zip(first, second, strict=True))
    first_positive = _mean(float(value) for value in first)
    second_positive = _mean(float(value) for value in second)
    expected = first_positive * second_positive + (1 - first_positive) * (1 - second_positive)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def _failure_cases(routers: list[dict[str, Any]]) -> str:
    lines = [
        "# CATCH-v3 boundary-audit failure cases",
        "",
        "The immutable raw chain is in [agent turns](turns/agent_turns.jsonl), "
        "[router decisions](turns/router_decisions.jsonl), and "
        "[sample outcomes](sample_outcomes.jsonl). Search those files by the exact sample ID below.",
        "",
    ]
    for dataset in sorted({str(row.get("dataset")) for row in routers}):
        lines.extend([f"## {dataset}", ""])
        examples = [row for row in routers if row.get("dataset") == dataset and (row.get("dropped_contrasts") or not row.get("eligible_challengers"))][:5]
        if not examples:
            lines.append("No selector failure example was available.")
            lines.append("")
            continue
        for row in examples:
            drops = Counter(str(item.get("reason") or "unknown") for item in row.get("dropped_contrasts") or [])
            lines.append(
                f"- `{row.get('sample_id')}`: eligible={bool(row.get('eligible_challengers'))}; "
                f"decision=`{(row.get('decision') or {}).get('resolver')}`; drops={dict(drops)}"
                f"; accepted_coordinates={len(row.get('validated_contrasts') or [])}."
            )
        lines.append("")
    return "\n".join(lines) + "\n"


def _index_markdown(root: Path) -> str:
    del root
    return """# CATCH experiment lineage

- [CATCH-v1 report](../../../catch_gate/development/20260717T140404Z-xiaomimimo-mimo-v2.5/report.md): `catch_v1_failed_no_observation_signal`
- [CATCH-v2 report](../../../catch_gate/development/20260718T040024Z-xiaomimimo-mimo-v2.5/report.md): `catch_v2_failed_undefined_generated_measurement`
- [CATCH-v3 BBEH preflight report](../../../catch_gate/development/20260718T090517Z-xiaomimimo-mimo-v2.5/report.md): `failed_structural_preflight`
- [Current four-dataset report](report.md): exploratory mechanism-boundary audit; not a confirmatory gate.

Machine-readable views: [metrics](metrics.json), [selector funnel](selector_funnel.json), [witness analysis](witness_analysis.json), [sample outcomes](sample_outcomes.jsonl), and [failure cases](failure_cases.md).
"""


def _render_report_legacy(metrics, selector, witness, manifest, checkpoints) -> str:
    lines = [
        "# CATCH-v3 four-dataset mechanism boundary audit",
        "",
        "## 1. Executive conclusion",
        "",
        "This is a post-failure exploratory audit. It does not reopen the failed BBEH gate, authorize heldout, or support a positive CATCH claim.",
        "",
        "The prior BBEH v3 preflight failed structurally: coordinate reference validity was 75% (required 100%), eligible-packet coverage was 40% (required 60%), one automated answer leakage was detected, and the raw-output coverage upper bound was only 55%.",
        "",
        "## 2. Run and data integrity",
        "",
        "| Dataset | Revision | SHA-256 | Screening | Selected disagreements | Status |",
        "|---|---|---|---:|---:|---|",
    ]
    sources = {row["dataset"]: row for row in manifest.get("dataset_sources", [])}
    screening_manifests = manifest.get("screening_manifests", {})
    selection_manifests = manifest.get("disagreement_manifests", {})
    for dataset in sorted(sources):
        source = sources[dataset]
        lines.append(
            f"| {dataset} | `{source.get('revision')}` | `{str(source.get('sha256'))[:12]}…` | "
            f"{len(screening_manifests.get(dataset, {}).get('sample_ids', []))} | "
            f"{len(selection_manifests.get(dataset, {}).get('sample_ids', []))} | "
            f"{checkpoints.get(dataset, {}).get('status', 'unknown')} |"
        )
    lines.extend(["", "## 3. Screening-pool candidate headroom", "", "Adaptive-SC8 was not run over all 100 screening items; it is evaluated only on the preregistered disagreement subset.", "", "| Dataset | SC5 | Candidate oracle | Target oracle | Disagreements | Invalid Stage-A |", "|---|---:|---:|---:|---:|---:|"])
    for dataset, row in sorted(metrics.get("screening", {}).items()):
        lines.append(
            f"| {dataset} | {row['sc5_micro_accuracy']:.3f} | {row['candidate_oracle_micro']:.3f} | "
            f"{row['target_oracle_micro']:.3f} | {row['disagreement_count']}/{row['sample_count']} | "
            f"{row['invalid_stage_answer_count']} |"
        )
    lines.extend(["", "## 4. Conditional matched-method results", "", "These accuracies are conditional on Stage-A disagreement and are not benchmark-level headline accuracies.", "", "| Dataset | Method | Accuracy | Corrected | Harmed | Overrides | Precision | Mean tokens |", "|---|---|---:|---:|---:|---:|---:|---:|"])
    for dataset, block in sorted(metrics.get("conditional_disagreement_audit", {}).items()):
        for method, row in block.get("methods", {}).items():
            precision = "—" if row.get("override_precision") is None else f"{row['override_precision']:.3f}"
            lines.append(
                f"| {dataset} | {method} | {row['micro_accuracy']:.3f} | {row['corrected']} | {row['harmed']} | "
                f"{row['override_count']} | {precision} | {row['mean_total_tokens']:.1f} |"
            )
    lines.extend(["", "## 5. Selector funnel", "", "| Dataset | Parsed | Raw | Accepted | Dropped | Reference validity | Eligible samples | Raw coverage upper bound | Leakage |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, row in sorted(selector.items()):
        lines.append(
            f"| {dataset} | {row['top_level_parse_rate']:.3f} | {row['raw_coordinate_count']} | "
            f"{row['accepted_coordinate_count']} | {row['dropped_coordinate_count']} | "
            f"{row['coordinate_reference_validity_rate']:.3f} | {row['eligible_sample_count']}/{row['selector_sample_count']} | "
            f"{row['raw_three_coordinate_coverage_upper_bound']:.3f} | {row['answer_leakage_count']} |"
        )
        lines.append(f"\nDrop reasons for **{dataset}**: `{json.dumps(row['drop_reasons'], ensure_ascii=False, sort_keys=True)}`\n")
    lines.extend(["", "## 6. Witness measurements", "", "| Dataset | Calls | Parsed | Valid coordinates | Decisive | Panel agreement | False-pass correlation | Left-only share |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for dataset, row in sorted(witness.items()):
        valid = "—" if row.get("valid_coordinate_rate") is None else f"{row['valid_coordinate_rate']:.3f}"
        decisive = "—" if row.get("decisive_verdict_rate") is None else f"{row['decisive_verdict_rate']:.3f}"
        agreement = "—" if row.get("inverse_mapped_panel_agreement_rate") is None else f"{row['inverse_mapped_panel_agreement_rate']:.3f}"
        correlation_value = (row.get("false_pass_dependence") or {}).get("bernoulli_correlation")
        correlation = "—" if correlation_value is None else f"{float(correlation_value):.3f}"
        left_value = (row.get("position_and_agreement") or {}).get("left_only_share_among_decisive")
        left_share = "—" if left_value is None else f"{float(left_value):.3f}"
        lines.append(f"| {dataset} | {row['witness_call_count']} | {row['top_level_parse_rate']:.3f} | {valid} | {decisive} | {agreement} | {correlation} | {left_share} |")
    lines.extend(["", "## 7. Mechanism attribution", ""])
    for dataset, conclusion in sorted(metrics.get("mechanism_assessment", {}).get("by_dataset", {}).items()):
        lines.append(f"- **{dataset}**: `{conclusion}`")
    lines.extend(
        [
            "",
            "## 8. Cost and cache accounting",
            "",
            "| Dataset | Logical calls | Cached | Physical attempts | Retries | Actual tokens | Reasoning tokens | Mean provider latency ms |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, row in sorted(metrics.get("costs", {}).items()):
        lines.append(
            f"| {dataset} | {row['logical_calls']} | {row['cached_logical_calls']} | {row['physical_network_attempts']} | "
            f"{row['retry_attempts']} | {row['actual_total_tokens']:.0f} | {row['reported_reasoning_tokens']:.0f} | "
            f"{row['mean_provider_latency_ms']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## 9. Human validity audit",
            "",
            "`diagnostics/human_audit_sample.json` contains up to ten accepted coordinates per dataset. Two blinded annotators must fill decidability, exclusivity, atomicity, context sufficiency, and leakage fields; no quota is transferred between datasets.",
            "",
            "## 10. Audit trail",
            "",
            "Per-turn payload, cache namespace/source, usage, finish reason, physical attempts, retries, latency, and provider request ID remain in `turns/agent_turns.jsonl`. Dataset checkpoints are in `diagnostics/dataset_checkpoints.json`.",
            "",
            "## 11. Paper boundary",
            "",
            "No pooled four-dataset headline accuracy is reported. The unweighted macro is descriptive only. Even a multi-dataset positive pattern cannot retroactively pass the failed preregistered BBEH v3 gate.",
        ]
    )
    return "\n".join(lines) + "\n"


def _rerender_boundary_report(
    root: Path,
    *,
    human_evaluation: dict[str, Any] | None = None,
) -> None:
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    selector = json.loads((root / "selector_funnel.json").read_text(encoding="utf-8"))
    witness = json.loads((root / "witness_analysis.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "reproducibility_manifest.json").read_text(encoding="utf-8"))
    checkpoints_path = root / "diagnostics" / "dataset_checkpoints.json"
    checkpoints = json.loads(checkpoints_path.read_text(encoding="utf-8")) if checkpoints_path.exists() else {}
    if human_evaluation is None:
        evaluation_path = root / "diagnostics" / "human_audit_evaluation.json"
        if evaluation_path.exists():
            human_evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    (root / "report.md").write_text(
        _render_report(
            metrics,
            selector,
            witness,
            manifest,
            checkpoints,
            human_evaluation=human_evaluation,
        ),
        encoding="utf-8",
    )


def _render_report(
    metrics: dict[str, Any],
    selector: dict[str, Any],
    witness: dict[str, Any],
    manifest: dict[str, Any],
    checkpoints: dict[str, Any],
    *,
    human_evaluation: dict[str, Any] | None = None,
) -> str:
    """Render the scientific report; machine validation remains separate."""

    lines = [
        "# CATCH-v3 four-dataset mechanism boundary audit",
        "",
        "## 1. Executive conclusion and recommended decision",
        "",
        "This is a post-failure exploratory audit. It cannot reopen the failed BBEH gate, authorize heldout or confirmation, or support a positive CATCH claim.",
        "",
        "The recommended decision is to use the four domain-specific failure diagnoses below to delimit the paper claim. Do not pool the four datasets into a headline accuracy and do not create a v4 repair cycle.",
        "",
        "## 2. Frozen BBEH-v3 structural-preflight failure",
        "",
        "| Check | Observed | Required | Delta | Denominator | Result |",
        "|---|---:|---:|---:|---:|---|",
        "| Selector ID/ownership validity | 75.0% | 100.0% | -25.0 pp | raw coordinates | fail |",
        "| Eligible three-coordinate packet coverage | 40.0% | 60.0% | -20.0 pp | 20 samples | fail |",
        "| Automated answer leakage | 1 | 0 | +1 | 20 samples | fail |",
        "| Raw-output packet-coverage upper bound | 55.0% | 60.0% | -5.0 pp | 20 samples | impossible |",
        "",
        "Even accepting every validator-dropped coordinate would leave the raw upper bound below the preregistered threshold. This run therefore records `failed_structural_preflight`; it does not change selector, witness, decoder, or leakage rules.",
        "",
        "## 3. Data integrity, sampling, and target headroom",
        "",
        "| Dataset | Frozen revision | Source SHA-256 | Screening pool | Selected disagreements | Checkpoint |",
        "|---|---|---|---:|---:|---|",
    ]
    sources = {str(row.get("dataset")): row for row in manifest.get("dataset_sources", [])}
    screening_manifests = manifest.get("screening_manifests", {})
    selection_manifests = manifest.get("disagreement_manifests", {})
    for dataset in sorted(sources):
        source = sources[dataset]
        short_sha = str(source.get("sha256") or "")[:12]
        lines.append(
            f"| {dataset} | `{source.get('revision')}` | `{short_sha}...` | "
            f"{len(screening_manifests.get(dataset, {}).get('sample_ids', []))} | "
            f"{len(selection_manifests.get(dataset, {}).get('sample_ids', []))} | "
            f"{checkpoints.get(dataset, {}).get('status', 'unknown')} |"
        )
    compatibility = manifest.get("frozen_bbeh_v3_mechanism_compatibility") or {}
    lines.extend(
        [
            "",
            f"Frozen BBEH-v3 mechanism hash match: `{compatibility.get('exact_component_hash_match')}` "
            f"against source run `{compatibility.get('source_run_id', 'unknown')}`. This covers the prompt, "
            "indexed selector validator/decoder, and shared algorithm file; audit-only loaders, lifecycle, and reports are separately versioned.",
            "",
            "The screening pool uses only five shared Stage-A calls. Adaptive-SC8 is therefore `n/a` on the full 100 and is reported only on the gold-free selected disagreement subset.",
            "",
            "| Dataset | SC5 | Candidate oracle | Target oracle | Disagreement rate | Invalid Stage-A outputs | Candidate-count distribution |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    for dataset, row in sorted(metrics.get("screening", {}).items()):
        lines.append(
            f"| {dataset} | {_fmt_rate(row.get('sc5_micro_accuracy'))} | "
            f"{_fmt_rate(row.get('candidate_oracle_micro'))} | {_fmt_rate(row.get('target_oracle_micro'))} | "
            f"{row.get('disagreement_count', 0)}/{row.get('sample_count', 0)} "
            f"({_fmt_rate(row.get('disagreement_rate'))}) | {row.get('invalid_stage_answer_count', 0)} | "
            f"`{json.dumps(row.get('candidate_count_distribution') or {}, sort_keys=True)}` |"
        )

    lines.extend(
        [
            "",
            "## 4. Selector funnel and raw feasibility upper bound",
            "",
            "| Dataset | Top-level parsed | Raw coordinates | Accepted | Dropped | ID/ownership validity | Eligible packets | Raw 3-coordinate upper bound | Leakage |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in sorted(set(metrics.get("screening", {})) | set(selector)):
        row = selector.get(dataset, {})
        lines.append(
            f"| {dataset} | {_fmt_rate(row.get('top_level_parse_rate'))} | {row.get('raw_coordinate_count', 0)} | "
            f"{row.get('accepted_coordinate_count', 0)} | {row.get('dropped_coordinate_count', 0)} | "
            f"{_fmt_rate(row.get('coordinate_reference_validity_rate'))} | "
            f"{row.get('eligible_sample_count', 0)}/{row.get('selector_sample_count', 0)} | "
            f"{_fmt_rate(row.get('raw_three_coordinate_coverage_upper_bound'))} | "
            f"{row.get('answer_leakage_count', 0)} |"
        )
        lines.append(
            f"\nDrop reasons for **{dataset}**: `{json.dumps(row.get('drop_reasons') or {}, ensure_ascii=False, sort_keys=True)}`\n"
        )
    lines.extend(
        [
            "The raw upper bound counts a sample whenever the selector emitted at least three coordinates for any pair before validation. It is an optimistic structural ceiling, not evidence that those coordinates are mutually exclusive, decidable, or correct.",
            "",
            "## 5. Conditional matched-method outcomes",
            "",
            "All rows below are conditional on the preselected Stage-A disagreements; they are not benchmark-level accuracies.",
            "",
            "| Dataset | Method | n | Accuracy | Corrected | Harmed | Overrides | Override precision | Mean actual tokens |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, block in sorted(metrics.get("conditional_disagreement_audit", {}).items()):
        for method, row in sorted(block.get("methods", {}).items()):
            lines.append(
                f"| {dataset} | {method} | {row.get('sample_count', 0)} | {_fmt_rate(row.get('micro_accuracy'))} | "
                f"{row.get('corrected', 0)} | {row.get('harmed', 0)} | {row.get('override_count', 0)} | "
                f"{_fmt_rate(row.get('override_precision'))} | {_fmt_number(row.get('mean_total_tokens'))} |"
            )
    macro = metrics.get("descriptive_unweighted_macro", {})
    lines.extend(
        [
            "",
            "Descriptive, unweighted cross-dataset macro (not a primary estimand): "
            + ", ".join(f"`{method}`={value:.3f}" for method, value in sorted(macro.items())),
            "",
            "## 6. Witness measurement quality",
            "",
            "| Dataset | Calls | Parsed | Valid coordinate rows | Decisive verdicts | Inverse-mapped panel agreement | False-pass correlation | Left-only share |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset in sorted(set(metrics.get("screening", {})) | set(witness)):
        row = witness.get(dataset, {})
        dependence = row.get("false_pass_dependence") or {}
        position = row.get("position_and_agreement") or {}
        lines.append(
            f"| {dataset} | {row.get('witness_call_count', 0)} | {_fmt_rate(row.get('top_level_parse_rate'))} | "
            f"{row.get('valid_coordinate_rows', 0)}/{row.get('expected_coordinate_rows', 0)} "
            f"({_fmt_rate(row.get('valid_coordinate_rate'))}) | {_fmt_rate(row.get('decisive_verdict_rate'))} | "
            f"{_fmt_rate(row.get('inverse_mapped_panel_agreement_rate'))} | "
            f"{_fmt_number(dependence.get('bernoulli_correlation'))} | "
            f"{_fmt_rate(position.get('left_only_share_among_decisive'))} |"
        )
    lines.extend(
        [
            "",
            "False-pass correlation is reported only when both Bernoulli marginals have nonzero variance. `n/a` means the empirical correlation is undefined, not independent.",
            "",
            "## 7. Calls, cache, token, and latency accounting",
            "",
            "| Dataset | Logical calls | Cached | Physical attempts | Retries | Total tokens | Throttle wait (s) | Provider latency (s) | Retry backoff (s) | Cache lookup (s) |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for dataset, row in sorted(metrics.get("costs", {}).items()):
        lines.append(
            f"| {dataset} | {row.get('logical_calls', 0)} | {row.get('cached_logical_calls', 0)} | "
            f"{row.get('physical_network_attempts', 0)} | {row.get('retry_attempts', 0)} | "
            f"{_fmt_number(row.get('actual_total_tokens'))} | "
            f"{float(row.get('throttle_wait_ms_total') or 0) / 1000:.1f} | "
            f"{float(row.get('provider_network_latency_ms_total') or 0) / 1000:.1f} | "
            f"{_fmt_number(row.get('retry_backoff_seconds_total'))} | "
            f"{float(row.get('cache_lookup_ms_total') or 0) / 1000:.1f} |"
        )
        lines.append(
            f"\nToken details for **{dataset}**: prompt={_fmt_number(row.get('actual_prompt_tokens'))}, "
            f"completion={_fmt_number(row.get('actual_completion_tokens'))}, "
            f"reasoning={_fmt_number(row.get('reported_reasoning_tokens'))}; "
            f"mean physical-attempt latency={_fmt_number(row.get('mean_provider_latency_ms'))} ms; "
            f"cache sources=`{json.dumps(row.get('cache_source_counts') or {}, sort_keys=True)}`.\n"
        )
    lines.extend(["", "## 8. Human validity audit", ""])
    if human_evaluation is None:
        lines.append(
            "Pending: `diagnostics/human_audit_sample.json` contains up to ten accepted coordinates per dataset, with the answer region removed from the source question. Two blinded annotators must complete all five Boolean criteria and an adjudicator must resolve every disagreement."
        )
    else:
        lines.extend(
            [
                f"Audit complete: `{human_evaluation.get('audit_complete')}`; sample contract matches: `{human_evaluation.get('sample_contract_matches')}`; exploratory validity thresholds met: `{human_evaluation.get('mechanism_validity_thresholds_met')}`.",
                "",
                "| Dataset | n | Decidable | Exclusive | Atomic | Context sufficient | Leakage | Cohen kappa |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for dataset, row in sorted((human_evaluation.get("by_dataset") or {}).items()):
            rates = row.get("rates") or {}
            lines.append(
                f"| {dataset} | {row.get('item_count', 0)} | {_fmt_rate(rates.get('decidable_from_source'))} | "
                f"{_fmt_rate(rates.get('mutually_exclusive'))} | {_fmt_rate(rates.get('atomic'))} | "
                f"{_fmt_rate(rates.get('context_sufficient'))} | {_fmt_rate(rates.get('answer_leakage'))} | "
                f"{_fmt_number(row.get('cohen_kappa_pooled_non_leakage'))} |"
            )
    lines.extend(["", "## 9. Representative failure cases and audit links", ""])
    lines.append(
        "See `failure_cases.md` for sample IDs and drop reasons. The complete source chain is in `turns/agent_turns.jsonl`, `turns/router_decisions.jsonl`, `sample_outcomes.jsonl`, and the per-dataset checkpoints under `diagnostics/`."
    )
    lines.extend(["", "## 10. Preregistered mechanism attribution", ""])
    for dataset, conclusion in sorted((metrics.get("mechanism_assessment") or {}).get("by_dataset", {}).items()):
        lines.append(f"- **{dataset}**: `{conclusion}`")
    lines.extend(
        [
            "",
            "Interpretation order is fixed: low eligibility indicates indexed-trace contrast formation failure; adequate eligibility with weak decisive/agreement rates indicates witness measurement failure; signal with corrected <= harmed indicates outcome misalignment; PairJudge-3 >= CATCH indicates no complexity benefit over direct target selection.",
            "",
            "## 11. Reproducibility and paper boundary",
            "",
            "`reproducibility_manifest.json` freezes source revisions and hashes, sampling manifests, prompt/schema versions, and the full configuration hash. Every turn records payload, request source, namespace, cache hit, usage, finish reason, physical attempts, retry timeline, latency, and provider request ID.",
            "",
            "No result in this exploratory audit restores confirmatory status. A signal confined to one dataset is a task-condition boundary; even a multi-dataset positive pattern remains exploratory and does not authorize heldout, confirmation, or a general CATCH claim.",
        ]
    )
    return "\n".join(lines) + "\n"


def _fmt_rate(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.3f}"


def _fmt_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.1f}"


def _group(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get(key) or "unknown")].append(row)
    return grouped


def _mean(values) -> float:
    rows = list(values)
    return sum(rows) / len(rows) if rows else 0.0


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
