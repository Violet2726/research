"""CATCH-Cert v2 与 CATCH-Kernel 运行的离线因果账本。"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Any


def build_causal_ledger(
    run_dirs: Iterable[str | Path],
    output_dir: str | Path,
    *,
    matched_safe_count: int = 64,
) -> dict[str, Any]:
    """Materialize one first-failure record per sample without API calls."""

    rows: list[dict[str, Any]] = []
    for raw_dir in run_dirs:
        run_dir = Path(raw_dir)
        predictions = _read_jsonl(run_dir / "views" / "predictions.jsonl")
        routers = {
            (str(item.get("dataset")), str(item.get("sample_id"))): item
            for item in _read_jsonl(run_dir / "turns" / "router_decisions.jsonl")
        }
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for prediction in predictions:
            grouped.setdefault((str(prediction.get("dataset")), str(prediction.get("sample_id"))), []).append(
                prediction
            )
        for key, sample_predictions in grouped.items():
            by_method = {str(item.get("method_name")): item for item in sample_predictions}
            main_name = next(
                (name for name in ("catch_kernel", "catch_cert_v2", "catch_cert", "catch") if name in by_method),
                None,
            )
            if main_name is None or "sc_5" not in by_method:
                continue
            rows.append(
                _ledger_row(
                    run_dir=run_dir,
                    sc=by_method["sc_5"],
                    main=by_method[main_name],
                    router=routers.get(key, {}),
                )
            )

    rows.sort(key=lambda item: (item["dataset"], item.get("task") or "", item["sample_id"]))
    recoverable = [item for item in rows if item["case_class"] == "recoverable_wrong"]
    unrecovered = [item for item in recoverable if not item["final_correct"]]
    harms = [item for item in rows if item["case_class"] == "harm"]
    safe_pool = [item for item in rows if item["case_class"] == "sc_correct_abstention"]
    matched_safe, match_diagnostics = _matched_safe_rows(
        [*recoverable, *harms],
        safe_pool,
        count=matched_safe_count,
    )
    intensive = [*recoverable, *harms, *matched_safe]
    intensive_ids = {(item["run_id"], item["dataset"], item["sample_id"]) for item in intensive}
    for item in rows:
        item["in_intensive_audit"] = (item["run_id"], item["dataset"], item["sample_id"]) in intensive_ids

    first_failure = Counter(item["first_failure_layer"] for item in rows)
    recoverable_failures = Counter(item["first_failure_layer"] for item in unrecovered)
    transitions = Counter(item["transition"] for item in rows)
    summary = {
        "schema_version": "catch_kernel_causal_ledger_v1",
        "run_count": len({item["run_id"] for item in rows}),
        "sample_count": len(rows),
        "recoverable_wrong_count": len(recoverable),
        "harm_count": len(harms),
        "matched_sc_correct_abstention_count": len(matched_safe),
        "safe_match_diagnostics": match_diagnostics,
        "intensive_audit_count": len(intensive_ids),
        "transition_counts": dict(sorted(transitions.items())),
        "stratified_results": _stratified_results(rows),
        "first_failure_counts": dict(sorted(first_failure.items())),
        "recoverable_headroom_loss": dict(sorted(recoverable_failures.items())),
        "first_failure_bookkeeping": _headroom_decomposition(recoverable, unrecovered),
        "counterfactual_decomposition": _counterfactual_decomposition(
            [item for item in intensive if item["case_class"] == "recoverable_wrong"]
        ),
        "non_blocking": True,
        "interpretation": "Every failure remains evidence; no layer-level count blocks later experiments.",
    }
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / "causal_ledger.jsonl", rows)
    (output / "causal_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "intensive_audit.json").write_text(
        json.dumps(
            {
                "schema_version": "catch_kernel_intensive_audit_v1",
                "cases": [item for item in rows if item["in_intensive_audit"]],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary


def _ledger_row(
    *,
    run_dir: Path,
    sc: dict[str, Any],
    main: dict[str, Any],
    router: dict[str, Any],
) -> dict[str, Any]:
    sc_correct = float(sc.get("score") or 0) > 0
    final_correct = float(main.get("score") or 0) > 0
    candidate_oracle = bool(main.get("candidate_oracle_correct"))
    target_oracle = bool(main.get("target_oracle_correct"))
    override = bool(main.get("override_accepted"))
    transition = f"{'correct' if sc_correct else 'wrong'}→{'correct' if final_correct else 'wrong'}"
    recoverable = not sc_correct and candidate_oracle
    if recoverable:
        case_class = "recoverable_wrong"
    elif sc_correct and not final_correct:
        case_class = "harm"
    elif sc_correct and not override:
        case_class = "sc_correct_abstention"
    elif not candidate_oracle:
        case_class = "candidate_oracle_failure"
    else:
        case_class = "other"
    contract_status, contract_reason = _contract_semantics_status(main, router)
    first_layer, reason = _first_failure(
        sc_correct=sc_correct,
        final_correct=final_correct,
        candidate_oracle=candidate_oracle,
        target_oracle=target_oracle,
        main=main,
        router=router,
        contract_status=contract_status,
        contract_reason=contract_reason,
    )
    return {
        "run_id": str(main.get("run_id") or run_dir.name),
        "run_dir": str(run_dir.resolve()),
        "dataset": str(main.get("dataset")),
        "sample_id": str(main.get("sample_id")),
        "task": main.get("task") or router.get("task"),
        "protocol_version": main.get("protocol_version") or router.get("protocol_version"),
        "method_name": main.get("method_name"),
        "phase_name": main.get("phase_name") or router.get("phase_name") or run_dir.parent.name,
        "case_class": case_class,
        "transition": transition,
        "sc_correct": sc_correct,
        "final_correct": final_correct,
        "candidate_oracle_correct": candidate_oracle,
        "target_oracle_correct": target_oracle,
        "override_accepted": override,
        "resolver": main.get("resolver"),
        "task_family": main.get("task_family"),
        "query_operator": main.get("query_operator"),
        "adapter_kind": main.get("adapter_kind"),
        "candidate_set_hash": _candidate_set_hash(router),
        "contract_semantics_status": contract_status,
        "contract_semantics_reason": contract_reason,
        "answer_link_coverage": float(main.get("answer_link_coverage") or 0),
        "obligation_coverage": float(main.get("obligation_coverage") or 0),
        "jurisdiction_coverage": float(main.get("verifier_jurisdiction_coverage") or 0),
        "proof_completeness": float(main.get("proof_completeness") or 0),
        "first_failure_layer": first_layer,
        "first_failure_reason": reason,
        "counterfactual_contract_correct": None,
        "counterfactual_verifier_correct": None,
        "counterfactual_certificate_correct": None,
        "counterfactual_complete_proof": None,
        "adjudicated_first_failure_layer": None,
        "counterfactual_interaction_notes": "",
        "counterfactual_annotation_status": "pending",
        "human_notes": "",
    }


def _first_failure(
    *,
    sc_correct: bool,
    final_correct: bool,
    candidate_oracle: bool,
    target_oracle: bool,
    main: dict[str, Any],
    router: dict[str, Any],
    contract_status: str,
    contract_reason: str,
) -> tuple[str, str]:
    if not candidate_oracle:
        return "candidate", "candidate_oracle_unavailable"
    if not target_oracle:
        return "target", "correct_candidate_not_targeted"
    if not sc_correct and final_correct:
        return "none", "headroom_recovered"
    if contract_status == "KNOWN_MISMATCH":
        return "contract", contract_reason
    protocol_error = str(router.get("certificate_protocol_error") or "")
    if protocol_error:
        return "compiler", protocol_error
    if float(main.get("answer_link_coverage") or 0) < 1:
        return "answer_link", "answer_link_incomplete"
    if float(main.get("obligation_coverage") or 0) < 1:
        return "obligation", "mandatory_obligation_incomplete"
    adapter_results = router.get("adapter_results") or {}
    statuses = {str(item.get("execution_status")) for item in adapter_results.values() if isinstance(item, dict)}
    if "CONFLICT" in statuses:
        return "adapter", "adapter_conflict"
    if main.get("protocol_version") == "catch_kernel_v1" and "UNSUPPORTED" in statuses:
        return "jurisdiction", "executable_jurisdiction_unsupported"
    if sc_correct and not final_correct:
        return "verifier_or_kernel", "false_pass_harm"
    resolver = str(main.get("resolver") or "")
    if "verifier" in resolver or "ambiguous" in resolver:
        return "verifier", resolver
    if not bool(main.get("override_accepted")):
        return "decoder", resolver or "abstention"
    return "verifier_or_decoder", resolver or "wrong_override"


def _contract_semantics_status(
    main: dict[str, Any],
    router: dict[str, Any],
) -> tuple[str, str]:
    if str(main.get("protocol_version") or router.get("protocol_version")) == "catch_kernel_v1":
        return "REGISTRY_DEFINED", "typed_task_semantics"
    task = str(main.get("task") or router.get("task") or "").casefold()
    mismatches = {
        "object_placements": "objective_final_state_used_for_observer_belief_query",
        "team_allocation": "exact_set_constraint_used_for_utility_argmax",
        "murder_mysteries": "holistic_comparison_used_for_means_motive_opportunity_conjunction",
    }
    if task in mismatches:
        return "KNOWN_MISMATCH", mismatches[task]
    source = str(router.get("audit_source_question") or "").casefold()
    if task == "word_sorting" and "new alphabet" in source:
        return "KNOWN_MISMATCH", "conventional_sort_used_for_source_defined_alphabet"
    return "NOT_AUDITED", "requires_task_semantics_adjudication"


def _headroom_decomposition(
    recoverable_rows: list[dict[str, Any]],
    unrecovered_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    counts = Counter(item["first_failure_layer"] for item in unrecovered_rows)
    total = len(recoverable_rows)
    return {
        "status": "heuristic_first_failure_bookkeeping_not_causal",
        "recoverable_count": total,
        "recovered_count": total - len(unrecovered_rows),
        "unrecovered_count": len(unrecovered_rows),
        "layers": {
            key: {"count": value, "share": value / total if total else 0.0} for key, value in sorted(counts.items())
        },
    }


def _stratified_results(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for field in ("phase_name", "dataset"):
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row.get(field) or "unknown"), []).append(row)
        output[field] = {
            key: {
                "sample_count": len(values),
                "recoverable_wrong_count": sum(item.get("case_class") == "recoverable_wrong" for item in values),
                "harm_count": sum(item.get("case_class") == "harm" for item in values),
                "transition_counts": dict(sorted(Counter(item["transition"] for item in values).items())),
            }
            for key, values in sorted(grouped.items())
        }
    return output


def summarize_counterfactual_ledger(
    ledger_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate completed human counterfactuals and compute additive attribution."""

    rows = _read_jsonl(Path(ledger_path))
    intensive = [item for item in rows if item.get("in_intensive_audit")]
    recoverable = [item for item in intensive if item.get("case_class") == "recoverable_wrong"]
    summary = {
        "schema_version": "catch_kernel_counterfactual_summary_v1",
        "intensive_case_count": len(intensive),
        "recoverable_case_count": len(recoverable),
        "decomposition": _counterfactual_decomposition(recoverable),
        "non_blocking": True,
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _counterfactual_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = (
        "counterfactual_contract_correct",
        "counterfactual_verifier_correct",
        "counterfactual_certificate_correct",
        "counterfactual_complete_proof",
    )
    completed = [
        item
        for item in rows
        if all(item.get(field) is not None for field in fields) and item.get("adjudicated_first_failure_layer")
    ]
    if len(completed) != len(rows):
        return {
            "status": "pending_counterfactual_adjudication",
            "required": len(rows),
            "completed": len(completed),
            "remaining": len(rows) - len(completed),
            "additive_layer_counts": None,
            "counterfactual_rescue_counts": None,
        }
    allowed = {"contract", "compiler", "verifier", "adapter", "decoder", "interaction", "none"}
    invalid = [item["sample_id"] for item in completed if item["adjudicated_first_failure_layer"] not in allowed]
    if invalid:
        return {
            "status": "invalid_adjudicated_layer",
            "invalid_sample_ids": invalid,
            "additive_layer_counts": None,
        }
    layers = Counter(str(item["adjudicated_first_failure_layer"]) for item in completed)
    return {
        "status": "complete",
        "required": len(rows),
        "completed": len(completed),
        "additive_layer_counts": dict(sorted(layers.items())),
        "counterfactual_rescue_counts": {field: sum(item.get(field) is True for item in completed) for field in fields},
        "interaction_count": layers.get("interaction", 0),
        "interpretation": (
            "Layer counts are additive only because each case received one adjudicated first causal layer; "
            "the four intervention outcomes remain non-additive and are reported separately."
        ),
    }


def _matched_safe_rows(
    target_rows: list[dict[str, Any]],
    safe_pool: list[dict[str, Any]],
    *,
    count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Match safe abstentions to the task/obligation distribution of risk cases."""

    if count <= 0 or not safe_pool:
        return [], {"requested": max(0, count), "selected": 0, "exact_stratum_matches": 0}
    target_counts = Counter(_audit_stratum(item) for item in target_rows)
    total_target = sum(target_counts.values()) or 1
    raw_quotas = {key: count * value / total_target for key, value in target_counts.items()}
    quotas = {key: int(value) for key, value in raw_quotas.items()}
    remainder = count - sum(quotas.values())
    for key in sorted(raw_quotas, key=lambda item: (raw_quotas[item] - quotas[item], str(item)), reverse=True):
        if remainder <= 0:
            break
        quotas[key] += 1
        remainder -= 1
    pools: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for item in safe_pool:
        pools.setdefault(_audit_stratum(item), []).append(item)
    for items in pools.values():
        items.sort(key=lambda item: _stable_key(item["dataset"], item.get("task"), item["sample_id"]))
    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, str, str]] = set()
    exact = 0
    for stratum, quota in sorted(quotas.items()):
        for item in pools.get(stratum, [])[:quota]:
            selected.append(item)
            selected_ids.add((item["run_id"], item["dataset"], item["sample_id"]))
            exact += 1
    fallback = sorted(
        (item for item in safe_pool if (item["run_id"], item["dataset"], item["sample_id"]) not in selected_ids),
        key=lambda item: _stable_key(item["dataset"], item.get("task"), item["sample_id"]),
    )
    selected.extend(fallback[: max(0, count - len(selected))])
    return selected[:count], {
        "requested": count,
        "selected": min(count, len(selected)),
        "exact_stratum_matches": exact,
        "fallback_matches": max(0, min(count, len(selected)) - exact),
        "stratum_fields": ["dataset", "task", "query_operator", "adapter_kind"],
    }


def _audit_stratum(item: dict[str, Any]) -> tuple[str, str, str, str]:
    return (
        str(item.get("dataset") or ""),
        str(item.get("task") or ""),
        str(item.get("query_operator") or ""),
        str(item.get("adapter_kind") or ""),
    )


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in rows),
        encoding="utf-8",
    )


def _stable_key(dataset: str, task: Any, sample_id: str) -> str:
    return hashlib.sha256(f"{dataset}\0{task or ''}\0{sample_id}".encode()).hexdigest()


def _candidate_set_hash(router: dict[str, Any]) -> str:
    payload = {
        "candidate_answer_nodes": router.get("candidate_answer_nodes") or {},
        "candidate_public_to_answer_class_key": router.get("candidate_public_to_answer_class_key") or {},
        "public_pairs": router.get("public_pairs") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
