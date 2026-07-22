"""CATCH-Kernel 预声明且不阻断后续实验的机制识别矩阵。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

STUDY_ARMS = {
    "representation": (
        "tagged_free_text",
        "generic_json_object",
        "local_strict_schema",
    ),
    "contract_verifier_2x2": (
        "model_obligation__model_verifier",
        "human_obligation__model_verifier",
        "model_obligation__human_or_deterministic_verifier",
        "human_obligation__human_or_deterministic_verifier",
    ),
    "jurisdiction": (
        "unscoped_model_verifier",
        "adapter_then_model_fallback",
        "adapter_conflict_abstain",
        "capability_routed_kernel",
    ),
    "proof_completeness": (
        "support_only",
        "support_and_refutation",
        "support_refutation_and_complete_obligations",
    ),
}

ARM_EXECUTION_MODES = {
    "tagged_free_text": "api_required",
    "generic_json_object": "api_required",
    "local_strict_schema": "api_required",
    "model_obligation__model_verifier": "api_required",
    "human_obligation__model_verifier": "human_annotation_plus_api",
    "model_obligation__human_or_deterministic_verifier": "api_plus_human_or_deterministic",
    "human_obligation__human_or_deterministic_verifier": "human_or_deterministic",
    "unscoped_model_verifier": "offline_v2_replay",
    "adapter_then_model_fallback": "offline_v2_replay",
    "adapter_conflict_abstain": "offline_counterfactual_replay",
    "capability_routed_kernel": "kernel_run_or_replay",
    "support_only": "offline_kernel_proof_replay",
    "support_and_refutation": "offline_kernel_proof_replay",
    "support_refutation_and_complete_obligations": "offline_kernel_proof_replay",
}

_KERNEL_REPLAY_ARMS = {
    "capability_routed_kernel",
    "support_only",
    "support_and_refutation",
    "support_refutation_and_complete_obligations",
}


def write_kernel_mechanism_template(
    intensive_audit_path: str | Path,
    output_path: str | Path,
) -> dict[str, Any]:
    audit = json.loads(Path(intensive_audit_path).read_text(encoding="utf-8"))
    cases = list(audit.get("cases") or [])
    if not cases:
        raise ValueError("The kernel mechanism template requires at least one audit case.")
    rows = []
    for case in cases:
        for study, arms in STUDY_ARMS.items():
            for arm in arms:
                rows.append(
                    {
                        "run_id": case.get("run_id"),
                        "sample_id": case["sample_id"],
                        "dataset": case["dataset"],
                        "task": case.get("task"),
                        "case_class": case.get("case_class"),
                        "candidate_set_hash": case.get("candidate_set_hash"),
                        "study": study,
                        "arm": arm,
                        "execution_mode": ARM_EXECUTION_MODES[arm],
                        "completed": False,
                        "initial_correct": case.get("sc_correct"),
                        "final_correct": None,
                        "syntax_valid": None,
                        "schema_valid": None,
                        "semantic_valid": None,
                        "contract_correct": None,
                        "answer_link_correct": None,
                        "mandatory_obligations_complete": None,
                        "commitment_direction_correct": None,
                        "verifier_outcome_correct": None,
                        "jurisdiction_bound": None,
                        "proof_complete": None,
                        "override_accepted": None,
                        "token_count": None,
                        "first_failure_layer": None,
                        "notes": "",
                        "result_source": None,
                    }
                )
    payload = {
        "schema_version": "catch_kernel_mechanism_matrix_v1",
        "development_only": True,
        "non_blocking": True,
        "study_arms": {key: list(value) for key, value in STUDY_ARMS.items()},
        "case_count": len(cases),
        "row_count": len(rows),
        "rows": rows,
    }
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def ingest_kernel_mechanism_results(
    matrix_path: str | Path,
    result_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge arm runner outputs while enforcing case, arm, and candidate identity."""

    payload = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])
    keyed = {_mechanism_key(item): item for item in rows}
    if len(keyed) != len(rows):
        raise ValueError("Mechanism matrix contains duplicate case-arm rows.")
    result_fields = {
        "final_correct",
        "syntax_valid",
        "schema_valid",
        "semantic_valid",
        "contract_correct",
        "answer_link_correct",
        "mandatory_obligations_complete",
        "commitment_direction_correct",
        "verifier_outcome_correct",
        "jurisdiction_bound",
        "proof_complete",
        "override_accepted",
        "token_count",
        "first_failure_layer",
        "notes",
    }
    imported = 0
    for result_path in result_paths:
        source = Path(result_path)
        results = _read_result_rows(source)
        seen: set[tuple[str, str, str, str, str]] = set()
        for result in results:
            key = _mechanism_key(result)
            if key in seen:
                raise ValueError(f"Duplicate mechanism result in {source}: {key}")
            seen.add(key)
            target = keyed.get(key)
            if target is None:
                raise ValueError(f"Mechanism result does not belong to the frozen matrix: {key}")
            expected_hash = target.get("candidate_set_hash")
            actual_hash = result.get("candidate_set_hash")
            if expected_hash and actual_hash != expected_hash:
                raise ValueError(f"Candidate-set hash mismatch for {key}")
            unknown = set(result) - {
                "run_id",
                "sample_id",
                "dataset",
                "study",
                "arm",
                "candidate_set_hash",
                "gold_blind",
                *result_fields,
            }
            if unknown:
                raise ValueError(f"Unknown mechanism result fields for {key}: {sorted(unknown)}")
            for field in result_fields:
                if field in result:
                    target[field] = result[field]
            target["completed"] = True
            target["result_source"] = source.resolve().as_posix()
            target["gold_blind"] = result.get("gold_blind")
            imported += 1
    payload["rows"] = rows
    payload["imported_result_count"] = imported
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def write_kernel_run_mechanism_results(
    matrix_path: str | Path,
    run_paths: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Materialize Kernel and proof-decoder arms from completed D1 artifacts.

    The three proof-completeness arms reuse one frozen proof set and therefore
    never trigger another model request.  Gold is consulted only for offline
    scoring after each decoder decision has been made.
    """

    matrix = json.loads(Path(matrix_path).read_text(encoding="utf-8"))
    planned = {
        (str(row.get("dataset")), str(row.get("sample_id")), str(row.get("arm"))): row
        for row in matrix.get("rows") or []
        if row.get("arm") in _KERNEL_REPLAY_ARMS
    }
    routers: dict[tuple[str, str], dict[str, Any]] = {}
    predictions: dict[tuple[str, str], dict[str, Any]] = {}
    source_by_key: dict[tuple[str, str], str] = {}
    for raw_path in run_paths:
        run = Path(raw_path)
        for row in _read_jsonl(run / "turns" / "router_decisions.jsonl"):
            if row.get("protocol_version") != "catch_kernel_v1":
                continue
            key = (str(row.get("dataset")), str(row.get("sample_id")))
            if key in routers:
                raise ValueError(f"Duplicate Kernel router across runs: {key}")
            routers[key] = row
            source_by_key[key] = run.resolve().as_posix()
        for row in _read_jsonl(run / "views" / "predictions.jsonl"):
            if row.get("method_name") != "catch_kernel":
                continue
            key = (str(row.get("dataset")), str(row.get("sample_id")))
            predictions[key] = row

    results: list[dict[str, Any]] = []
    for (dataset, sample_id, arm), target in sorted(planned.items()):
        key = (dataset, sample_id)
        router = routers.get(key)
        prediction = predictions.get(key)
        if router is None or prediction is None:
            continue
        actual_hash = _candidate_set_hash(router)
        expected_hash = str(target.get("candidate_set_hash") or "")
        if expected_hash and actual_hash != expected_hash:
            raise ValueError(f"Kernel candidate-set hash mismatch for {key}")
        final_key, override, proof_complete = _kernel_arm_decision(router, arm)
        gold_keys = set(router.get("gold_candidate_keys") or ())
        if not gold_keys and router.get("gold_candidate_key") is not None:
            gold_keys.add(router.get("gold_candidate_key"))
        final_correct = final_key in gold_keys
        bindings = [
            item
            for item in dict(router.get("verifier_bindings") or {}).values()
            if isinstance(item, dict)
        ]
        kernel_decision = dict(router.get("kernel_decision") or {})
        results.append(
            {
                "run_id": target.get("run_id"),
                "sample_id": sample_id,
                "dataset": dataset,
                "study": target.get("study"),
                "arm": arm,
                "candidate_set_hash": actual_hash,
                "gold_blind": True,
                "final_correct": final_correct,
                "syntax_valid": prediction.get("syntax_validity"),
                "schema_valid": prediction.get("schema_validity"),
                "mandatory_obligations_complete": bool(proof_complete),
                "jurisdiction_bound": bool(bindings) and all(
                    item.get("binding_status") == "BOUND" for item in bindings
                ),
                "proof_complete": bool(proof_complete),
                "override_accepted": bool(override),
                "token_count": prediction.get("total_tokens_per_question"),
                "first_failure_layer": kernel_decision.get("failure_layer"),
                "notes": f"offline_kernel_proof_replay:{source_by_key[key]}",
            }
        )
    target_path = Path(output_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    target_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    return {
        "schema_version": "catch_kernel_run_mechanism_results_v1",
        "result_count": len(results),
        "covered_case_count": len({(row["dataset"], row["sample_id"]) for row in results}),
        "arms": sorted({str(row["arm"]) for row in results}),
        "output": target_path.resolve().as_posix(),
    }


def summarize_kernel_mechanism(
    results_path: str | Path,
    output_path: str | Path | None = None,
) -> dict[str, Any]:
    payload = json.loads(Path(results_path).read_text(encoding="utf-8"))
    rows = list(payload.get("rows") or [])
    if not rows:
        raise ValueError("Kernel mechanism results contain no rows.")
    cells: dict[str, Any] = {}
    for study, arms in STUDY_ARMS.items():
        cells[study] = {}
        for arm in arms:
            selected = [item for item in rows if item.get("study") == study and item.get("arm") == arm]
            completed = [item for item in selected if item.get("completed") is True]
            cells[study][arm] = {
                "planned": len(selected),
                "completed": len(completed),
                "completion_rate": _ratio(len(completed), len(selected)),
                "accuracy": _rate(completed, "final_correct"),
                "wrong_to_correct": sum(
                    item.get("initial_correct") is False and item.get("final_correct") is True for item in completed
                ),
                "correct_to_wrong": sum(
                    item.get("initial_correct") is True and item.get("final_correct") is False for item in completed
                ),
                "syntax_validity": _rate(completed, "syntax_valid"),
                "schema_validity": _rate(completed, "schema_valid"),
                "semantic_validity": _rate(completed, "semantic_valid"),
                "contract_accuracy": _rate(completed, "contract_correct"),
                "answer_link_accuracy": _rate(completed, "answer_link_correct"),
                "obligation_completeness": _rate(completed, "mandatory_obligations_complete"),
                "commitment_direction_accuracy": _rate(completed, "commitment_direction_correct"),
                "verifier_outcome_accuracy": _rate(completed, "verifier_outcome_correct"),
                "jurisdiction_coverage": _rate(completed, "jurisdiction_bound"),
                "proof_completeness": _rate(completed, "proof_complete"),
                "mean_tokens": _mean_numeric(completed, "token_count"),
                "first_failure_counts": _counts(completed, "first_failure_layer"),
            }
    completed_rows = sum(item.get("completed") is True for item in rows)
    matrix_complete = completed_rows == len(rows)
    summary = {
        "schema_version": "catch_kernel_mechanism_summary_v1",
        "development_only": True,
        "non_blocking": True,
        "planned_rows": len(rows),
        "completed_rows": completed_rows,
        "matrix_complete": matrix_complete,
        "cells": cells,
        "attribution": _attribution(cells) if matrix_complete else None,
        "interpretation": _interpretation(cells) if matrix_complete else "incomplete_mechanism_matrix",
    }
    if output_path is not None:
        target = Path(output_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _attribution(cells: dict[str, Any]) -> dict[str, float]:
    factorial = cells["contract_verifier_2x2"]
    end_to_end = factorial["model_obligation__model_verifier"]["accuracy"]
    human_contract = factorial["human_obligation__model_verifier"]["accuracy"]
    human_verifier = factorial["model_obligation__human_or_deterministic_verifier"]["accuracy"]
    oracle = factorial["human_obligation__human_or_deterministic_verifier"]["accuracy"]
    return {
        "contract_or_compiler_loss": oracle - human_verifier,
        "verifier_loss": oracle - human_contract,
        "end_to_end_loss": oracle - end_to_end,
        "interaction_or_decoder_residual": oracle - end_to_end - (oracle - human_verifier) - (oracle - human_contract),
    }


def _interpretation(cells: dict[str, Any]) -> str:
    factorial = cells["contract_verifier_2x2"]
    end_to_end = factorial["model_obligation__model_verifier"]["accuracy"]
    human_contract = factorial["human_obligation__model_verifier"]["accuracy"]
    oracle = factorial["human_obligation__human_or_deterministic_verifier"]["accuracy"]
    kernel = cells["jurisdiction"]["capability_routed_kernel"]["accuracy"]
    fallback = cells["jurisdiction"]["adapter_then_model_fallback"]["accuracy"]
    if kernel > fallback:
        return "verifier_jurisdiction_improves_decision_safety"
    if human_contract > end_to_end:
        return "obligation_compilation_bottleneck"
    if human_contract < oracle:
        return "homogeneous_verifier_structural_boundary"
    return "no_single_dominant_bottleneck"


def _rate(rows: list[dict[str, Any]], field: str) -> float:
    values = [bool(item[field]) for item in rows if item.get(field) is not None]
    return _ratio(sum(values), len(values))


def _mean_numeric(rows: list[dict[str, Any]], field: str) -> float:
    values = [float(item[field]) for item in rows if item.get(field) is not None]
    return _ratio(sum(values), len(values))


def _counts(rows: list[dict[str, Any]], field: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for item in rows:
        value = str(item.get(field) or "unlabeled")
        values[value] = values.get(value, 0) + 1
    return dict(sorted(values.items()))


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _mechanism_key(item: dict[str, Any]) -> tuple[str, str, str, str, str]:
    return (
        str(item.get("run_id") or ""),
        str(item.get("dataset") or ""),
        str(item.get("sample_id") or ""),
        str(item.get("study") or ""),
        str(item.get("arm") or ""),
    )


def _kernel_arm_decision(router: dict[str, Any], arm: str) -> tuple[str | None, bool, bool]:
    anchor_key = str(router.get("anchor_key") or "") or None
    if arm == "capability_routed_kernel" or arm == "support_refutation_and_complete_obligations":
        decision = dict(router.get("kernel_decision") or {})
        public_to_key = dict(router.get("candidate_public_to_answer_class_key") or {})
        challenger = public_to_key.get(decision.get("challenger_id"))
        override = decision.get("decision") == "OVERRIDE" and challenger is not None
        return challenger if override else anchor_key, override, override

    required = {
        str(item.get("obligation_id"))
        for item in (router.get("task_semantics") or {}).get("mandatory_obligation_templates") or []
        if item.get("required") is not False
    }
    proofs_by_candidate: dict[str, list[dict[str, Any]]] = {}
    for proof in router.get("proof_results") or []:
        if isinstance(proof, dict):
            proofs_by_candidate.setdefault(str(proof.get("candidate_key_anon") or ""), []).append(proof)
    passing: list[tuple[str, bool]] = []
    for candidate, proofs in proofs_by_candidate.items():
        support = [proof for proof in proofs if not str(proof.get("test_id") or "").endswith("_refute_anchor")]
        refutation = [proof for proof in proofs if not str(proof.get("test_id") or "").endswith("_support")]
        support_ok = bool(support) and all(_proof_passes(proof) for proof in support)
        refutation_ok = bool(refutation) and all(_proof_passes(proof) for proof in refutation)
        covered = {
            item
            for proof in proofs
            for item in str(proof.get("obligation_id") or "").split("+")
            if item
        }
        complete = required.issubset(covered)
        if support_ok and (arm == "support_only" or refutation_ok):
            passing.append((candidate, complete))
    if len(passing) != 1:
        return anchor_key, False, False
    public_to_key = dict(router.get("candidate_public_to_answer_class_key") or {})
    candidate, complete = passing[0]
    return public_to_key.get(candidate), True, complete


def _proof_passes(proof: dict[str, Any]) -> bool:
    return bool(
        proof.get("status") == "PASS"
        and proof.get("provenance_valid")
        and proof.get("entailment_valid")
        and proof.get("obligation_valid")
        and proof.get("sufficiency_valid")
    )


def _candidate_set_hash(router: dict[str, Any]) -> str:
    payload = {
        "candidate_answer_nodes": router.get("candidate_answer_nodes") or {},
        "candidate_public_to_answer_class_key": router.get("candidate_public_to_answer_class_key") or {},
        "public_pairs": router.get("public_pairs") or [],
    }
    raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _read_result_rows(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.casefold() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and isinstance(payload.get("rows"), list):
        return list(payload["rows"])
    raise ValueError(f"Unsupported mechanism result format: {path}")
