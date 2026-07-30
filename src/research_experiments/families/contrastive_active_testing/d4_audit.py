"""CATCH-Kernel D4 的独立盲审与 metamorphic 验收门。"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_BLINDED_FORBIDDEN_FIELDS = {
    "candidate",
    "candidates",
    "stage_a_candidates",
    "gold",
    "solver_answer",
    "final_accuracy",
    "anchor",
    "vote_counts",
}


@dataclass(frozen=True)
class KernelAuditResult:
    kernel_id: str
    item_count: int
    annotator_count: int
    cohen_kappa: float | None
    kappa_estimable: bool
    gwet_ac1: float
    raw_agreement: float
    critical_semantic_error_rate: float
    adjudicated_ir_validity: float
    unexplained_high_severity_false_pass: int
    candidate_blindness_passed: bool
    metamorphic_pass_rate: float
    passed: bool


def evaluate_d4_human_audit(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema") != "catch_d4_blind_ir_audit_v1":
        raise ValueError("Invalid D4 blind-audit schema.")
    items = payload.get("items")
    if not isinstance(items, list):
        raise ValueError("D4 blind audit must contain an items list.")
    allowed_top_level = {"schema", "blinding_contract", "items", "metamorphic", "required_kernels"}
    if set(payload) - allowed_top_level:
        raise ValueError("D4 blind audit contains unrecognized top-level fields.")
    blinding = payload.get("blinding_contract")
    if not isinstance(blinding, dict):
        raise ValueError("D4 blind audit requires an explicit blinding_contract.")
    hidden = {
        str(item).casefold().replace("-", "_")
        for item in list(blinding.get("hidden") or [])
    }
    blinding_contract_valid = bool(
        _BLINDED_FORBIDDEN_FIELDS.issubset(hidden)
        and blinding.get("two_independent_annotators") is True
        and blinding.get("third_person_adjudication") is True
    )
    metamorphic = dict(payload.get("metamorphic") or {})
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    item_ids: list[str] = []
    ir_hashes: list[str] = []
    item_schema_valid = True
    for item in items:
        if not isinstance(item, dict):
            raise ValueError("D4 blind-audit items must be objects.")
        kernel_id = str(item.get("kernel_id") or "")
        if not kernel_id:
            raise ValueError("D4 blind-audit item is missing kernel_id.")
        item_id = str(item.get("item_id") or "")
        ir_hash = str(item.get("ir_hash") or "")
        capability_id = str(item.get("capability_id") or "")
        item_schema_valid &= bool(
            item_id
            and capability_id
            and re_full_sha256(ir_hash)
            and isinstance(item.get("visible_fields"), list)
            and isinstance(item.get("annotations"), list)
            and isinstance(item.get("adjudicated_ir_valid"), bool)
            and isinstance(item.get("critical_semantic_error"), bool)
            and isinstance(item.get("unexplained_high_severity_false_pass"), bool)
        )
        item_ids.append(item_id)
        ir_hashes.append(ir_hash)
        grouped[kernel_id].append(item)

    requested = payload.get("required_kernels")
    required = (
        {str(item) for item in requested}
        if isinstance(requested, list) and requested
        else {
            "sequence_trace_kernel_v1",
            "event_state_kernel_v1",
            "constraint_calculator_kernel_v1",
        }
    )
    known_kernels = {
        "sequence_trace_kernel_v1",
        "event_state_kernel_v1",
        "constraint_calculator_kernel_v1",
    }
    required_kernels_valid = bool(required and required.issubset(known_kernels))
    results = []
    for kernel_id, rows in sorted(grouped.items()):
        annotators = sorted(
            {
                str(annotation.get("annotator_id") or "")
                for row in rows
                for annotation in list(row.get("annotations") or [])
                if isinstance(annotation, dict) and annotation.get("annotator_id")
            }
        )
        paired = []
        row_annotator_sets: list[set[str]] = []
        annotation_rows_complete = True
        adjudication_complete = True
        for row in rows:
            annotations = [item for item in list(row.get("annotations") or []) if isinstance(item, dict)]
            annotation_ids = [str(item.get("annotator_id") or "") for item in annotations]
            if len(annotations) != 2 or any(not item for item in annotation_ids) or len(set(annotation_ids)) != 2:
                annotation_rows_complete = False
                continue
            if any(
                not isinstance(item.get("ir_valid"), bool)
                or not isinstance(item.get("critical_error"), bool)
                for item in annotations
            ):
                annotation_rows_complete = False
                continue
            ordered = sorted(annotations, key=lambda item: str(item["annotator_id"]))
            pair = (bool(ordered[0]["ir_valid"]), bool(ordered[1]["ir_valid"]))
            paired.append(pair)
            row_annotator_sets.append(set(annotation_ids))
            adjudicator_id = str(row.get("adjudicator_id") or "")
            if pair[0] != pair[1]:
                if not adjudicator_id or adjudicator_id in set(annotation_ids):
                    adjudication_complete = False
            elif bool(row.get("adjudicated_ir_valid")) != pair[0]:
                adjudication_complete = False
        kappa_estimable = _kappa_estimable(paired)
        kappa = cohen_kappa(paired) if kappa_estimable else None
        ac1 = gwet_ac1(paired)
        raw_agreement = sum(left == right for left, right in paired) / len(paired) if paired else 0.0
        critical = sum(bool(row.get("critical_semantic_error")) for row in rows)
        valid = sum(bool(row.get("adjudicated_ir_valid")) for row in rows)
        false_pass = sum(bool(row.get("unexplained_high_severity_false_pass")) for row in rows)
        leaked = any(
            bool(
                {
                    str(field).casefold().replace("-", "_")
                    for field in list(row.get("visible_fields") or [])
                }
                & _BLINDED_FORBIDDEN_FIELDS
            )
            or bool(_find_forbidden_keys(row))
            for row in rows
        )
        meta = dict(metamorphic.get(kernel_id) or {})
        meta_total = max(0, int(meta.get("total") or 0))
        meta_passed = max(0, int(meta.get("passed") or 0))
        meta_counts_valid = 0 <= meta_passed <= meta_total
        meta_rate = meta_passed / meta_total if meta_total and meta_counts_valid else 0.0
        item_count = len(rows)
        critical_rate = critical / item_count if item_count else 1.0
        validity = valid / item_count if item_count else 0.0
        conditions = {
            "item_count": item_count >= 60,
            "annotators": len(annotators) == 2,
            "paired_annotations": annotation_rows_complete and len(paired) == item_count,
            "consistent_annotator_pair": bool(row_annotator_sets)
            and all(item == row_annotator_sets[0] for item in row_annotator_sets),
            "agreement": ac1 >= 0.80,
            "critical_error": critical_rate <= 0.02,
            "validity": validity >= 0.95,
            "false_pass": false_pass == 0,
            "candidate_blindness": not leaked,
            "adjudication_complete": adjudication_complete
            and all("adjudicated_ir_valid" in row for row in rows),
        }
        results.append(
            KernelAuditResult(
                kernel_id=kernel_id,
                item_count=item_count,
                annotator_count=len(annotators),
                cohen_kappa=kappa,
                kappa_estimable=kappa_estimable,
                gwet_ac1=ac1,
                raw_agreement=raw_agreement,
                critical_semantic_error_rate=critical_rate,
                adjudicated_ir_validity=validity,
                unexplained_high_severity_false_pass=false_pass,
                candidate_blindness_passed=not leaked,
                metamorphic_pass_rate=meta_rate,
                passed=all(conditions.values()),
            )
        )
    present = {row.kernel_id for row in results}
    by_kernel = {row.kernel_id: row for row in results}
    global_conditions = {
        "blinding_contract": blinding_contract_valid,
        "item_schema": item_schema_valid,
        "unique_item_ids": bool(item_ids) and len(item_ids) == len(set(item_ids)),
        "unique_ir_hashes": bool(ir_hashes) and len(ir_hashes) == len(set(ir_hashes)),
        "known_kernels": set(grouped).issubset(known_kernels),
        "required_kernels": required_kernels_valid,
    }
    return {
        "schema": "catch_d4_blind_ir_audit_assessment_v1",
        "kernel_results": [asdict(row) for row in results],
        "required_kernels": sorted(required),
        "missing_kernels": sorted(required - present),
        "agreement_gate": "gwet_ac1_at_least_0.80",
        "cohen_kappa_interpretation": "reported_when_estimable_not_used_as_a_high_prevalence_gate",
        "global_conditions": global_conditions,
        "passed": all(global_conditions.values())
        and not (required - present)
        and all(by_kernel[kernel_id].passed for kernel_id in required),
    }


def cohen_kappa(pairs: list[tuple[bool, bool]]) -> float:
    if not pairs:
        return 0.0
    observed = sum(left == right for left, right in pairs) / len(pairs)
    left_true = sum(left for left, _ in pairs) / len(pairs)
    right_true = sum(right for _, right in pairs) / len(pairs)
    expected = left_true * right_true + (1.0 - left_true) * (1.0 - right_true)
    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1.0 - expected)


def gwet_ac1(pairs: list[tuple[bool, bool]]) -> float:
    """Binary two-rater AC1, stable when valid IRs have very high prevalence."""

    if not pairs:
        return 0.0
    observed = sum(left == right for left, right in pairs) / len(pairs)
    positive = (sum(left for left, _ in pairs) + sum(right for _, right in pairs)) / (2.0 * len(pairs))
    chance = 2.0 * positive * (1.0 - positive)
    if chance >= 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - chance) / (1.0 - chance)


def _kappa_estimable(pairs: list[tuple[bool, bool]]) -> bool:
    if not pairs:
        return False
    return len({left for left, _ in pairs}) == 2 and len({right for _, right in pairs}) == 2


def _find_forbidden_keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            normalized = str(key).casefold().replace("-", "_")
            if normalized in _BLINDED_FORBIDDEN_FIELDS:
                found.add(normalized)
            found.update(_find_forbidden_keys(item))
    elif isinstance(value, list):
        for item in value:
            found.update(_find_forbidden_keys(item))
    return found


def forbidden_blind_fields() -> tuple[str, ...]:
    return tuple(sorted(_BLINDED_FORBIDDEN_FIELDS))


def build_d4_gate_evidence(
    predictions: list[dict[str, Any]],
    *,
    predictions_sha256: str,
    predictions_path: str | None = None,
    turns: list[dict[str, Any]] | None = None,
    turns_sha256: str | None = None,
    turns_path: str | None = None,
    run_manifest: dict[str, Any] | None = None,
    run_manifest_sha256: str | None = None,
    run_manifest_path: str | None = None,
    human_audit: dict[str, Any] | None = None,
    human_audit_sha256: str | None = None,
    human_audit_path: str | None = None,
) -> dict[str, Any]:
    """Build frozen, traceable development evidence from shadow outcomes.

    This consumes only an independent post-freeze calibration run recorded
    under the development phase.  It never turns public engineering runs or
    confirmation outcomes into an activation threshold.
    """

    rows = [row for row in predictions if row.get("method_name") == "catch_kernel_d4"]
    if not rows:
        raise ValueError("D4 gate evidence requires catch_kernel_d4 prediction rows.")
    sample_keys = [
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in rows
    ]
    if any(not dataset or not sample_id for dataset, sample_id in sample_keys) or len(sample_keys) != len(
        set(sample_keys)
    ):
        raise ValueError("D4 gate evidence requires one unique prediction per dataset/sample pair.")
    if not re_full_sha256(predictions_sha256):
        raise ValueError("D4 gate evidence requires the SHA-256 of the development predictions file.")
    audit_assessment = evaluate_d4_human_audit(human_audit) if human_audit is not None else None
    audit_by_kernel = {
        str(row["kernel_id"]): row
        for row in list((audit_assessment or {}).get("kernel_results") or [])
        if isinstance(row, dict) and row.get("kernel_id")
    }
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        capability_id = str(row.get("d4_capability_id") or "")
        if capability_id:
            grouped[capability_id].append(row)
    capabilities: dict[str, Any] = {}
    from research_experiments.families.contrastive_active_testing.kernel_d4 import capability_spec

    for capability_id, capability_rows in sorted(grouped.items()):
        route = str(capability_rows[0].get("d4_route") or "")
        kernel_id = str(capability_rows[0].get("d4_kernel_id") or "")
        spec = capability_spec(capability_id)
        if spec is None:
            raise ValueError(f"D4 capability {capability_id} is not in the frozen registry.")
        if any(
            str(row.get("d4_route") or "") != route
            or str(row.get("d4_kernel_id") or "") != kernel_id
            for row in capability_rows
        ):
            raise ValueError(f"D4 capability {capability_id} mixes route or kernel labels.")
        if route != spec["route"] or kernel_id != spec["kernel_id"]:
            raise ValueError(f"D4 capability {capability_id} does not match its frozen route/kernel identity.")
        shadow_rows = [row for row in capability_rows if row.get("d4_shadow_score") is not None]
        override_rows = [row for row in shadow_rows if bool(row.get("d4_shadow_override"))]
        corrections = sum(bool(row.get("d4_shadow_correction")) for row in override_rows)
        harms = sum(bool(row.get("d4_shadow_harm")) for row in override_rows)
        correct_overrides = sum(float(row.get("d4_shadow_score") or 0.0) == 1.0 for row in override_rows)
        meta_total = 0
        meta_passed = 0
        meta_item_passed = 0
        for row in shadow_rows:
            statuses = dict(
                (row.get("d4_proof_package") or {}).get("metamorphic_transformation_status") or {}
            )
            executed = [value for value in statuses.values() if value not in {"NOT_APPLICABLE", "NOT_RUN"}]
            meta_total += len(executed)
            meta_passed += sum(value == "PASSED" for value in executed)
            meta_item_passed += int(
                bool(executed)
                and row.get("d4_metamorphic_checks_passed") is True
                and all(value == "PASSED" for value in executed)
            )
        audit = dict(audit_by_kernel.get(kernel_id) or {})
        semantic = route == "SEMANTIC_EXECUTABLE"
        semantic_audit_frozen = bool(
            semantic
            and audit.get("passed")
            and human_audit_sha256
            and re_full_sha256(human_audit_sha256)
        )
        evidence_frozen = bool(
            shadow_rows
            and meta_total > 0
            and meta_passed == meta_total
            and meta_item_passed == len(shadow_rows)
            and (semantic_audit_frozen if semantic else True)
        )
        capabilities[capability_id] = {
            "route": route,
            "kernel_id": kernel_id,
            "evidence_frozen": evidence_frozen,
            "sample_count": len(capability_rows),
            "solver_unique_count": len(shadow_rows),
            "override_count": len(override_rows),
            "correct_override_count": correct_overrides,
            "correction_count": corrections,
            "harm_count": harms,
            "neutral_override_count": len(override_rows) - corrections - harms,
            "coverage": len(override_rows) / len(capability_rows),
            "solver_coverage": len(shadow_rows) / len(capability_rows),
            "audit_sample_count": int(audit.get("item_count") or 0),
            "inter_rater_agreement": audit.get("gwet_ac1"),
            "inter_rater_agreement_metric": "gwet_ac1" if audit else None,
            "cohen_kappa": audit.get("cohen_kappa"),
            "kappa_estimable": bool(audit.get("kappa_estimable", False)),
            "critical_semantic_error_rate": float(audit.get("critical_semantic_error_rate", 1.0)),
            "adjudicated_ir_validity": float(audit.get("adjudicated_ir_validity") or 0.0),
            "unexplained_high_severity_false_pass": int(
                audit.get("unexplained_high_severity_false_pass") or 0
            ),
            "metamorphic_test_count": meta_total,
            "metamorphic_pass_rate": meta_passed / meta_total if meta_total else 0.0,
            "metamorphic_item_count": len(shadow_rows),
            "metamorphic_item_pass_count": meta_item_passed,
        }
    run_ids = sorted({str(row.get("run_id") or "") for row in rows if row.get("run_id")})
    if len(run_ids) != 1:
        raise ValueError("D4 gate evidence must come from exactly one development run.")
    manifest_identity = _manifest_identity(run_manifest)
    if run_manifest is not None and (
        manifest_identity.get("phase_name") != "development"
        or manifest_identity.get("kernel_revision") != "d4_proof_carrying_v1"
        or manifest_identity.get("run_id") != run_ids[0]
    ):
        raise ValueError("D4 gate evidence source manifest is not the matching D4 development run.")
    if turns is not None and any(str(row.get("run_id") or "") != run_ids[0] for row in turns):
        raise ValueError("D4 gate evidence turns do not all belong to the prediction run.")
    unsigned = {
        "schema": "catch_d4_gate_evidence_v1",
        "status": "frozen_independent_calibration_evidence",
        "source": {
            "phase": "development",
            "run_ids": run_ids,
            "predictions_sha256": predictions_sha256,
            "predictions_path": predictions_path,
            "turns_sha256": turns_sha256,
            "turns_path": turns_path,
            "run_manifest_sha256": run_manifest_sha256,
            "run_manifest_path": run_manifest_path,
            "run_manifest_identity": manifest_identity,
            "human_audit_sha256": human_audit_sha256,
            "human_audit_path": human_audit_path,
        },
        "capabilities": capabilities,
    }
    return {**unsigned, "sha256": _sha256(unsigned)}


def validate_d4_gate_evidence(
    payload: dict[str, Any],
    *,
    verify_source_files: bool = False,
) -> dict[str, Any]:
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    capabilities = payload.get("capabilities")
    source = payload.get("source")
    rows_valid = True
    if not isinstance(capabilities, dict):
        capabilities = {}
        rows_valid = False
    for capability_id, raw in capabilities.items():
        if not capability_id or not isinstance(raw, dict):
            rows_valid = False
            continue
        from research_experiments.families.contrastive_active_testing.kernel_d4 import capability_spec

        spec = capability_spec(str(capability_id))
        total = int(raw.get("sample_count") or 0)
        solver = int(raw.get("solver_unique_count") or 0)
        overrides = int(raw.get("override_count") or 0)
        correct = int(raw.get("correct_override_count") or 0)
        corrections = int(raw.get("correction_count") or 0)
        harms = int(raw.get("harm_count") or 0)
        neutral = int(raw.get("neutral_override_count") or 0)
        coverage = float(raw.get("coverage") or 0.0)
        solver_coverage = float(raw.get("solver_coverage") or 0.0)
        meta_count = int(raw.get("metamorphic_test_count") or 0)
        meta_rate = float(raw.get("metamorphic_pass_rate") or 0.0)
        meta_items = int(raw.get("metamorphic_item_count") or 0)
        meta_item_passes = int(raw.get("metamorphic_item_pass_count") or 0)
        rows_valid &= (
            spec is not None
            and raw.get("route") == spec["route"]
            and raw.get("kernel_id") == spec["kernel_id"]
            and 0 <= overrides <= solver <= total
            and 0 <= correct <= overrides
            and 0 <= corrections <= overrides
            and 0 <= harms <= overrides
            and 0 <= neutral <= overrides
            and correct == corrections
            and corrections + harms + neutral == overrides
            and coverage == (overrides / total if total else 0.0)
            and solver_coverage == (solver / total if total else 0.0)
            and 0.0 <= meta_rate <= 1.0
            and meta_count >= 0
            and meta_items == solver
            and 0 <= meta_item_passes <= meta_items
            and (
                not bool(raw.get("evidence_frozen"))
                or (
                    meta_count > 0
                    and meta_rate == 1.0
                    and meta_item_passes == meta_items
                )
            )
        )
    conditions = {
        "schema": payload.get("schema") == "catch_d4_gate_evidence_v1",
        "status": payload.get("status") == "frozen_independent_calibration_evidence",
        "top_level_keys": set(payload) == {"schema", "status", "source", "capabilities", "sha256"},
        "hash": payload.get("sha256") == _sha256(unsigned),
        "source": isinstance(source, dict)
        and source.get("phase") == "development"
        and re_full_sha256(str(source.get("predictions_sha256") or "")),
        "capability_rows": rows_valid and bool(capabilities),
    }
    if verify_source_files:
        predictions_path = Path(str((source or {}).get("predictions_path") or ""))
        conditions["predictions_file"] = predictions_path.is_file() and hashlib.sha256(
            predictions_path.read_bytes()
        ).hexdigest() == str((source or {}).get("predictions_sha256") or "")
        turns_path = Path(str((source or {}).get("turns_path") or ""))
        manifest_path = Path(str((source or {}).get("run_manifest_path") or ""))
        conditions["turns_file"] = turns_path.is_file() and hashlib.sha256(
            turns_path.read_bytes()
        ).hexdigest() == str((source or {}).get("turns_sha256") or "")
        conditions["run_manifest_file"] = manifest_path.is_file() and hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest() == str((source or {}).get("run_manifest_sha256") or "")
        audit_sha = str((source or {}).get("human_audit_sha256") or "")
        audit_path = Path(str((source or {}).get("human_audit_path") or ""))
        conditions["human_audit_file"] = (
            not audit_sha
            or (audit_path.is_file() and hashlib.sha256(audit_path.read_bytes()).hexdigest() == audit_sha)
        )
        recomputed_matches = False
        if (
            conditions["predictions_file"]
            and conditions["turns_file"]
            and conditions["run_manifest_file"]
            and conditions["human_audit_file"]
        ):
            try:
                prediction_rows = [
                    json.loads(line)
                    for line in predictions_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                turn_rows = [
                    json.loads(line)
                    for line in turns_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
                manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                source_identity_valid = _source_run_is_valid_development(
                    prediction_rows,
                    turn_rows,
                    manifest_payload,
                )
                audit_payload = (
                    json.loads(audit_path.read_text(encoding="utf-8"))
                    if audit_sha
                    else None
                )
                recomputed = build_d4_gate_evidence(
                    prediction_rows,
                    predictions_sha256=str(source.get("predictions_sha256") or ""),
                    predictions_path=predictions_path.resolve().as_posix(),
                    turns=turn_rows,
                    turns_sha256=str(source.get("turns_sha256") or ""),
                    turns_path=turns_path.resolve().as_posix(),
                    run_manifest=manifest_payload,
                    run_manifest_sha256=str(source.get("run_manifest_sha256") or ""),
                    run_manifest_path=manifest_path.resolve().as_posix(),
                    human_audit=audit_payload,
                    human_audit_sha256=audit_sha or None,
                    human_audit_path=audit_path.resolve().as_posix() if audit_sha else None,
                )
                recomputed_matches = source_identity_valid and recomputed == payload
            except (OSError, ValueError, json.JSONDecodeError):
                recomputed_matches = False
        conditions["recomputed_from_sources"] = recomputed_matches
    return {"passed": all(conditions.values()), "conditions": conditions}


def write_d4_gate_evidence(
    predictions_path: str | Path,
    output_path: str | Path,
    *,
    human_audit_path: str | Path | None = None,
) -> dict[str, Any]:
    predictions_target = Path(predictions_path)
    run_root = predictions_target.parent.parent
    turns_target = run_root / "turns" / "agent_turns.jsonl"
    manifest_target = run_root / "manifest.json"
    for required in (predictions_target, turns_target, manifest_target):
        if not required.is_file():
            raise FileNotFoundError(f"D4 gate evidence source artifact is missing: {required}")
    predictions = [
        json.loads(line)
        for line in predictions_target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    turns = [
        json.loads(line)
        for line in turns_target.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    manifest = json.loads(manifest_target.read_text(encoding="utf-8"))
    audit = None
    audit_sha = None
    if human_audit_path:
        audit_target = Path(human_audit_path)
        audit = json.loads(audit_target.read_text(encoding="utf-8"))
        audit_sha = hashlib.sha256(audit_target.read_bytes()).hexdigest()
    payload = build_d4_gate_evidence(
        predictions,
        predictions_sha256=hashlib.sha256(predictions_target.read_bytes()).hexdigest(),
        predictions_path=predictions_target.resolve().as_posix(),
        turns=turns,
        turns_sha256=hashlib.sha256(turns_target.read_bytes()).hexdigest(),
        turns_path=turns_target.resolve().as_posix(),
        run_manifest=manifest,
        run_manifest_sha256=hashlib.sha256(manifest_target.read_bytes()).hexdigest(),
        run_manifest_path=manifest_target.resolve().as_posix(),
        human_audit=audit,
        human_audit_sha256=audit_sha,
        human_audit_path=Path(human_audit_path).resolve().as_posix() if human_audit_path else None,
    )
    validation = validate_d4_gate_evidence(payload, verify_source_files=True)
    if not validation.get("passed"):
        raise ValueError(
            "D4 gate evidence source is not a completed independent post-freeze calibration run: "
            f"{validation.get('conditions', {})}"
        )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def _manifest_identity(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return {}
    phase = dict(payload.get("phase_metadata") or {})
    return {
        "run_id": str(payload.get("run_id") or ""),
        "experiment_name": str(payload.get("experiment_name") or ""),
        "phase_name": str(payload.get("phase_name") or ""),
        "kernel_revision": str(payload.get("kernel_revision") or ""),
        "evaluation_role": str(phase.get("evaluation_role") or ""),
        "frozen_config_sha256": str(payload.get("frozen_config_sha256") or ""),
        "run_status": str(payload.get("run_status") or ""),
        "sample_count": int(payload.get("sample_count") or 0),
        "dataset_error_count": len(list(payload.get("dataset_errors") or [])),
    }


def _source_run_is_valid_development(
    predictions: list[dict[str, Any]],
    turns: list[dict[str, Any]],
    manifest: dict[str, Any],
) -> bool:
    identity = _manifest_identity(manifest)
    run_id = identity.get("run_id")
    if not (
        run_id
        and identity.get("phase_name") == "development"
        and identity.get("kernel_revision") == "d4_proof_carrying_v1"
        and identity.get("evaluation_role") == "d4_independent_calibration_after_method_freeze"
        and identity.get("run_status") == "completed"
        and identity.get("dataset_error_count") == 0
    ):
        return False
    d4_rows = [row for row in predictions if row.get("method_name") == "catch_kernel_d4"]
    if not d4_rows or any(str(row.get("run_id") or "") != run_id for row in d4_rows):
        return False
    if identity.get("sample_count") != len(d4_rows):
        return False
    if any(str(row.get("run_id") or "") != run_id for row in turns):
        return False
    compiler_expected = int(dict(manifest.get("protocol") or {}).get("resample_candidates") or 0)
    stage_expected = int(dict(manifest.get("protocol") or {}).get("stage_candidates") or 0)
    stage_agents: dict[tuple[str, str], set[int]] = defaultdict(set)
    resample_agents: dict[tuple[str, str], set[int]] = defaultdict(set)
    compiler_counts: dict[tuple[str, str], int] = defaultdict(int)
    compiler_agents: dict[tuple[str, str], set[int]] = defaultdict(set)
    for row in turns:
        key = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        role = row.get("role")
        if role == "stage_a_solver":
            stage_agents[key].add(int(row.get("agent_id") or 0))
        elif role == "independent_resample":
            resample_agents[key].add(int(row.get("agent_id") or 0))
        elif role == "d4_source_compiler":
            compiler_counts[key] += 1
            compiler_agents[key].add(int(row.get("agent_id") or 0))
        else:
            return False
    for row in d4_rows:
        key = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        expected = compiler_expected if row.get("d4_route") == "SEMANTIC_EXECUTABLE" else 0
        logical_expected = stage_expected + expected
        if (
            stage_agents.get(key) != set(range(1, stage_expected + 1))
            or resample_agents.get(key) != set(range(1, compiler_expected + 1))
            or compiler_counts.get(key, 0) != expected
            or compiler_agents.get(key, set()) != set(range(1, expected + 1))
            or int(row.get("logical_calls_per_question") or 0) != logical_expected
        ):
            return False
    semantic_count = sum(row.get("d4_route") == "SEMANTIC_EXECUTABLE" for row in d4_rows)
    return (
        set(stage_agents) == {
            (str(row.get("dataset") or ""), str(row.get("sample_id") or "")) for row in d4_rows
        }
        and set(resample_agents) == set(stage_agents)
        and sum(compiler_counts.values()) == compiler_expected * semantic_count
        and len(turns) == (stage_expected + compiler_expected) * len(d4_rows) + compiler_expected * semantic_count
    )


def re_full_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdefABCDEF" for character in value)


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
