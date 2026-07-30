"""D4 Stage-A 输出协议 A/B 的冻结验收分析。"""

from __future__ import annotations

import hashlib
import json
import math
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import beta

INDEPENDENT_VALIDATION_DATASETS = {
    "bbeh_extension",
    "musr_x",
    "supergpqa_science",
}


def evaluate_output_protocol_ab(
    arms: dict[str, dict[str, list[dict[str, Any]]]],
    *,
    parse_failure_target: float = 0.002,
    accuracy_noninferiority_margin: float = 0.005,
    reference_arm: str = "tagged_text",
) -> dict[str, Any]:
    expected_arms = {"tagged_text", "reasoning_first_json", "answer_first_json"}
    if set(arms) != expected_arms:
        raise ValueError("The output-protocol A/B requires exactly the three preregistered arms.")
    if reference_arm not in arms:
        raise ValueError("The output-protocol A/B is missing its reference arm.")
    summaries = {}
    sources = {}
    sample_keys_by_arm: dict[str, set[tuple[str, str]]] = {}
    for arm, payload in sorted(arms.items()):
        all_turns = list(payload.get("turns", []))
        all_predictions = list(payload.get("predictions", []))
        turns = [row for row in all_turns if row.get("role") == "stage_a_solver"]
        predictions = [row for row in all_predictions if row.get("method_name") == "sc_5"]
        sample_keys_by_arm[arm] = {
            (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
            for row in predictions
        }
        prediction_keys = [
            (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
            for row in predictions
        ]
        turn_keys = [
            (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
            for row in turns
        ]
        turn_counts = Counter(turn_keys)
        agent_ids_by_key: dict[tuple[str, str], set[int]] = {}
        for key in turn_counts:
            agent_ids_by_key[key] = {
                int(row.get("agent_id") or 0)
                for row in turns
                if (str(row.get("dataset") or ""), str(row.get("sample_id") or "")) == key
            }
        unique_predictions = bool(prediction_keys) and len(prediction_keys) == len(set(prediction_keys))
        per_sample_turns_complete = bool(prediction_keys) and set(turn_counts) == set(prediction_keys) and all(
            turn_counts[key] == 5 and agent_ids_by_key[key] == {1, 2, 3, 4, 5}
            for key in prediction_keys
        )
        valid_turn_counts = Counter(
            (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
            for row in turns
            if row.get("protocol_parse_status") == "ok"
        )
        quorum_failures = sum(valid_turn_counts.get(key, 0) < 3 for key in prediction_keys)
        stage_a_only = bool(predictions) and all(
            row.get("output_protocol_ab_only") is True
            and int(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0) == 5
            for row in predictions
        )
        artifact_roles_only = len(turns) == len(all_turns) and len(predictions) == len(all_predictions)
        scores_valid = all(
            isinstance(row.get("score"), (int, float)) and 0.0 <= float(row["score"]) <= 1.0
            for row in predictions
        )
        failures = sum(row.get("protocol_parse_status") != "ok" for row in turns)
        accuracy = sum(float(row.get("score") or 0.0) for row in predictions) / len(predictions) if predictions else 0.0
        summaries[arm] = {
            "stage_a_turn_count": len(turns),
            "parse_failure_count": failures,
            "parse_failure_rate": failures / len(turns) if turns else 1.0,
            "sc5_sample_count": len(predictions),
            "sc5_accuracy": accuracy,
            "parse_failure_one_sided_95_upper": _clopper_upper(failures, len(turns)),
            "unique_predictions": unique_predictions,
            "five_turns_per_sample": per_sample_turns_complete,
            "stage_a_quorum_minimum_valid_turns": 3,
            "stage_a_quorum_failure_count": quorum_failures,
            "stage_a_quorum_failure_rate": quorum_failures / len(predictions) if predictions else 1.0,
            "stage_a_quorum_failure_one_sided_95_upper": _clopper_upper(
                quorum_failures,
                len(predictions),
            ),
            "stage_a_only": stage_a_only,
            "artifact_roles_only": artifact_roles_only,
            "scores_valid": scores_valid,
            "failure_diagnostics": diagnose_output_protocol_failures(turns),
        }
        if isinstance(payload.get("source"), dict):
            sources[arm] = dict(payload["source"])
    reference_accuracy = summaries[reference_arm]["sc5_accuracy"]
    reference_keys = sample_keys_by_arm[reference_arm]
    selections_match = bool(reference_keys) and all(
        keys == reference_keys for keys in sample_keys_by_arm.values()
    )
    for _arm, summary in summaries.items():
        summary["accuracy_delta_vs_reference"] = summary["sc5_accuracy"] - reference_accuracy
        summary["parse_target_point_passed"] = summary["parse_failure_rate"] < parse_failure_target
        summary["parse_target_certified"] = (
            summary["parse_failure_one_sided_95_upper"] < parse_failure_target
        )
        summary["accuracy_noninferiority_passed"] = (
            summary["accuracy_delta_vs_reference"] >= -accuracy_noninferiority_margin
        )
        summary["passed"] = bool(
            selections_match
            and summary["unique_predictions"]
            and summary["five_turns_per_sample"]
            and summary["stage_a_only"]
            and summary["artifact_roles_only"]
            and summary["scores_valid"]
            and summary["parse_target_point_passed"]
            and summary["accuracy_noninferiority_passed"]
        )
        summary["certified"] = bool(
            selections_match
            and summary["unique_predictions"]
            and summary["five_turns_per_sample"]
            and summary["stage_a_only"]
            and summary["artifact_roles_only"]
            and summary["scores_valid"]
            and summary["parse_target_certified"]
            and summary["accuracy_noninferiority_passed"]
        )
    answer_first = summaries.get("answer_first_json")
    return {
        "schema": "catch_d4_output_protocol_ab_assessment_v1",
        "reference_arm": reference_arm,
        "parse_failure_target_strictly_below": parse_failure_target,
        "minimum_zero_failure_turns_for_95pct_certification": _minimum_zero_failure_trials(
            parse_failure_target
        ),
        "accuracy_noninferiority_margin": accuracy_noninferiority_margin,
        "accuracy_check_interpretation": "paired_point_estimate_operational_screen_not_confidence_certification",
        "arms": summaries,
        "sources": sources,
        "selection_matched_across_arms": selections_match,
        "answer_first_json_accepted": bool(answer_first and answer_first["passed"]),
        "answer_first_json_certified": bool(answer_first and answer_first["certified"]),
        "answer_first_json_operationally_accepted": bool(answer_first and answer_first["passed"]),
        "answer_first_json_parse_certified": bool(answer_first and answer_first["parse_target_certified"]),
    }


def write_output_protocol_ab_assessment(
    *,
    tagged_text_run: str | Path,
    reasoning_first_json_run: str | Path,
    answer_first_json_run: str | Path,
    output_path: str | Path,
    parse_failure_target: float = 0.002,
    accuracy_noninferiority_margin: float = 0.005,
) -> dict[str, Any]:
    arms = {
        "tagged_text": _load_run_arm(tagged_text_run),
        "reasoning_first_json": _load_run_arm(reasoning_first_json_run),
        "answer_first_json": _load_run_arm(answer_first_json_run),
    }
    result = evaluate_output_protocol_ab(
        arms,
        parse_failure_target=parse_failure_target,
        accuracy_noninferiority_margin=accuracy_noninferiority_margin,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def validate_output_protocol_ab_assessment(
    payload: dict[str, Any],
    *,
    verify_source_files: bool = False,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "reference_arm",
        "parse_failure_target_strictly_below",
        "minimum_zero_failure_turns_for_95pct_certification",
        "accuracy_noninferiority_margin",
        "accuracy_check_interpretation",
        "arms",
        "sources",
        "selection_matched_across_arms",
        "answer_first_json_accepted",
        "answer_first_json_certified",
        "answer_first_json_operationally_accepted",
        "answer_first_json_parse_certified",
    }
    conditions = {
        "schema": payload.get("schema") == "catch_d4_output_protocol_ab_assessment_v1",
        "top_level_keys": set(payload) == expected_keys,
        "three_arms": set(dict(payload.get("arms") or {}))
        == {"tagged_text", "reasoning_first_json", "answer_first_json"},
        "operational_acceptance": payload.get("answer_first_json_operationally_accepted") is True,
        "parse_certification": payload.get("answer_first_json_parse_certified") is True,
        "legacy_aliases_consistent": payload.get("answer_first_json_accepted")
        == payload.get("answer_first_json_operationally_accepted")
        and payload.get("answer_first_json_certified")
        == bool(
            payload.get("answer_first_json_operationally_accepted")
            and payload.get("answer_first_json_parse_certified")
        ),
    }
    if verify_source_files:
        sources = dict(payload.get("sources") or {})
        recomputed_matches = False
        try:
            if set(sources) != {"tagged_text", "reasoning_first_json", "answer_first_json"}:
                raise ValueError("Output-protocol source arms are incomplete.")
            arms = {
                arm: _load_run_arm(str(source.get("run_root") or ""))
                for arm, source in sources.items()
                if isinstance(source, dict)
            }
            recomputed = evaluate_output_protocol_ab(
                arms,
                parse_failure_target=float(payload.get("parse_failure_target_strictly_below")),
                accuracy_noninferiority_margin=float(payload.get("accuracy_noninferiority_margin")),
                reference_arm=str(payload.get("reference_arm") or ""),
            )
            recomputed_matches = recomputed == payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            recomputed_matches = False
        conditions["recomputed_from_sources"] = recomputed_matches
    return {"passed": all(conditions.values()), "conditions": conditions}


def write_tagged_protocol_validation_assessment(
    *,
    run: str | Path,
    output_path: str | Path,
    parse_failure_target: float = 0.01,
    quorum_failure_target: float = 0.01,
    quorum_minimum_valid_turns: int = 3,
) -> dict[str, Any]:
    """Assess a fresh independent tagged-text validation run.

    This artifact is intentionally separate from the inspected three-arm
    development A/B.  It is the only output-protocol artifact eligible for a
    later D4 freeze under the revised tagged-text contract.
    """

    payload = evaluate_tagged_protocol_validation(
        _load_run_arm(run),
        parse_failure_target=parse_failure_target,
        quorum_failure_target=quorum_failure_target,
        quorum_minimum_valid_turns=quorum_minimum_valid_turns,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def evaluate_tagged_protocol_validation(
    arm: dict[str, Any],
    *,
    parse_failure_target: float = 0.01,
    quorum_failure_target: float = 0.01,
    quorum_minimum_valid_turns: int = 3,
) -> dict[str, Any]:
    if not 0.0 < parse_failure_target < 1.0 or not 0.0 < quorum_failure_target < 1.0:
        raise ValueError("D4 tagged validation targets must be probabilities strictly between zero and one.")
    if not 1 <= quorum_minimum_valid_turns <= 5:
        raise ValueError("D4 tagged validation quorum must be between one and five turns.")
    all_turns = list(arm.get("turns") or [])
    all_predictions = list(arm.get("predictions") or [])
    turns = [row for row in all_turns if row.get("role") == "stage_a_solver"]
    predictions = [row for row in all_predictions if row.get("method_name") == "sc_5"]
    prediction_keys = [
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in predictions
    ]
    turn_counts = Counter(
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in turns
    )
    valid_counts = Counter(
        (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        for row in turns
        if row.get("protocol_parse_status") == "ok"
    )
    agent_ids = {
        key: {
            int(row.get("agent_id") or 0)
            for row in turns
            if (str(row.get("dataset") or ""), str(row.get("sample_id") or "")) == key
        }
        for key in turn_counts
    }
    failures = sum(row.get("protocol_parse_status") != "ok" for row in turns)
    quorum_failures = sum(
        valid_counts.get(key, 0) < quorum_minimum_valid_turns
        for key in prediction_keys
    )
    parse_upper = _clopper_upper(failures, len(turns))
    quorum_upper = _clopper_upper(quorum_failures, len(predictions))
    source = dict(arm.get("source") or {})
    conditions = {
        "source_contract": _tagged_validation_source_contract_valid(source),
        "exact_300_samples": len(predictions) == 300,
        "unique_predictions": bool(prediction_keys) and len(prediction_keys) == len(set(prediction_keys)),
        "five_turns_per_sample": bool(prediction_keys)
        and set(turn_counts) == set(prediction_keys)
        and all(
            turn_counts[key] == 5 and agent_ids[key] == {1, 2, 3, 4, 5}
            for key in prediction_keys
        ),
        "stage_a_only": bool(predictions)
        and all(
            row.get("output_protocol_ab_only") is True
            and int(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0) == 5
            for row in predictions
        ),
        "artifact_roles_only": len(turns) == len(all_turns) and len(predictions) == len(all_predictions),
        "scores_valid": all(
            isinstance(row.get("score"), (int, float)) and 0.0 <= float(row["score"]) <= 1.0
            for row in predictions
        ),
        "parse_failure_bound": parse_upper < parse_failure_target,
        "quorum_failure_bound": quorum_upper < quorum_failure_target,
    }
    return {
        "schema": "catch_d4_tagged_protocol_validation_v2",
        "protocol": "tagged_text",
        "prompt_variant": "legacy",
        "parse_failure_target_strictly_below": parse_failure_target,
        "quorum_minimum_valid_turns": quorum_minimum_valid_turns,
        "quorum_failure_target_strictly_below": quorum_failure_target,
        "summary": {
            "stage_a_turn_count": len(turns),
            "parse_failure_count": failures,
            "parse_failure_rate": failures / len(turns) if turns else 1.0,
            "parse_failure_one_sided_95_upper": parse_upper,
            "sc5_sample_count": len(predictions),
            "sc5_accuracy": (
                sum(float(row.get("score") or 0.0) for row in predictions) / len(predictions)
                if predictions
                else 0.0
            ),
            "quorum_failure_count": quorum_failures,
            "quorum_failure_rate": quorum_failures / len(predictions) if predictions else 1.0,
            "quorum_failure_one_sided_95_upper": quorum_upper,
            "failure_diagnostics": diagnose_output_protocol_failures(turns),
        },
        "source": source,
        "conditions": conditions,
        "independent_validation_passed": all(conditions.values()),
    }


def validate_tagged_protocol_validation_assessment(
    payload: dict[str, Any],
    *,
    verify_source_files: bool = False,
) -> dict[str, Any]:
    expected_keys = {
        "schema",
        "protocol",
        "prompt_variant",
        "parse_failure_target_strictly_below",
        "quorum_minimum_valid_turns",
        "quorum_failure_target_strictly_below",
        "summary",
        "source",
        "conditions",
        "independent_validation_passed",
    }
    conditions = {
        "top_level_keys": set(payload) == expected_keys,
        "schema": payload.get("schema") == "catch_d4_tagged_protocol_validation_v2",
        "protocol": payload.get("protocol") == "tagged_text",
        "prompt_variant": payload.get("prompt_variant") == "legacy",
        "frozen_targets": payload.get("parse_failure_target_strictly_below") == 0.01
        and payload.get("quorum_minimum_valid_turns") == 3
        and payload.get("quorum_failure_target_strictly_below") == 0.01,
        "source_contract": _tagged_validation_source_contract_valid(dict(payload.get("source") or {})),
        "passed": payload.get("independent_validation_passed") is True
        and all(dict(payload.get("conditions") or {}).values()),
    }
    if verify_source_files:
        source = dict(payload.get("source") or {})
        recomputed_matches = False
        try:
            arm = _load_run_arm(str(source.get("run_root") or ""))
            recomputed = evaluate_tagged_protocol_validation(
                arm,
                parse_failure_target=float(payload.get("parse_failure_target_strictly_below")),
                quorum_failure_target=float(payload.get("quorum_failure_target_strictly_below")),
                quorum_minimum_valid_turns=int(payload.get("quorum_minimum_valid_turns")),
            )
            recomputed_matches = recomputed == payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            recomputed_matches = False
        conditions["recomputed_from_sources"] = recomputed_matches
    return {"passed": all(conditions.values()), "conditions": conditions}


def _tagged_validation_source_contract_valid(source: dict[str, Any]) -> bool:
    expected_hashes = dict(source.get("expected_selection_sha256") or {})
    selected_hashes = dict(source.get("selected_selection_sha256") or {})
    sealed_manifest_hashes = dict(source.get("sealed_manifest_sha256") or {})
    selected_counts = {
        str(key): int(value)
        for key, value in dict(source.get("selected_sample_counts") or {}).items()
    }
    return bool(
        source.get("kernel_revision") == "d4_proof_carrying_v1"
        and source.get("evaluation_role") == "d4_output_protocol_independent_validation_tagged_v2"
        and source.get("stage_a_protocol") == "tagged_text"
        and source.get("prompt_variant") == "legacy"
        and source.get("phase_name") == "development"
        and source.get("run_id")
        and source.get("run_status") in {"completed", "completed_with_errors"}
        and int(source.get("dataset_error_count") or 0) == 0
        and int(source.get("sample_count") or 0) == 300
        and int(source.get("selected_sample_count") or 0) == 300
        and selected_counts == {dataset: 100 for dataset in INDEPENDENT_VALIDATION_DATASETS}
        and source.get("selection_strategy") == "d4_sealed_manifest_only"
        and source.get("sealed_manifest_split") == "protocol_validation"
        and source.get("sealed_data_ready") is True
        and source.get("validation_data_role")
        == "fresh_independent_protocol_validation_after_prompt_freeze"
        and set(expected_hashes) == INDEPENDENT_VALIDATION_DATASETS
        and expected_hashes == selected_hashes
        and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in expected_hashes.values())
        and set(sealed_manifest_hashes) == INDEPENDENT_VALIDATION_DATASETS
        and all(re.fullmatch(r"[0-9a-f]{64}", str(value)) for value in sealed_manifest_hashes.values())
        and source.get("sealed_manifest_validation_passed") is True
        and source.get("sealed_manifest_files_verified") is True
        and source.get("provider_audit_required") is True
        and source.get("provider_audit_passed") is True
        and re.fullmatch(r"[0-9a-f]{64}", str(source.get("provider_audit_sha256") or ""))
        is not None
        and source.get("provider_audit_file_verified") is True
    )


def _load_run_arm(run_path: str | Path) -> dict[str, Any]:
    root = Path(run_path)
    turns_path = root / "turns" / "agent_turns.jsonl"
    predictions_path = root / "views" / "predictions.jsonl"
    manifest_path = root / "manifest.json"
    for path in (turns_path, predictions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"D4 protocol A/B run artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase_metadata = dict(manifest.get("phase_metadata") or {})
    selected_manifest = dict(manifest.get("selected_sample_manifest") or {})
    expected_selection = {
        str(key): str(value)
        for key, value in dict(phase_metadata.get("expected_selection_sha256") or {}).items()
    }
    selected_selection = {
        str(key): str(dict(value).get("sha256") or "")
        for key, value in selected_manifest.items()
        if isinstance(value, dict)
    }
    sealed_manifests = dict(manifest.get("d4_sealed_manifests") or {})
    sealed_manifest_sha256 = {
        str(key): str(dict(value).get("sha256") or "")
        for key, value in sealed_manifests.items()
        if isinstance(value, dict)
    }
    sealed_manifest_files_verified = bool(sealed_manifests) and all(
        _file_matches_sha256(
            str(dict(value).get("path") or ""),
            str(dict(value).get("sha256") or ""),
        )
        for value in sealed_manifests.values()
        if isinstance(value, dict)
    ) and len(sealed_manifest_sha256) == len(sealed_manifests)
    provider_audit = dict(manifest.get("provider_audit") or {})
    provider_audit_path = str(provider_audit.get("path") or "")
    provider_audit_sha256 = str(provider_audit.get("sha256") or "")
    return {
        "turns": [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines() if line.strip()],
        "predictions": [
            json.loads(line)
            for line in predictions_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ],
        "source": {
            "run_root": root.resolve().as_posix(),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
            "turns_sha256": hashlib.sha256(turns_path.read_bytes()).hexdigest(),
            "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
            "kernel_revision": manifest.get("kernel_revision"),
            "evaluation_role": phase_metadata.get("evaluation_role"),
            "stage_a_protocol": (manifest.get("d4_output") or {}).get("stage_a_protocol"),
            "prompt_variant": (manifest.get("d4_output") or {}).get("prompt_variant") or "legacy",
            "phase_name": manifest.get("phase_name"),
            "run_id": manifest.get("run_id"),
            "experiment_name": manifest.get("experiment_name"),
            "run_status": manifest.get("run_status"),
            "dataset_error_count": len(list(manifest.get("dataset_errors") or [])),
            "sample_count": manifest.get("sample_count"),
            "selected_sample_count": sum(
                int(dict(value).get("count") or 0)
                for value in selected_manifest.values()
                if isinstance(value, dict)
            ),
            "selected_sample_counts": {
                str(key): int(dict(value).get("count") or 0)
                for key, value in selected_manifest.items()
                if isinstance(value, dict)
            },
            "selection_strategy": phase_metadata.get("selection_strategy"),
            "expected_selection_sha256": expected_selection,
            "selected_selection_sha256": selected_selection,
            "sealed_manifest_split": phase_metadata.get("sealed_manifest_split"),
            "sealed_data_ready": phase_metadata.get("sealed_data_ready"),
            "validation_data_role": phase_metadata.get("validation_data_role"),
            "sealed_manifest_sha256": sealed_manifest_sha256,
            "sealed_manifest_validation_passed": bool(sealed_manifests)
            and all(
                dict(dict(value).get("validation") or {}).get("passed") is True
                for value in sealed_manifests.values()
                if isinstance(value, dict)
            )
            and len(sealed_manifest_sha256) == len(sealed_manifests),
            "sealed_manifest_files_verified": sealed_manifest_files_verified,
            "provider_audit_required": provider_audit.get("required") is True,
            "provider_audit_passed": provider_audit.get("status") == "passed"
            and provider_audit.get("passed") is True,
            "provider_audit_sha256": provider_audit_sha256,
            "provider_audit_file_verified": _file_matches_sha256(
                provider_audit_path,
                provider_audit_sha256,
            ),
        },
    }


def _file_matches_sha256(path_value: str, expected_sha256: str) -> bool:
    if re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None:
        return False
    path = Path(path_value)
    try:
        return path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == expected_sha256
    except OSError:
        return False


def _clopper_upper(failures: int, total: int) -> float:
    if total <= 0:
        return 1.0
    if failures >= total:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, total - failures))


def diagnose_output_protocol_failures(turns: list[dict[str, Any]]) -> dict[str, Any]:
    """Separate transport, completion-limit, loop, and stopped parse failures.

    The loop flag is deliberately diagnostic rather than a gate.  It requires
    both a provider ``length`` finish and an extreme repeated-20-token pattern,
    so ordinary long derivations are not mislabeled as API/model loops.
    """

    failed = [row for row in turns if row.get("protocol_parse_status") != "ok"]
    request_failures = [row for row in failed if row.get("request_error")]
    provider_failures = [row for row in failed if not row.get("request_error")]
    completion_limit = [row for row in provider_failures if row.get("raw_finish_reason") == "length"]
    high_repetition = [
        row
        for row in completion_limit
        if _extreme_repetition_loop(str(row.get("assistant_text") or ""))
    ]
    stopped = [row for row in provider_failures if row.get("raw_finish_reason") == "stop"]
    other_finish = [
        row
        for row in provider_failures
        if row.get("raw_finish_reason") not in {"length", "stop"}
    ]
    return {
        "interpretation": "diagnostic_only_not_an_acceptance_gate",
        "protocol_failure_count": len(failed),
        "request_failure_count": len(request_failures),
        "provider_response_failure_count": len(provider_failures),
        "completion_limit_count": len(completion_limit),
        "completion_limit_high_repetition_loop_count": len(high_repetition),
        "completion_limit_non_loop_count": len(completion_limit) - len(high_repetition),
        "stopped_protocol_violation_count": len(stopped),
        "other_finish_reason_count": len(other_finish),
        "failed_response_cache_hit_count": sum(bool(row.get("cache_hit")) for row in failed),
    }


def _extreme_repetition_loop(text: str) -> bool:
    tokens = re.findall(r"\S+", str(text or "").casefold())
    if len(tokens) < 100:
        return False
    ngrams = [tuple(tokens[index : index + 20]) for index in range(len(tokens) - 19)]
    counts = Counter(ngrams)
    repeated_fraction = sum(count - 1 for count in counts.values() if count > 1) / len(ngrams)
    encoded = str(text).encode("utf-8")
    compression_ratio = len(zlib.compress(encoded)) / max(1, len(encoded))
    return repeated_fraction >= 0.50 or compression_ratio <= 0.05


def _minimum_zero_failure_trials(target: float) -> int | None:
    if not 0.0 < target < 1.0:
        return None
    return math.floor(math.log(0.05) / math.log(1.0 - target)) + 1
