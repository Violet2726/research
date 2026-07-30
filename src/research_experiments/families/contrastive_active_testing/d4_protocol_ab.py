"""D4 Stage-A 输出协议 A/B 的冻结验收分析。"""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import beta


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
            "stage_a_only": stage_a_only,
            "artifact_roles_only": artifact_roles_only,
            "scores_valid": scores_valid,
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


def _load_run_arm(run_path: str | Path) -> dict[str, Any]:
    root = Path(run_path)
    turns_path = root / "turns" / "agent_turns.jsonl"
    predictions_path = root / "views" / "predictions.jsonl"
    manifest_path = root / "manifest.json"
    for path in (turns_path, predictions_path, manifest_path):
        if not path.is_file():
            raise FileNotFoundError(f"D4 protocol A/B run artifact is missing: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
            "evaluation_role": (manifest.get("phase_metadata") or {}).get("evaluation_role"),
            "stage_a_protocol": (manifest.get("d4_output") or {}).get("stage_a_protocol"),
            "phase_name": manifest.get("phase_name"),
            "run_id": manifest.get("run_id"),
            "experiment_name": manifest.get("experiment_name"),
            "run_status": manifest.get("run_status"),
            "dataset_error_count": len(list(manifest.get("dataset_errors") or [])),
            "sample_count": manifest.get("sample_count"),
        },
    }


def _clopper_upper(failures: int, total: int) -> float:
    if total <= 0:
        return 1.0
    if failures >= total:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, total - failures))


def _minimum_zero_failure_trials(target: float) -> int | None:
    if not 0.0 < target < 1.0:
        return None
    return math.floor(math.log(0.05) / math.log(1.0 - target)) + 1
