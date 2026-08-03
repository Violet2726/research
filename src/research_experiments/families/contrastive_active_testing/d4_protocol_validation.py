"""D4 唯一、哈希链接的 tagged-text 输出协议验证产物。"""

from __future__ import annotations

import hashlib
import json
import re
import zlib
from collections import Counter
from pathlib import Path
from typing import Any

from scipy.stats import beta

INDEPENDENT_VALIDATION_DATASETS = {"bbeh_extension", "musr_x", "supergpqa_science"}
INDEPENDENT_VALIDATION_ROLE = "d4_output_protocol_independent_validation_tagged_v3"


def write_tagged_protocol_validation_assessment(
    *,
    run: str | Path,
    output_path: str | Path,
    parse_failure_target: float = 0.01,
    quorum_failure_target: float = 0.01,
    quorum_minimum_valid_turns: int = 3,
) -> dict[str, Any]:
    payload = evaluate_tagged_protocol_validation(
        _load_run(run),
        parse_failure_target=parse_failure_target,
        quorum_failure_target=quorum_failure_target,
        quorum_minimum_valid_turns=quorum_minimum_valid_turns,
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def evaluate_tagged_protocol_validation(
    run: dict[str, Any],
    *,
    parse_failure_target: float = 0.01,
    quorum_failure_target: float = 0.01,
    quorum_minimum_valid_turns: int = 3,
) -> dict[str, Any]:
    if not 0.0 < parse_failure_target < 1.0 or not 0.0 < quorum_failure_target < 1.0:
        raise ValueError("D4 validation targets must be strictly between zero and one.")
    if quorum_minimum_valid_turns != 3:
        raise ValueError("The frozen D4 tagged validation quorum is exactly three of five.")
    turns = [row for row in run["turns"] if row.get("role") == "stage_a_solver"]
    predictions = [row for row in run["predictions"] if row.get("method_name") == "sc_5"]
    keys = [(str(row.get("dataset") or ""), str(row.get("sample_id") or "")) for row in predictions]
    turn_counts = Counter((str(row.get("dataset") or ""), str(row.get("sample_id") or "")) for row in turns)
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
    quorum_failures = sum(valid_counts.get(key, 0) < quorum_minimum_valid_turns for key in keys)
    parse_upper = _clopper_upper(failures, len(turns))
    quorum_upper = _clopper_upper(quorum_failures, len(predictions))
    source = dict(run["source"])
    conditions = {
        "source_contract": _source_contract_valid(source),
        "exact_300_samples": len(predictions) == 300,
        "unique_predictions": bool(keys) and len(keys) == len(set(keys)),
        "five_turns_per_sample": bool(keys)
        and set(turn_counts) == set(keys)
        and all(turn_counts[key] == 5 and agent_ids[key] == {1, 2, 3, 4, 5} for key in keys),
        "stage_a_only": bool(predictions)
        and all(
            row.get("output_protocol_validation_only") is True
            and int(row.get("logical_calls_per_question") or row.get("calls_per_question") or 0) == 5
            for row in predictions
        ),
        "artifact_roles_only": len(turns) == len(run["turns"]) and len(predictions) == len(run["predictions"]),
        "parse_failure_bound": parse_upper < parse_failure_target,
        "quorum_failure_bound": quorum_upper < quorum_failure_target,
    }
    return {
        "schema": "catch_d4_tagged_protocol_validation_v3",
        "protocol": "tagged_text",
        "cache_key_policy": "request_identity_without_completion_cap_v2",
        "parse_failure_target_strictly_below": parse_failure_target,
        "quorum_minimum_valid_turns": quorum_minimum_valid_turns,
        "quorum_failure_target_strictly_below": quorum_failure_target,
        "summary": {
            "stage_a_turn_count": len(turns),
            "parse_failure_count": failures,
            "parse_failure_rate": failures / len(turns) if turns else 1.0,
            "parse_failure_one_sided_95_upper": parse_upper,
            "sc5_sample_count": len(predictions),
            "sc5_accuracy": sum(float(row.get("score") or 0.0) for row in predictions) / len(predictions)
            if predictions
            else 0.0,
            "quorum_failure_count": quorum_failures,
            "quorum_failure_rate": quorum_failures / len(predictions) if predictions else 1.0,
            "quorum_failure_one_sided_95_upper": quorum_upper,
            "failure_diagnostics": diagnose_tagged_protocol_failures(turns),
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
    expected = {
        "schema",
        "protocol",
        "cache_key_policy",
        "parse_failure_target_strictly_below",
        "quorum_minimum_valid_turns",
        "quorum_failure_target_strictly_below",
        "summary",
        "source",
        "conditions",
        "independent_validation_passed",
    }
    conditions = {
        "top_level_keys": set(payload) == expected,
        "schema": payload.get("schema") == "catch_d4_tagged_protocol_validation_v3",
        "tagged_protocol": payload.get("protocol") == "tagged_text",
        "cache_policy": payload.get("cache_key_policy") == "request_identity_without_completion_cap_v2",
        "frozen_targets": payload.get("parse_failure_target_strictly_below") == 0.01
        and payload.get("quorum_minimum_valid_turns") == 3
        and payload.get("quorum_failure_target_strictly_below") == 0.01,
        "source_contract": _source_contract_valid(dict(payload.get("source") or {})),
        "passed": payload.get("independent_validation_passed") is True
        and all(dict(payload.get("conditions") or {}).values()),
    }
    if verify_source_files:
        try:
            recomputed = evaluate_tagged_protocol_validation(
                _load_run(str(dict(payload.get("source") or {}).get("run_root") or ""))
            )
            conditions["recomputed_from_sources"] = recomputed == payload
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            conditions["recomputed_from_sources"] = False
    return {"passed": all(conditions.values()), "conditions": conditions}


def diagnose_tagged_protocol_failures(turns: list[dict[str, Any]]) -> dict[str, int]:
    failed = [row for row in turns if row.get("protocol_parse_status") != "ok"]
    provider = [row for row in failed if not row.get("request_error")]
    finish = [str(row.get("raw_finish_reason") or "") for row in provider]
    return {
        "protocol_failure_count": len(failed),
        "request_failure_count": sum(bool(row.get("request_error")) for row in failed),
        "completion_limit_count": finish.count("length"),
        "repetition_truncation_count": finish.count("repetition_truncation"),
        "stopped_protocol_violation_count": finish.count("stop"),
        "high_repetition_loop_count": sum(
            _extreme_repetition_loop(str(row.get("assistant_text") or "")) for row in provider
        ),
    }


def _load_run(run_path: str | Path) -> dict[str, Any]:
    root = Path(run_path)
    manifest_path = root / "manifest.json"
    turns_path = root / "turns" / "agent_turns.jsonl"
    predictions_path = root / "views" / "predictions.jsonl"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    phase = dict(manifest.get("phase_metadata") or {})
    source = {
        "run_root": root.resolve().as_posix(),
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "turns_sha256": hashlib.sha256(turns_path.read_bytes()).hexdigest(),
        "predictions_sha256": hashlib.sha256(predictions_path.read_bytes()).hexdigest(),
        "kernel_revision": manifest.get("kernel_revision"),
        "evaluation_role": phase.get("evaluation_role"),
        "stage_a_protocol": dict(manifest.get("d4_output") or {}).get("stage_a_protocol"),
        "phase_name": manifest.get("phase_name"),
        "run_id": manifest.get("run_id"),
        "run_status": manifest.get("run_status"),
        "cache_policy": manifest.get("cache_policy"),
        "expected_selection_sha256": dict(phase.get("expected_selection_sha256") or {}),
        "selected_selection_sha256": {
            str(key): str(value.get("sha256") or "")
            for key, value in dict(manifest.get("selected_sample_manifest") or {}).items()
            if isinstance(value, dict)
        },
        "selected_sample_counts": {
            str(key): int(value.get("count") or 0)
            for key, value in dict(manifest.get("selected_sample_manifest") or {}).items()
            if isinstance(value, dict)
        },
        "provider_audit": dict(manifest.get("provider_audit") or {}),
    }
    return {
        "turns": [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines() if line.strip()],
        "predictions": [
            json.loads(line) for line in predictions_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ],
        "source": source,
    }


def _source_contract_valid(source: dict[str, Any]) -> bool:
    expected = {str(key): str(value) for key, value in dict(source.get("expected_selection_sha256") or {}).items()}
    selected = {str(key): str(value) for key, value in dict(source.get("selected_selection_sha256") or {}).items()}
    counts = {str(key): int(value) for key, value in dict(source.get("selected_sample_counts") or {}).items()}
    audit = dict(source.get("provider_audit") or {})
    return bool(
        source.get("kernel_revision") == "d4_proof_carrying_v1"
        and source.get("evaluation_role") == INDEPENDENT_VALIDATION_ROLE
        and source.get("stage_a_protocol") == "tagged_text"
        and source.get("phase_name") == "development"
        and source.get("run_id")
        and source.get("run_status") in {"completed", "completed_with_errors"}
        and set(expected) == INDEPENDENT_VALIDATION_DATASETS
        and expected == selected
        and counts == {dataset: 100 for dataset in INDEPENDENT_VALIDATION_DATASETS}
        and source.get("cache_policy") == "global_validated_response_v3"
        and audit.get("required") is True
        and audit.get("status") == "passed"
    )


def _clopper_upper(failures: int, total: int) -> float:
    if total <= 0 or failures >= total:
        return 1.0
    return float(beta.ppf(0.95, failures + 1, total - failures))


def _extreme_repetition_loop(text: str) -> bool:
    tokens = re.findall(r"\S+", text.casefold())
    if len(tokens) < 100:
        return False
    ngrams = [tuple(tokens[index : index + 20]) for index in range(len(tokens) - 19)]
    counts = Counter(ngrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1) / len(ngrams)
    compressed = len(zlib.compress(text.encode("utf-8"))) / max(1, len(text.encode("utf-8")))
    return repeated >= 0.50 or compressed <= 0.05
