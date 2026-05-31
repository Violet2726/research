"""CUE run artifact completeness and consistency validation."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import (
    named_diagnostic_paths,
    named_turn_record_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.validation import (
    load_json,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate a single CUE run directory meets minimum reproducibility requirements."""
    index = resolve_run_artifact_index(run_dir, family_name="cue")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="cue")
    diagnostic_paths = named_diagnostic_paths(root, family_name="cue")
    required_paths = [
        index.manifest_path,
        turn_paths["stage_a_turns.jsonl"],
        turn_paths["communication_turns.jsonl"],
        turn_paths["audit_turns.jsonl"],
        index.prediction_records_path,
        index.metrics_view_path,
        diagnostic_paths["policy_diagnostics.json"],
        diagnostic_paths["oracle_trigger_eval.json"],
        index.progress_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing = [path.relative_to(root).as_posix() for path in required_paths if not path.exists()]
    stage_a_rows = _load_jsonl(turn_paths["stage_a_turns.jsonl"])
    communication_rows = _load_jsonl(turn_paths["communication_turns.jsonl"])
    audit_rows = _load_jsonl(turn_paths["audit_turns.jsonl"])
    control_rows = _load_jsonl(turn_paths.get("control_turns.jsonl", root / "turns" / "control_turns.jsonl"))
    prediction_rows = _load_jsonl(index.prediction_records_path)
    all_turn_rows = stage_a_rows + communication_rows + audit_rows + control_rows

    status_summary = summarize_turn_statuses(all_turn_rows)
    stage_a_hash_check = _validate_stage_a_hashes(prediction_rows)
    manifest = load_json(index.manifest_path)
    rate_limit_check = validate_rate_limit_check(index.progress_path, all_turn_rows, manifest=manifest)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and status_summary["output_success_rate"] >= 0.90
        and stage_a_hash_check["passed"]
        and rate_limit_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "checks": {
            "output_success_threshold": 0.90,
            "stage_a_hash_check": stage_a_hash_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
        "policy_methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _validate_stage_a_hashes(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in prediction_rows:
        grouped.setdefault((row["dataset"], row["sample_id"]), []).append(row)
    for (dataset, sample_id), rows in grouped.items():
        stage_a_hashes = {row.get("stage_a_trace_hash") for row in rows if row.get("method_name") != "mv_6" and row.get("method_name") != "sc_6"}
        stage_a_hashes.discard(None)
        if len(stage_a_hashes) > 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "issue": "stage_a_hash_mismatch", "values": sorted(stage_a_hashes)})
    return {"passed": len(mismatches) == 0, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}
def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    from research_experiments.core.io import read_jsonl

    return read_jsonl(path)

