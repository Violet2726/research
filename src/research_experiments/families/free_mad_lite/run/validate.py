"""Free-MAD-lite run artifact validation.

Checks shared frontend hashes, single-round constraint, judge output
structure, and trajectory judge log-level reproducibility.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import summarize_turn_statuses, validate_shared_contracts

REQUIRED_FILES = [
    "manifest.json",
    "agent_turns.jsonl",
    "debate_messages.jsonl",
    "trajectory_scores.jsonl",
    "final_predictions.jsonl",
    "metrics.json",
    "diagnostics.json",
    "progress.json",
    "report.md",
    "figure_manifest.json",
    "archive_manifest.json",
]


def validate_run(run_dir: str | Path) -> dict[str, Any]:
    """Validate Free-MAD-lite run meets smoke experiment contract."""
    root = Path(run_dir)
    missing = [name for name in REQUIRED_FILES if not (root / name).exists()]
    manifest = _load_json(root / "manifest.json") if (root / "manifest.json").exists() else {}
    turn_rows = _load_jsonl(root / "agent_turns.jsonl")
    score_rows = _load_jsonl(root / "trajectory_scores.jsonl")
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl")

    # Exclude trajectory_judge role — those have their own schema check below
    filtered_turns = [row for row in turn_rows if row.get("role") != "trajectory_judge"]
    status_summary = summarize_turn_statuses(filtered_turns)

    paired_check = _paired_design_check(manifest, prediction_rows)
    stage_hash_check = _shared_stage_hash_check(prediction_rows)
    round_check = _single_round_check(manifest)
    prompt_hash_check = {
        "passed": bool(manifest.get("anti_conformity_prompt_hash")),
        "anti_conformity_prompt_hash": manifest.get("anti_conformity_prompt_hash"),
    }
    judge_schema_check = _judge_schema_check(score_rows)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]
    passed = (
        not missing
        and status_summary["request_failures"] == 0
        and status_summary["schema_failures"] == 0
        and paired_check["passed"]
        and stage_hash_check["passed"]
        and round_check["passed"]
        and prompt_hash_check["passed"]
        and judge_schema_check["passed"]
        and figure_contract["passed"]
        and archive_contract["passed"]
    )
    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "checks": {
            "paired_design_check": paired_check,
            "shared_stage_a_hash_check": stage_hash_check,
            "single_round_check": round_check,
            "anti_conformity_prompt_hash_check": prompt_hash_check,
            "trajectory_judge_schema_check": judge_schema_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "methods": dict(Counter(row.get("method_name") for row in prediction_rows)),
    }


def _paired_design_check(manifest: dict[str, Any], prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    methods = list(
        manifest.get("methods")
        or [
            "mv_3",
            "vanilla_mad_r1_final_vote",
            "anti_conformity_final_vote",
            "free_mad_lite_llm_trajectory",
        ]
    )
    by_sample: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        by_sample[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("method_name")))
    missing = [
        {"dataset": dataset, "sample_id": sample_id, "missing_methods": sorted(set(methods) - observed)}
        for (dataset, sample_id), observed in sorted(by_sample.items())
        if set(methods) - observed
    ]
    counts = Counter(row.get("method_name") for row in prediction_rows)
    expected_count = len(by_sample)
    count_mismatches = [
        {"method_name": method, "expected": expected_count, "observed": counts.get(method, 0)}
        for method in methods
        if counts.get(method, 0) != expected_count
    ]
    return {
        "passed": not missing and not count_mismatches and expected_count > 0,
        "sample_count": expected_count,
        "missing_methods": missing[:20],
        "count_mismatches": count_mismatches,
    }


def _shared_stage_hash_check(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    mismatches = []
    grouped: dict[tuple[str, str], set[str]] = defaultdict(set)
    for row in prediction_rows:
        grouped[(str(row.get("dataset")), str(row.get("sample_id")))].add(str(row.get("stage_a_trace_hash")))
    for (dataset, sample_id), hashes in sorted(grouped.items()):
        hashes.discard("")
        hashes.discard("None")
        if len(hashes) != 1:
            mismatches.append({"dataset": dataset, "sample_id": sample_id, "hashes": sorted(hashes)})
    return {"passed": not mismatches, "mismatch_count": len(mismatches), "mismatches": mismatches[:20]}


def _single_round_check(manifest: dict[str, Any]) -> dict[str, Any]:
    protocol = manifest.get("protocol", {})
    debate_rounds = int(protocol.get("debate_rounds") or 0)
    return {"passed": debate_rounds == 1, "debate_rounds": debate_rounds}


def _judge_schema_check(score_rows: list[dict[str, Any]]) -> dict[str, Any]:
    invalid = [
        row
        for row in score_rows
        if not row.get("judge_fallback_used") and row.get("output_status") != "ok"
    ]
    return {"passed": not invalid, "invalid_count": len(invalid), "invalid_rows": _compact_rows(invalid)}


def _compact_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "dataset": row.get("dataset"),
            "sample_id": row.get("sample_id"),
            "method_name": row.get("method_name"),
            "output_status": row.get("output_status"),
        }
        for row in rows[:20]
    ]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
