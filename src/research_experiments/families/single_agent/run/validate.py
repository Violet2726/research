"""Single-agent run result validation.

Focuses on whether the baseline experiment is "clean and comparable":
request failure rate, output success rate, and prediction row counts
aligned across different methods on the same split.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.shared.validate_common import summarize_turn_statuses, validate_shared_contracts


def validate_run(
    run_dir: str | Path,
    output_success_threshold: float = 0.95,
) -> dict[str, Any]:
    """Run completeness and consistency checks on single-agent run artifacts."""
    root = Path(run_dir)
    required = [
        "manifest.json",
        "metrics.json",
        "raw_responses.jsonl",
        "predictions.jsonl",
        "report.md",
        "figure_manifest.json",
        "archive_manifest.json",
    ]
    missing_files = [name for name in required if not (root / name).exists()]
    raw_rows = load_jsonl(root / "raw_responses.jsonl")
    prediction_rows = load_jsonl(root / "predictions.jsonl")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    status_summary = summarize_turn_statuses(raw_rows)

    output_by_group: dict[str, Any] = {}
    grouped_parse: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in raw_rows:
        grouped_parse[(row["dataset"], row["method_name"])][row["output_status"]] += 1
    for (dataset, method_name), counts in sorted(grouped_parse.items()):
        total = sum(counts.values())
        output_by_group[f"{dataset}:{method_name}"] = {
            "total_calls": total,
            "schema_failures": counts.get("schema_fail", 0),
            "request_failures": counts.get("request_fail", 0),
            "output_success_rate": counts.get("ok", 0) / total if total else 0.0,
        }

    split_count_check = _validate_prediction_counts(prediction_rows)
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing_files,
            status_summary["request_failures"] == 0,
            status_summary["output_success_rate"] >= output_success_threshold,
            split_count_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing_files,
        "request_failures": status_summary["request_failures"],
        "schema_failures": status_summary["schema_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "checks": {
            "output_success_threshold": output_success_threshold,
            "prediction_count_check": split_count_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "output_by_group": output_by_group,
        "metric_rows": metrics.get("summary", []),
    }


def _validate_prediction_counts(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Check that different methods under the same dataset have aligned prediction row counts."""
    grouped: Counter = Counter((row["dataset"], row["method_name"], row["rerun_index"]) for row in prediction_rows)
    if not grouped:
        return {"passed": False, "details": "No prediction rows found."}
    grouped_by_dataset: dict[str, list[int]] = defaultdict(list)
    for (dataset, _, _), count in grouped.items():
        grouped_by_dataset[dataset].append(count)
    per_dataset_ok = {
        dataset: min(counts) == max(counts)
        for dataset, counts in grouped_by_dataset.items()
    }
    return {
        "passed": all(per_dataset_ok.values()),
        "per_dataset": per_dataset_ok,
        "details": {
            f"{dataset}:{method_name}:rerun{rerun_index}": count
            for (dataset, method_name, rerun_index), count in sorted(grouped.items())
        },
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
