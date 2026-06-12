"""单智能体运行结果校验。

关注基线实验是否“干净且可比较”：
请求失败率、输出成功率，以及同一 split 上不同方法的预测行数是否对齐。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from research_experiments.family_runtime.artifact_index import named_turn_record_paths, resolve_run_artifact_index
from research_experiments.family_runtime.validation import (
    load_json,
    load_jsonl,
    missing_relative_paths,
    summarize_turn_statuses,
    validate_rate_limit_check,
    validate_shared_contracts,
)


def validate_run(
    run_dir: str | Path,
    output_success_threshold: float = 0.95,
) -> dict[str, Any]:
    """Run completeness and consistency checks on single-agent run artifacts."""
    index = resolve_run_artifact_index(run_dir, family_name="single_agent")
    root = index.run_dir
    turn_paths = named_turn_record_paths(root, family_name="single_agent")
    required_paths = [
        index.manifest_path,
        index.metrics_view_path,
        turn_paths["raw_responses.jsonl"],
        index.prediction_records_path,
        index.report_path,
        index.figure_manifest_path,
        index.archive_manifest_path,
    ]
    missing_files = missing_relative_paths(root, required_paths)
    manifest = load_json(index.manifest_path)
    raw_rows = load_jsonl(turn_paths["raw_responses.jsonl"])
    prediction_rows = load_jsonl(index.prediction_records_path)
    metrics = load_json(index.metrics_view_path)

    status_summary = summarize_turn_statuses(raw_rows)

    output_by_group: dict[str, Any] = {}
    grouped_parse: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in raw_rows:
        grouped_parse[(row["dataset"], row["method_name"])][row["output_status"]] += 1
    for (dataset, method_name), counts in sorted(grouped_parse.items()):
        total = sum(counts.values())
        output_by_group[f"{dataset}:{method_name}"] = {
            "total_calls": total,
            "protocol_failures": counts.get("protocol_fail", 0),
            "request_failures": counts.get("request_fail", 0),
            "output_success_rate": counts.get("ok", 0) / total if total else 0.0,
        }

    split_count_check = _validate_prediction_counts(prediction_rows)
    rate_limit_check = validate_rate_limit_check(
        index.progress_path,
        raw_rows,
        manifest=manifest,
    )
    shared_contracts = validate_shared_contracts(root)
    figure_contract = shared_contracts["figure_contract"]
    archive_contract = shared_contracts["archive_contract"]

    passed = all(
        [
            not missing_files,
            status_summary["request_failures"] == 0,
            status_summary["output_success_rate"] >= output_success_threshold,
            split_count_check["passed"],
            rate_limit_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing_files,
        "request_failures": status_summary["request_failures"],
        "protocol_failures": status_summary["protocol_failures"],
        "output_success_rate": status_summary["output_success_rate"],
        "checks": {
            "output_success_threshold": output_success_threshold,
            "prediction_count_check": split_count_check,
            "rate_limit_check": rate_limit_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "rate_limit_check": rate_limit_check,
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
