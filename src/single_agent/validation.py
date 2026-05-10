"""单智能体运行结果校验。

这里关注的不是机制复杂性，而是基线实验是否“干净可比”：
请求失败率、输出成功率，以及不同方法在同一 split 上的预测行数是否对齐。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.run_archives import validate_archive_contract
from experiment_core.reporting.run_figures import validate_figure_contract


def validate_run(
    run_dir: str | Path,
    output_success_threshold: float = 0.95,
) -> dict[str, Any]:
    """对单智能体运行产物执行完整性与一致性检查。"""
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
    raw_rows = _load_jsonl(root / "raw_responses.jsonl")
    prediction_rows = _load_jsonl(root / "predictions.jsonl")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    request_failures = sum(1 for row in raw_rows if row.get("output_status") == "request_fail")
    output_success_count = sum(1 for row in raw_rows if row.get("output_status") == "ok")
    output_success_rate = output_success_count / len(raw_rows) if raw_rows else 0.0

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
    figure_contract = validate_figure_contract(root)
    archive_contract = validate_archive_contract(root)

    passed = all(
        [
            not missing_files,
            request_failures == 0,
            output_success_rate >= output_success_threshold,
            split_count_check["passed"],
            figure_contract["passed"],
            archive_contract["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "missing_files": missing_files,
        "checks": {
            "request_failures_total": request_failures,
            "output_success_rate": output_success_rate,
            "output_success_threshold": output_success_threshold,
            "prediction_count_check": split_count_check,
            "figure_contract": figure_contract,
            "archive_contract": archive_contract,
        },
        "output_by_group": output_by_group,
        "metric_rows": metrics.get("summary", []),
    }


def _validate_prediction_counts(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查同数据集下不同方法同 rerun 的预测行数是否对齐。"""
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


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
