"""单智能体运行结果校验。"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import json
from typing import Any

from single_agent_baselines.reporting import budget_fairness_check


def validate_run(
    run_dir: str | Path,
    output_success_threshold: float = 0.95,
    budget_threshold: float = 0.10,
) -> dict[str, Any]:
    """对单智能体运行产物执行完整性与一致性检查。"""
    root = Path(run_dir)
    raw_rows = _load_jsonl(root / "raw_responses.jsonl")
    prediction_rows = _load_jsonl(root / "predictions.jsonl")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

    request_failures = sum(1 for row in raw_rows if row.get("output_status") == "request_fail")
    output_success_count = sum(
        1 for row in raw_rows if row.get("output_status") == "ok"
    )
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

    prompt_hash_check = _validate_prompt_hash_parity(raw_rows)
    fairness_rows = budget_fairness_check(root, threshold=budget_threshold)
    fairness_ok = all(row["within_threshold"] for row in fairness_rows)
    split_count_check = _validate_prediction_counts(prediction_rows)

    passed = all(
        [
            request_failures == 0,
            output_success_rate >= output_success_threshold,
            fairness_ok,
            prompt_hash_check["passed"],
            split_count_check["passed"],
        ]
    )

    return {
        "run_dir": str(root),
        "passed": passed,
        "checks": {
            "request_failures_total": request_failures,
            "output_success_rate": output_success_rate,
            "output_success_threshold": output_success_threshold,
            "budget_threshold": budget_threshold,
            "fairness_ok": fairness_ok,
            "prompt_hash_parity": prompt_hash_check,
            "prediction_count_check": split_count_check,
        },
        "output_by_group": output_by_group,
        "budget_fairness": fairness_rows,
        "metric_rows": metrics.get("summary", []),
    }


def _validate_prompt_hash_parity(raw_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查同题同 rerun 下的 SC / MV prompt 是否一致。"""
    sc_map: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    mv_map: dict[tuple[str, str, int], set[str]] = defaultdict(set)
    for row in raw_rows:
        key = (row["dataset"], row["sample_id"], int(row["rerun_index"]))
        if row["method_name"].startswith("sc_"):
            sc_map[key].add(row["prompt_hash"])
        if row["method_name"].startswith("mv_"):
            mv_map[key].add(row["prompt_hash"])

    mismatches: list[dict[str, Any]] = []
    for key, sc_hashes in sc_map.items():
        mv_hashes = mv_map.get(key)
        if not mv_hashes:
            continue
        if sc_hashes != mv_hashes:
            dataset, sample_id, rerun_index = key
            mismatches.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "rerun_index": rerun_index,
                    "sc_hashes": sorted(sc_hashes),
                    "mv_hashes": sorted(mv_hashes),
                }
            )

    return {"passed": len(mismatches) == 0, "mismatches": mismatches[:20], "mismatch_count": len(mismatches)}


def _validate_prediction_counts(prediction_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """检查同数据集下不同方法与 rerun 的预测行数是否对齐。"""
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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
