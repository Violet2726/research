"""实验报告导出与摘要统计。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取某次运行的 ``metrics.json``。"""
    path = Path(run_dir) / "metrics.json"
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集对 summary 行做一次轻量汇总。"""
    metrics = load_metrics(run_dir)
    summary_rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in summary_rows:
        grouped[row["dataset"]].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "dataset_count": len(grouped),
        "rows": len(summary_rows),
        "by_dataset": {dataset: rows for dataset, rows in grouped.items()},
    }


def export_paper_tables(run_dir: str | Path, output_path: str | Path) -> Path:
    """把运行结果导出为论文草稿可直接引用的 Markdown 表格。"""
    metrics = load_metrics(run_dir)
    rows = metrics.get("summary", [])
    lower_bound_rows = [row for row in rows if row["method"] == "cot_1"]
    equal_budget_rows = [row for row in rows if row["method"].startswith("sc_") or row["method"].startswith("mv_")]
    fairness_rows = budget_fairness_check_from_rows(rows)

    lines: list[str] = []
    lines.append("# 论文表格导出")
    lines.append("")
    lines.append("## Lower Bound 表")
    lines.append("")
    lines.extend(_markdown_table(
        lower_bound_rows,
        [
            "dataset",
            "model_name",
            "method",
            "questions_per_rerun",
            "accuracy_mean",
            "prompt_tokens_mean",
            "completion_tokens_mean",
            "total_tokens_mean",
            "calls_per_question_mean",
            "latency_ms_mean",
        ],
    ))
    lines.append("")
    lines.append("## Equal-Budget 无通信对比表")
    lines.append("")
    lines.extend(_markdown_table(
        equal_budget_rows,
        [
            "dataset",
            "model_name",
            "method",
            "questions_per_rerun",
            "accuracy_mean",
            "accuracy_std",
            "total_tokens_mean",
            "calls_per_question_mean",
            "acc_per_1k_tokens",
        ],
    ))
    lines.append("")
    lines.append("## 预算公平性检查")
    lines.append("")
    lines.extend(_markdown_table(
        fairness_rows,
        [
            "dataset",
            "model_name",
            "budget_calls",
            "sc_method",
            "mv_method",
            "sc_total_tokens_mean",
            "mv_total_tokens_mean",
            "token_gap_ratio",
            "within_threshold",
        ],
    ))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def budget_fairness_check(run_dir: str | Path, threshold: float = 0.10) -> list[dict[str, Any]]:
    """直接从运行目录读取 summary，并检查 SC/MV 的 token 差距。"""
    metrics = load_metrics(run_dir)
    return budget_fairness_check_from_rows(metrics.get("summary", []), threshold=threshold)


def budget_fairness_check_from_rows(rows: list[dict[str, Any]], threshold: float = 0.10) -> list[dict[str, Any]]:
    """对 summary 行做 SC 与 MV 的同预算公平性比较。"""
    by_key: dict[tuple[str, str, int], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        method = row["method"]
        if not (method.startswith("sc_") or method.startswith("mv_")):
            continue
        budget_calls = int(method.split("_")[1])
        family = "sc" if method.startswith("sc_") else "mv"
        key = (row["dataset"], row["model_name"], budget_calls)
        by_key[key][family] = row

    results: list[dict[str, Any]] = []
    for (dataset, model_name, budget_calls), pair in sorted(by_key.items()):
        if "sc" not in pair or "mv" not in pair:
            continue
        sc_row = pair["sc"]
        mv_row = pair["mv"]
        sc_tokens = float(sc_row["total_tokens_mean"])
        mv_tokens = float(mv_row["total_tokens_mean"])
        denom = max(sc_tokens, mv_tokens, 1e-9)
        gap_ratio = abs(sc_tokens - mv_tokens) / denom
        results.append(
            {
                "dataset": dataset,
                "model_name": model_name,
                "budget_calls": budget_calls,
                "sc_method": sc_row["method"],
                "mv_method": mv_row["method"],
                "sc_total_tokens_mean": sc_tokens,
                "mv_total_tokens_mean": mv_tokens,
                "token_gap_ratio": round(gap_ratio, 6),
                "within_threshold": gap_ratio <= threshold,
            }
        )
    return results


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """把表格行渲染成 Markdown。"""
    if not rows:
        return ["暂无数据。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines
