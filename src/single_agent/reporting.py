"""单智能体实验报告辅助工具。

本模块负责把基础指标整理成更适合论文与实验记录阅读的摘要形式，
重点保留 lower bound 与无通信多采样对照的核心结果。
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取单次运行目录下的 `metrics.json`。"""
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集聚合单智能体运行摘要。"""
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
    """导出论文表格风格的 Markdown 摘要。"""
    metrics = load_metrics(run_dir)
    rows = metrics.get("summary", [])
    lower_bound_rows = [row for row in rows if row["method_name"] == "cot_1"]
    self_consistency_rows = [row for row in rows if row["method_name"].startswith("sc_")]

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
            "method_name",
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
    lines.append("## Self-Consistency 无通信对照表")
    lines.append("")
    lines.extend(_markdown_table(
        self_consistency_rows,
        [
            "dataset",
            "model_name",
            "method_name",
            "questions_per_rerun",
            "accuracy_mean",
            "accuracy_std",
            "total_tokens_mean",
            "calls_per_question_mean",
            "acc_per_1k_tokens",
        ],
    ))

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _markdown_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    """把字典行渲染为简洁 Markdown 表格。"""
    if not rows:
        return ["暂无数据。"]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(header, "")) for header in headers) + " |")
    return lines
