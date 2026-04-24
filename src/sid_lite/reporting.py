"""SID-lite 报告与摘要。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from sid_lite.logic import METHOD_ORDER


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出简短运行摘要。"""
    metrics = _load_json(Path(run_dir) / "metrics.json")
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset"))].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }


def render_report(run_dir: str | Path, publish_dir: str | Path = "local/reports/sid_lite") -> dict[str, Any]:
    """渲染中文 Markdown 报告。"""
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    local_report = root / "sid_lite_report.md"
    local_report.write_text(markdown, encoding="utf-8")
    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(markdown, encoding="utf-8")
    return {"run_dir": str(root), "local_report": str(local_report), "published_report": str(publish_path)}


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone = manifest.get("backbone", {})
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    ci_text = _bootstrap_ci_text(predictions, "sid_lite", "always_full")
    lines = [
        "# SID-lite Smoke20 报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 方法：`mv_3`、`always_full`、`compression_only`、`sid_lite`。",
        "- 说明：本实验是黑盒 SID-lite 近似，DashScope Chat API 不暴露 logits/attention，因此用自报置信度和结构化语义字段近似 self signals。",
        "",
        "## 2. 主结果表",
        "",
        "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Early Exit | Compression |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        compression = "-" if row.get("compression_ratio_mean") is None else f"{row['compression_ratio_mean']:.4f}"
        lines.append(
            f"| `{row['method_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} | "
            f"{row['early_exit_rate']:.4f} | {compression} |"
        )
    lines.extend(
        [
            "",
            "## 3. 机制诊断",
            "",
            f"- `sid_lite` 相对 `always_full` 的 overall accuracy delta 95% bootstrap CI：{ci_text}（smoke20 小样本，仅作方向性参考）。",
            f"- SID 早退率：`{diagnostics.get('sid_early_exit_rate', 0.0)}`",
            f"- 非法 confidence fail-open 数：`{diagnostics.get('invalid_confidence_fail_open_count', 0)}`",
            "",
            "## 4. 数据集分表",
            "",
        ]
    )
    for dataset in sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"}):
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Acc / 1K Tokens |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset]):
            lines.append(
                f"| `{row['method_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
                f"{row['total_tokens_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## 5. 局限",
            "",
            "- 当前只运行 smoke20，不能作为最终显著性结论。",
            "- SID-lite 不读取真实 token logits 或 attention maps，因此不是 full SID reproduction。",
            "",
        ]
    )
    return "\n".join(lines)


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row.get("method_name") in METHOD_ORDER else 999)


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算"
    rng = random.Random(42)
    deltas: list[float] = []
    for _ in range(2000):
        picked = [paired[rng.randrange(len(paired))] for _ in range(len(paired))]
        deltas.append(round(sum(a - b for a, b in picked) / len(picked), 6))
    return f"[{_quantile(deltas, 0.025):.6f}, {_quantile(deltas, 0.975):.6f}]"


def _paired_rows(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> list[tuple[float, float]]:
    lookup = {(row["dataset"], row["sample_id"], row["method_name"]): row for row in predictions}
    paired: list[tuple[float, float]] = []
    for dataset, sample_id in sorted({(row["dataset"], row["sample_id"]) for row in predictions}):
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary and reference:
            paired.append((float(primary["score"]), float(reference["score"])))
    return paired


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "sid-lite")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
