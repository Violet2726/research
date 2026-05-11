"""运行报告层的共享统计证据工具。"""

from __future__ import annotations

from dataclasses import dataclass
import math
import random
from typing import Any, Iterable

from experiment_core.reporting.run_figures import build_interval_figure_spec
from experiment_core.reporting.scientific_report import format_float


@dataclass(frozen=True)
class PairwiseComparisonSpec:
    """单条报告内关键比较定义。"""

    comparison_id: str
    label: str
    method_a: str
    method_b: str


def build_pairwise_comparison_rows(
    predictions: list[dict[str, Any]],
    comparisons: Iterable[PairwiseComparisonSpec],
    *,
    score_field: str = "score",
    bootstrap_samples: int = 2000,
    bootstrap_seed: int = 42,
) -> list[dict[str, Any]]:
    """基于逐样本预测构造共享统计比较表。"""

    lookup = {
        (str(row.get("dataset")), str(row.get("sample_id")), str(row.get("method_name"))): row
        for row in predictions
    }
    sample_keys = sorted({(str(row.get("dataset")), str(row.get("sample_id"))) for row in predictions})

    rows: list[dict[str, Any]] = []
    for spec in comparisons:
        paired: list[tuple[float, float]] = []
        for dataset, sample_id in sample_keys:
            row_a = lookup.get((dataset, sample_id, spec.method_a))
            row_b = lookup.get((dataset, sample_id, spec.method_b))
            if row_a is None or row_b is None:
                continue
            paired.append((_as_float(row_a.get(score_field)), _as_float(row_b.get(score_field))))

        if not paired:
            continue

        deltas = _bootstrap_mean_deltas(paired, iterations=bootstrap_samples, seed=bootstrap_seed)
        wins = sum(1 for score_a, score_b in paired if score_a > score_b)
        losses = sum(1 for score_a, score_b in paired if score_a < score_b)
        ties = len(paired) - wins - losses
        rows.append(
            {
                "comparison_id": spec.comparison_id,
                "label": spec.label,
                "short_label": spec.label[:28],
                "method_a": spec.method_a,
                "method_b": spec.method_b,
                "paired_n": len(paired),
                "mean_a": round(sum(score_a for score_a, _ in paired) / len(paired), 6),
                "mean_b": round(sum(score_b for _, score_b in paired) / len(paired), 6),
                "mean_delta": round(sum(score_a - score_b for score_a, score_b in paired) / len(paired), 6),
                "ci_low": _quantile(deltas, 0.025),
                "ci_high": _quantile(deltas, 0.975),
                "wins": wins,
                "losses": losses,
                "ties": ties,
            }
        )
    return rows


def build_pairwise_statistics_section(
    *,
    title: str,
    rows: list[dict[str, Any]],
    metric_label: str,
    note_lines: list[str] | None = None,
) -> dict[str, Any] | None:
    """把共享统计比较结果渲染成统一报告章节。"""

    if not rows:
        return None
    bullets = [
        "所有比较均基于逐样本配对结果计算，避免不同样本集带来的均值偏移。",
        f"`{metric_label}` 的 95% CI 使用 bootstrap 估计，适合作为报告层的可信度提示，而非正式显著性定论。",
    ]
    if note_lines:
        bullets.extend(note_lines)
    return {
        "title": title,
        "table": {
            "headers": ["比较", "配对样本数", "方法 A 均值", "方法 B 均值", f"{metric_label}差值", "95% CI", "wins", "losses", "ties"],
            "rows": [
                [
                    f"`{row['label']}`",
                    str(row["paired_n"]),
                    format_float(row["mean_a"], 4),
                    format_float(row["mean_b"], 4),
                    f"{float(row['mean_delta']):+.4f}",
                    f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]",
                    str(row["wins"]),
                    str(row["losses"]),
                    str(row["ties"]),
                ]
                for row in rows
            ],
        },
        "bullets": bullets,
    }


def build_pairwise_interval_figure(
    *,
    figure_id: str,
    title: str,
    caption: str,
    metric_label: str,
    rows: list[dict[str, Any]],
    source_kind: str = "paired_statistics",
    dataset_scope: str = "overall",
    note: str,
    takeaway: str | None = None,
) -> dict[str, Any] | None:
    """构建关键比较的区间图。"""

    if not rows:
        return None
    best_row = max(rows, key=lambda item: float(item["mean_delta"]))
    figure = build_interval_figure_spec(
        figure_id=figure_id,
        title=title,
        caption=caption,
        primary_metric=metric_label,
        data=[
            {
                "label": row["label"],
                "short_label": row["short_label"],
                "low": row["ci_low"],
                "high": row["ci_high"],
                "value": row["mean_delta"],
            }
            for row in rows
        ],
        x_label=f"{metric_label}差值",
        source_kind=source_kind,
        dataset_scope=dataset_scope,
        note=note,
    )
    if takeaway:
        figure["takeaway"] = takeaway
    else:
        figure["takeaway"] = (
            f"当前差值最高的比较是 `{best_row['label']}`，均值差为 {float(best_row['mean_delta']):+.4f}。"
        )
    return figure


def format_pairwise_ci_text(rows: list[dict[str, Any]], comparison_id: str) -> str:
    """提取单条比较的 CI 文本。"""

    for row in rows:
        if row.get("comparison_id") == comparison_id:
            return f"[{float(row['ci_low']):+.4f}, {float(row['ci_high']):+.4f}]"
    return "未计算。"


def _bootstrap_mean_deltas(paired_scores: list[tuple[float, float]], *, iterations: int, seed: int) -> list[float]:
    rng = random.Random(seed)
    rows = list(paired_scores)
    if not rows:
        return [0.0]
    deltas: list[float] = []
    for _ in range(max(1, iterations)):
        picked = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        deltas.append(round(sum(score_a - score_b for score_a, score_b in picked) / len(picked), 6))
    return deltas


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


def _as_float(value: Any) -> float:
    try:
        return float(value or 0.0)
    except (TypeError, ValueError):
        return 0.0
