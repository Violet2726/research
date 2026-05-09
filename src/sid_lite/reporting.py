"""SID-lite 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
import math
import random
from typing import Any

from sid_lite.logic import METHOD_ORDER
from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.analysis_reports import render_frontier_report
from experiment_core.reporting.report_pipeline import SupplementalReport, render_report_bundle
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
)
from experiment_core.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
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


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("sid_lite")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    return render_report_bundle(
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, diagnostics),
        supplemental_reports=[
            SupplementalReport(
                result_key="frontier_report",
                filename="frontier_report.md",
                content=render_frontier_report(metrics.get("summary", []), title="SID-lite 前沿附录"),
            )
        ],
    )


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("summary", [])
    overall_rows = [row for row in rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            rows,
            title="SID-lite 成本-性能前沿",
            caption="总体结果上，各 SID-lite 变体的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="SID-lite 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="SID-lite 跨数据集表现",
            caption="各 SID-lite 变体在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_scatter_figure_spec(
            figure_id="sid_gate_tradeoff",
            title="SID 门控权衡",
            caption="总体早退率相对于准确率的变化。",
            primary_metric="准确率",
            data=[
                {
                    "label": str(row.get("method_name") or "unknown"),
                    "short_label": str(row.get("method_name") or "unknown"),
                    "x": float(row.get("early_exit_rate") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
                }
                for row in overall_rows
            ],
            x_label="早退率",
            y_label="准确率",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="左上区域代表在较高准确率下完成更多零通信早退。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="invalid_confidence_fail_open",
            title="置信度失效 fail-open 计数",
            caption="因置信度信号无效而触发 fail-open 的样本计数。",
            primary_metric="计数",
            data=[
                {
                    "label": "sid_lite",
                    "short_label": "sid_lite",
                    "invalid_confidence_fail_open_count": float(diagnostics.get("invalid_confidence_fail_open_count") or 0.0),
                }
            ],
            series=[("invalid_confidence_fail_open_count", "Fail-open 计数")],
            x_label="计数",
            source_kind="diagnostics",
            dataset_scope="run_level",
            note="该计数越低，说明黑盒条件下的 SID 近似信号越稳定。",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    best_row = max(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), default=None)
    best_efficiency_row = max(overall_rows, key=lambda item: float(item.get("acc_per_1k_tokens") or 0.0), default=None)
    ci_text = _bootstrap_ci_text(predictions, "sid_lite", "always_full")

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row['method_name']}`，准确率为 {format_float(best_row.get('accuracy_mean'))}。")
    if best_efficiency_row is not None:
        abstract.append(f"总体效率最高的方法是 `{best_efficiency_row['method_name']}`，每千 token 准确率为 {format_float(best_efficiency_row.get('acc_per_1k_tokens'), 6)}。")
    abstract.append(f"`sid_lite` 相对 `always_full` 的总体准确率差异 bootstrap 95% CI 为 `{ci_text}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "SID-lite 关注在黑盒条件下，能否用自报置信度和结构化语义字段近似 self-signals，从而在不读 logits 的情况下实现选择性通信。",
                "主指标为准确率；成本指标采用平均通信 token / 题与平均总 token / 题；机制指标重点是早退率、压缩比和 fail-open 计数。",
                "本实验固定比较 `mv_3`、`always_full`、`compression_only` 与 `sid_lite`，因此可以直接分离“只压缩”和“有门控”的差异。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率", "早退率", "压缩比"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("accuracy_mean")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("acc_per_1k_tokens"), 6),
                        format_float(row.get("early_exit_rate")),
                        format_float(row.get("compression_ratio_mean")) if row.get("compression_ratio_mean") is not None else "-",
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "机制诊断",
            "bullets": [
                f"SID 早退率：`{diagnostics.get('sid_early_exit_rate', 0.0)}`。",
                f"非法 confidence fail-open 计数：`{diagnostics.get('invalid_confidence_fail_open_count', 0)}`。",
                "如果 `sid_lite` 的早退率很高，但准确率与 `always_full` 接近，说明门控近似已经具备明显工程价值。",
                "如果 fail-open 计数偏高，则应优先改进置信度提取和结构化恢复逻辑，而不是继续压缩通信预算。",
            ],
        },
        {
            "title": "分数据集表现",
            "tables": [
                {
                    "title": dataset,
                    "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每千 token 准确率"],
                    "rows": [
                        [
                            f"`{row['method_name']}`",
                            format_float(row.get("accuracy_mean")),
                            format_float(row.get("communication_tokens_mean"), 2),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("acc_per_1k_tokens"), 6),
                        ]
                        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset])
                    ],
                }
                for dataset in sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"})
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "如果 `sid_lite` 能在明显降低通信成本时维持与 `always_full` 接近的准确率，则后续优先值得扩大样本确认。",
                "若 `compression_only` 已经贡献了大部分成本收益，而 `sid_lite` 仅带来有限额外收益，应更谨慎评估门控复杂度是否值得。",
                "正式进入更大规模 phase 前，应联合考察前沿图、早退率图和 fail-open 计数，避免只凭总体准确率判断。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "SID-lite 不读取真实 token logits 或 attention，因此并不是对完整 SID 的严格复现，而是黑盒近似版本。",
                "当前报告只反映当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于后续复核和引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="SID-lite 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


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


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
