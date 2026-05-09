"""`budget_comm` 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from budget_comm.logic import METHOD_ORDER
from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.analysis_reports import render_frontier_report, write_report
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
    write_figure_bundle,
)
from experiment_core.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出结构化运行摘要。"""
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


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """渲染中文科研报告并刷新图资产。"""
    publish_dir = publish_dir or default_reports_root("budget_comm")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "budget_diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="预算通信前沿附录"))

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )
    return {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "frontier_report": str(root / "frontier_report.md"),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("summary", [])
    overall_rows = [row for row in rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            rows,
            title="预算通信成本-性能前沿",
            caption="总体结果上，各预算通信方法的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="预算通信效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="预算通信跨数据集表现",
            caption="各预算通信方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_grouped_bar_figure_spec(
            figure_id="packet_mode_mix",
            title="消息包模式构成",
            caption="总体结果上，各方法选择 full / summary / keywords / silence 的平均比例。",
            primary_metric="平均选择比例",
            data=[
                {
                    "label": _method_label(row),
                    "short_label": _method_label(row),
                    "full_ratio_mean": float(row.get("full_ratio_mean") or 0.0),
                    "summary_ratio_mean": float(row.get("summary_ratio_mean") or 0.0),
                    "keywords_ratio_mean": float(row.get("keywords_ratio_mean") or 0.0),
                    "silence_ratio_mean": float(row.get("silence_ratio_mean") or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("full_ratio_mean", "Full"),
                ("summary_ratio_mean", "Summary"),
                ("keywords_ratio_mean", "Keywords"),
                ("silence_ratio_mean", "Silence"),
            ],
            x_label="平均比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="各比例之和应接近 1，用于展示通信预算最终被分配到哪类消息包。",
        ),
        build_scatter_figure_spec(
            figure_id="budget_utilization_tradeoff",
            title="预算利用率权衡",
            caption="总体准确率相对于平均预算利用率的变化。",
            primary_metric="准确率",
            data=[
                {
                    "label": _method_label(row),
                    "short_label": _method_label(row),
                    "x": float(row.get("budget_utilization_mean") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
                }
                for row in overall_rows
                if row.get("budget_utilization_mean") is not None
            ],
            x_label="平均预算利用率",
            y_label="准确率",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="参考线 x=1 表示预算被完全用满，可据此判断收益是否来自更激进的预算消耗。",
            reference_x=1.0,
        ),
    ]


def _method_label(row: dict[str, Any]) -> str:
    return str(row.get("display_name") or row.get("method_name") or "unknown")


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    track_name = str(manifest.get("context_view", {}).get("track_name", "unknown"))
    calibration = diagnostics.get("calibration", {})
    full_gate = diagnostics.get("full_dala_gate", {})
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    per_dataset = sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"})
    best_row = max(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), default=None)
    best_efficiency_row = max(overall_rows, key=lambda item: float(item.get("acc_per_1k_tokens") or 0.0), default=None)
    failure_cases = _select_failure_cases(predictions)
    ci_text = _bootstrap_ci_text(predictions, "dala_lite", "all_to_all_full")

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row['display_name']}`，准确率为 {format_float(best_row.get('accuracy_mean'))}。")
    if best_efficiency_row is not None:
        abstract.append(
            f"总体效率最高的方法是 `{best_efficiency_row['display_name']}`，每千 token 准确率为 {format_float(best_efficiency_row.get('acc_per_1k_tokens'), 6)}。"
        )
    abstract.append(f"`dala_lite` 相对 `all_to_all_full` 的总体准确率差异 bootstrap 95% CI 为 `{ci_text}`。")
    if full_gate:
        abstract.append(
            f"当前阶段对 Full DALA 的进入判断为 `{full_gate.get('ready_for_full_dala', False)}`，原因是 `{full_gate.get('reason', 'unknown')}`。"
        )

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                f"当前轨道为 `{track_name}`，核心问题是在受限通信预算下，DALA-lite 是否能逼近 `all_to_all_full` 的效果。",
                "主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；效率指标采用每千 token 准确率。",
                "本实验固定比较 `mv_3`、`all_to_all_full`、`budget_random`、`budget_confidence` 和 `dala_lite`，因此可以直接比较预算分配策略本身。",
            ],
        },
        {
            "title": "预算标定与进入门槛",
            "bullets": [
                f"`{dataset}`：样本数 {row['sample_count']}，`p50(all_to_all_full_comm_tokens)`={row['p50_all_to_all_full_communication_tokens']}`，`round_budget_tokens`={row['round_budget_tokens']}`。"
                for dataset, row in calibration.items()
            ] + [
                f"`{name}`：`{passed}`" for name, passed in sorted(full_gate.get("conditions", {}).items())
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row['display_name']}`",
                        format_float(row.get("accuracy_mean")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("acc_per_1k_tokens"), 6),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "机制诊断",
            "table": {
                "headers": ["方法", "平均胜者集合大小", "预算利用率", "Full 比例", "Summary 比例", "Keywords 比例", "Silence 比例", "纠正题数", "伤害题数"],
                "rows": [
                    [
                        f"`{row['display_name']}`",
                        format_float(row.get("winner_set_size_mean")),
                        "-" if row["method_name"] in {"mv_3", "all_to_all_full"} else format_float(row.get("budget_utilization_mean")),
                        format_float(row.get("full_ratio_mean")),
                        format_float(row.get("summary_ratio_mean")),
                        format_float(row.get("keywords_ratio_mean")),
                        format_float(row.get("silence_ratio_mean")),
                        str(int(row.get("corrected_count") or 0)),
                        str(int(row.get("harmed_count") or 0)),
                    ]
                    for row in overall_rows
                ],
            },
            "bullets": [
                "如果预算利用率长期偏低且准确率没有同步提升，应优先检查 winner set 大小与消息包模式是否过于保守。",
                "若纠正题数显著高于伤害题数，说明预算被更多用于识别真正需要通信的样本。",
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
                            f"`{row['display_name']}`",
                            format_float(row.get("accuracy_mean")),
                            format_float(row.get("communication_tokens_mean"), 2),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("acc_per_1k_tokens"), 6),
                        ]
                        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset])
                    ],
                }
                for dataset in per_dataset
            ],
        },
        {
            "title": "典型案例",
            "cases": [
                {
                    "数据集": case["dataset"],
                    "样本 ID": case["sample_id"],
                    "问题预览": case["question_preview"],
                    "金标": case["gold"],
                    "all_to_all_full": f"{case['full_prediction']} / {case['full_score']}",
                    "dala_lite": f"{case['dala_prediction']} / {case['dala_score']}",
                    "解释": case["reason"],
                }
                for case in failure_cases[:5]
            ],
            "bullets": ["当前阶段未收集到足够稳定的失败案例。"] if not failure_cases else [],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "若 `dala_lite` 已能在明显降低通信成本的同时保持与 `all_to_all_full` 接近的准确率，应优先进入更大样本 phase 做确认。",
                "如果预算利用率升高但准确率没有同步改善，则应优先调整密度打分或消息包层级，而不是直接放宽预算。",
                "正式推进 Full DALA 前，应同时复核成本-性能前沿图、预算利用率图和门槛条件，而不是只看单一准确率结果。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前报告只反映本 phase 的工程验证结果，不直接等同于更大样本下的正式结论。",
                "当前实现是 DALA-lite 的无训练近似，不包含 MAPPO / value network 的学习版，因此更适合做机制与预算分析，而非完整算法复现。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`budget_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核和后续引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="预算通信科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("轨道", track_name),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _select_failure_cases(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """挑选少量具有解释价值的失败或优势样例。"""
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    cases: list[dict[str, Any]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        full_row = lookup.get((dataset, sample_id, "all_to_all_full"))
        dala_row = lookup.get((dataset, sample_id, "dala_lite"))
        confidence_row = lookup.get((dataset, sample_id, "budget_confidence"))
        if full_row is None or dala_row is None:
            continue
        reason = None
        if float(dala_row["score"]) < float(full_row["score"]):
            reason = "dala_lite 在该题上弱于 all_to_all_full。"
        elif confidence_row is not None and float(dala_row["score"]) > float(confidence_row["score"]):
            reason = "dala_lite 在同预算下优于 budget_confidence。"
        if reason is None:
            continue
        cases.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "question_preview": dala_row["question_preview"],
                "gold": dala_row["gold"],
                "full_prediction": full_row["prediction"],
                "full_score": full_row["score"],
                "dala_prediction": dala_row["prediction"],
                "dala_score": dala_row["score"],
                "reason": reason,
            }
        )
        if len(cases) >= 5:
            break
    return cases


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    """生成两种方法准确率差异的 bootstrap 置信区间文本。"""
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算"
    deltas = _bootstrap_accuracy_delta(paired, iterations=2000, seed=42)
    return f"[{_quantile(deltas, 0.025):.6f}, {_quantile(deltas, 0.975):.6f}]（探索性）"


def _paired_rows(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> list[tuple[float, float]]:
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    paired: list[tuple[float, float]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary is None or reference is None:
            continue
        paired.append((float(primary["score"]), float(reference["score"])))
    return paired


def _bootstrap_accuracy_delta(
    paired_scores: list[tuple[float, float]],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    rows = list(paired_scores)
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        primary_acc = sum(primary for primary, _ in picked) / len(picked)
        reference_acc = sum(reference for _, reference in picked) / len(picked)
        samples.append(round(primary_acc - reference_acc, 6))
    return samples


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row["method_name"] in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "budget-comm")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-report.md"


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
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
