"""`budget_comm` 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from research_experiments.families.budget_comm.algorithms import METHOD_ORDER
from research_experiments.core.foundation.workspace import default_reports_root
from research_experiments.reporting.analysis_reports import render_frontier_report
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.report_statistics import (
    PairwiseComparisonSpec,
    build_pairwise_comparison_rows,
    build_pairwise_interval_figure,
    format_pairwise_ci_text,
)
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload, load_jsonl_rows
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """输出结构化运行摘要。"""
    summary = SummaryTableView.from_metrics_payload(load_json_payload(Path(run_dir) / "metrics.json"))
    grouped = summary.grouped_by_dataset()
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": {dataset: [row.raw for row in rows] for dataset, rows in grouped.items()},
    }


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """渲染中文科研报告并刷新图资产。"""
    publish_dir = publish_dir or default_reports_root("budget_comm")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "metrics.json")
    diagnostics = load_json_payload(root / "budget_diagnostics.json")
    predictions = load_jsonl_rows(root / "final_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    return render_report_bundle(
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, predictions),
        supplemental_reports=[
            SupplementalReport(
                result_key="frontier_report",
                filename="frontier_report.md",
                content=render_frontier_report(metrics.get("summary", []), title="预算通信前沿附录"),
            )
        ],
    )


def _build_figure_specs(metrics: dict[str, Any], predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall_rows = summary.overall_rows()
    figure_specs = [
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
                    "full_ratio_mean": float(row.full_ratio_mean or 0.0),
                    "summary_ratio_mean": float(row.summary_ratio_mean or 0.0),
                    "keywords_ratio_mean": float(row.keywords_ratio_mean or 0.0),
                    "silence_ratio_mean": float(row.silence_ratio_mean or 0.0),
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
                    "x": float(row.budget_utilization_mean or 0.0),
                    "y": float(row.accuracy_mean or 0.0),
                    "value": float(row.accuracy_mean or 0.0),
                }
                for row in overall_rows
                if row.budget_utilization_mean is not None
            ],
            x_label="平均预算利用率",
            y_label="准确率",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="参考线 x=1 表示预算被完全用满，可据此判断收益是否来自更激进的预算消耗。",
            reference_x=1.0,
        ),
    ]
    evidence_rows = _budget_evidence_rows(predictions)
    comparison_figure = build_pairwise_interval_figure(
        figure_id="pairwise_accuracy_ci",
        title="关键比较配对置信区间",
        caption="同一批样本上，DALA-lite 相对于关键预算基线的准确率差值与 bootstrap 95% CI。",
        metric_label="准确率",
        rows=evidence_rows,
        note="区间整体高于 0 表示该预算策略在当前 phase 中稳定优于对应基线。",
    )
    if comparison_figure is not None:
        figure_specs.append(comparison_figure)
    return figure_specs


def _method_label(row: Any) -> str:
    if hasattr(row, "display_name"):
        return str(row.display_name or row.method_name)
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
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = _ordered_rows(summary.overall_rows())
    per_dataset = summary.dataset_names()
    best_row = summary.best_by("accuracy_mean", rows=overall_rows)
    best_efficiency_row = summary.best_by("acc_per_1k_tokens", rows=overall_rows)
    failure_cases = _select_failure_cases(predictions)
    evidence_rows = _budget_evidence_rows(predictions)
    ci_text = format_pairwise_ci_text(evidence_rows, "dala_lite_vs_all_to_all_full")

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。")
    if best_efficiency_row is not None:
        abstract.append(
            f"总体效率最高的方法是 `{best_efficiency_row.display_name}`，每千 token 准确率为 {format_float(best_efficiency_row.acc_per_1k_tokens, 6)}。"
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
                        f"`{row.display_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.calls_per_question_mean, 2),
                        format_float(row.acc_per_1k_tokens, 6),
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
                        f"`{row.display_name}`",
                        format_float(row.number("winner_set_size_mean")),
                        "-" if row.method_name in {"mv_3", "all_to_all_full"} else format_float(row.budget_utilization_mean),
                        format_float(row.full_ratio_mean),
                        format_float(row.summary_ratio_mean),
                        format_float(row.keywords_ratio_mean),
                        format_float(row.silence_ratio_mean),
                        str(row.corrected_count),
                        str(row.harmed_count),
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
                            f"`{row.display_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                            format_float(row.acc_per_1k_tokens, 6),
                        ]
                        for row in _ordered_rows(summary.dataset_rows(dataset))
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


def _budget_evidence_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return build_pairwise_comparison_rows(
        predictions,
        [
            PairwiseComparisonSpec(
                comparison_id="dala_lite_vs_all_to_all_full",
                label="dala_lite vs all_to_all_full",
                method_a="dala_lite",
                method_b="all_to_all_full",
            ),
            PairwiseComparisonSpec(
                comparison_id="dala_lite_vs_mv_3",
                label="dala_lite vs mv_3",
                method_a="dala_lite",
                method_b="mv_3",
            ),
        ],
    )




def _ordered_rows(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row.method_name) if row.method_name in METHOD_ORDER else 999)


