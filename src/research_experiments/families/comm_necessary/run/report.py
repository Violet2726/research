"""`comm_necessary` 实验的科研报告与图资产生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.comm_necessary.algorithms import METHOD_ORDER
from research_experiments.workspace.layout import default_reports_root
from research_experiments.reporting.analysis_reports import render_split_context_report
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.families.shared.report_common import render_family_report_bundle, render_family_scientific_report
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.report_views import (
    DiagnosticTableView,
    SummaryTableView,
    load_json_payload,
    load_jsonl_rows,
)
from research_experiments.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
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
    publish_dir = publish_dir or default_reports_root("comm_necessary")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "metrics.json")
    diagnostics = load_json_payload(root / "diagnostics.json")
    predictions = load_jsonl_rows(root / "final_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    payload = render_family_report_bundle(
        family_name="comm_necessary",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, diagnostics),
        supplemental_reports=[
            SupplementalReport(
                result_key="split_context_report",
                filename="split_context_report.md",
                content=render_split_context_report(
                    metrics.get("summary", []),
                    title="Split-Context 联合指标附录",
                ),
            )
        ],
    )

    summary_path = Path(publish_dir) / "HotpotQA通信必要性最近结果汇总.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        append_figure_gallery_markdown(
            base_markdown,
            load_json_payload(root / "figure_manifest.json").get("figures", []),
            run_dir=root,
            published_path=summary_path,
        ),
        encoding="utf-8",
    )
    payload["summary_report"] = str(summary_path)
    return payload


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary = SummaryTableView.from_metrics_payload(metrics)
    diagnostic_rows = DiagnosticTableView.from_rows(diagnostics.get("key_deltas", []))
    rows = [row.raw for row in summary.rows]
    overall_rows = summary.overall_rows()
    return [
        build_frontier_figure_spec(
            rows,
            title="通信必要性成本-性能前沿",
            caption="总体 Joint F1 相对于平均总 token 的位置关系。",
            score_field="joint_f1_mean",
            primary_metric="Joint F1",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="通信必要性效率排序",
            caption="基于每千 token 得分的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 得分",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="通信必要性跨数据集表现",
            caption="各 split-context 方法在不同数据集上的 Joint F1 分布。",
            score_field="joint_f1_mean",
            primary_metric="Joint F1",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="joint_metric_panel",
            title="联合指标剖面",
            caption="总体层面 Answer F1、Supporting F1 和 Joint F1 的并列对比。",
            primary_metric="F1",
            data=[
                {
                    "label": row.method_name,
                    "short_label": row.method_name,
                    "answer_f1_mean": float(row.answer_f1_mean or 0.0),
                    "supporting_f1_mean": float(row.support_f1_mean or 0.0),
                    "joint_f1_mean": float(row.joint_f1_mean or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("answer_f1_mean", "Answer F1"),
                ("supporting_f1_mean", "Supporting F1"),
                ("joint_f1_mean", "Joint F1"),
            ],
            x_label="F1",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="把答案质量、证据质量和联合指标并列展示，便于判断通信收益来自哪一部分。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="delta_vs_controls",
            title="关键对照差值",
            caption="诊断文件中记录的关键控制组差值，重点关注 Joint F1 变化。",
            primary_metric="Joint F1 差值",
            data=[
                {
                    "label": row.comparison,
                    "short_label": row.comparison[:24],
                    "joint_f1_delta": float(row.joint_f1_delta or 0.0),
                }
                for row in diagnostic_rows.rows
            ],
            series=[("joint_f1_delta", "Joint F1 差值")],
            x_label="差值",
            source_kind="diagnostics",
            dataset_scope="overall",
            note="正值表示相对于对照方法有提升，负值表示通信设计带来退化。",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    phase = str(manifest.get("phase") or "unknown_phase")
    backbone_name = resolve_manifest_model_name(manifest)
    summary = SummaryTableView.from_metrics_payload(metrics)
    delta_rows = DiagnosticTableView.from_rows(diagnostics.get("key_deltas", []))
    overall_rows = _ordered_rows(summary.overall_rows())
    best_joint_row = summary.best_by("joint_f1_mean", rows=overall_rows)
    best_token_row = summary.best_by("acc_per_1k_tokens", rows=overall_rows)

    abstract: list[str] = []
    if best_joint_row is not None:
        abstract.append(f"总体 Joint F1 最优的方法是 `{best_joint_row.method_name}`，Joint F1 为 {format_float(best_joint_row.joint_f1_mean)}。")
    if best_token_row is not None:
        abstract.append(f"单位成本效率最优的方法是 `{best_token_row.method_name}`，每千 token 得分为 {format_float(best_token_row.acc_per_1k_tokens, 6)}。")
    if delta_rows.rows:
        strongest_delta = delta_rows.best_by("joint_f1_delta")
        abstract.append(
            f"关键对照中提升最大的比较是 `{strongest_delta.comparison}`，Joint F1 差值为 {format_float(strongest_delta.joint_f1_delta)}。"
        )

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验聚焦 split-context 场景，核心问题是：通信是否真正改善了答案质量与证据质量，而不是只改善其中一部分。",
                "主指标包括 Answer EM / F1、Supporting Facts F1 和 Joint F1；成本侧同时记录平均通信 token / 题和平均总 token / 题。",
                "由于 split-context 通信存在强设计约束，本报告特别关注关键对照差值，以判断收益来自答案交换、证据交换还是完整消息包交换。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "Answer EM", "Answer F1", "Supporting F1", "Joint F1", "平均通信 token / 题", "平均总 token / 题", "每题调用数"],
                "rows": [
                    [
                        f"`{row.method_name}`",
                        format_float(row.answer_em_mean),
                        format_float(row.answer_f1_mean),
                        format_float(row.support_f1_mean),
                        format_float(row.joint_f1_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.calls_per_question_mean, 2),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "关键对照差值",
            "table": {
                "headers": ["比较", "Answer EM 差值", "Supporting F1 差值", "Joint F1 差值", "通信 token 差值"],
                "rows": [
                    [
                        f"`{row.comparison}`",
                        format_float(row.answer_em_delta),
                        format_float(row.supporting_f1_delta),
                        format_float(row.joint_f1_delta),
                        format_float(row.communication_tokens_delta, 2),
                    ]
                    for row in delta_rows.rows
                ],
            },
            "bullets": [
                f"split 视图数：`{diagnostics.get('split_view_count', 0)}`；full-context 参考视图数：`{diagnostics.get('full_context_view_count', 0)}`。",
                "若 Joint F1 改善主要来自 Supporting F1，而 Answer EM / F1 提升有限，说明通信更像是在修复证据而非修复最终答案。",
            ],
        },
        {
            "title": "分数据集表现",
            "tables": [
                {
                    "title": dataset,
                    "headers": ["方法", "Answer EM", "Supporting F1", "Joint F1", "平均通信 token / 题", "平均总 token / 题"],
                    "rows": [
                        [
                            f"`{row.method_name}`",
                            format_float(row.answer_em_mean),
                            format_float(row.support_f1_mean),
                            format_float(row.joint_f1_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                        ]
                        for row in _ordered_rows(summary.dataset_rows(dataset))
                    ],
                }
                for dataset in summary.dataset_names()
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "正式比较 split-context 通信方法时，应优先以 Joint F1 为主结论，并同步报告 Supporting F1，避免把单纯的答案修正误判为真正的信息整合收益。",
                "如果某种交换方式带来更高 Joint F1，但通信成本也明显上升，应结合成本-性能前沿图判断它是否值得成为默认方案。",
                "进入更大样本 phase 前，应优先复核关键对照差值是否稳定，而不是只依据单轮 smoke 结论推进。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前报告主要面向当前 phase 的机制验证，不直接等同于更大样本上的最终论文结论。",
                "split-context 实验高度依赖任务视图切分方式，因此结论应和具体 view 设计一起解读。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`、`hotpot_predictions/`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。",
            ],
        ),
    ]
    return render_family_scientific_report(
        title="通信必要性科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", phase),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
            ("样本数", str(len({(row.get('dataset'), row.get('sample_id')) for row in predictions}))),
        ],
        sections=sections,
    )


def _ordered_rows(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row.method_name) if row.method_name in METHOD_ORDER else 999)

