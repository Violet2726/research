"""单智能体实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.report_pipeline import render_report_bundle
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
    build_scatter_figure_spec,
)
from experiment_core.reporting.scientific_report import (
    format_float,
    render_markdown_table,
    render_run_reproducibility_section,
    render_scientific_report,
)


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


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    """生成正式 `report.md` 与 run 级图资产。"""
    publish_dir = publish_dir or default_reports_root("single_agent")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = load_metrics(root)
    summary_rows = metrics.get("summary", [])
    base_markdown = _render_markdown(manifest, summary_rows, root)
    return render_report_bundle(
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(summary_rows),
    )


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
    lines.extend(
        render_markdown_table(
            headers=[
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
            rows=lower_bound_rows,
        )
    )
    lines.append("")
    lines.append("## Self-Consistency 对照表")
    lines.append("")
    lines.extend(
        render_markdown_table(
            headers=[
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
            rows=self_consistency_rows,
        )
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return output


def _build_figure_specs(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    sc_rows = [
        row for row in overall_rows
        if str(row.get("method_name") or "").startswith("sc_")
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="单智能体成本-性能前沿",
            caption="总体结果上，准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="单智能体效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="单智能体跨数据集表现",
            caption="各方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_scatter_figure_spec(
            figure_id="self_consistency_scaling",
            title="自洽采样规模效应",
            caption="自洽方法在调用预算增加时的准确率变化。",
            primary_metric="准确率",
            data=[
                {
                    "label": str(row["method_name"]),
                    "short_label": str(row["method_name"]),
                    "x": float(row.get("calls_per_question_mean") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
                }
                for row in sorted(sc_rows, key=lambda item: float(item.get("calls_per_question_mean") or 0.0))
            ],
            x_label="每题调用数",
            y_label="准确率",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="该图只展示自洽类方法，用于判断预算扩张是否仍有稳定收益。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="method_budget_profile",
            title="方法预算构成",
            caption="总体结果上，各方法的 prompt 和 completion token 构成。",
            primary_metric="平均 token / 题",
            data=[
                {
                    "label": str(row["method_name"]),
                    "short_label": str(row["method_name"]),
                    "prompt_tokens_mean": float(row.get("prompt_tokens_mean") or 0.0),
                    "completion_tokens_mean": float(row.get("completion_tokens_mean") or 0.0),
                }
                for row in sorted(overall_rows, key=lambda item: float(item.get("total_tokens_mean") or 0.0), reverse=True)
            ],
            series=[
                ("prompt_tokens_mean", "Prompt token"),
                ("completion_tokens_mean", "Completion token"),
            ],
            x_label="平均 token / 题",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="把输入和输出 token 分开，便于判断预算增长主要来自提示词还是生成长度。",
        ),
    ]


def _render_markdown(manifest: dict[str, Any], summary_rows: list[dict[str, Any]], run_dir: Path) -> str:
    backbone = resolve_manifest_model_name(manifest)
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    per_dataset = sorted({str(row.get("dataset")) for row in summary_rows if row.get("dataset") != "overall"})
    best_accuracy_row = max(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), default=None)
    best_efficiency_row = max(overall_rows, key=lambda item: float(item.get("acc_per_1k_tokens") or 0.0), default=None)
    most_expensive_row = max(overall_rows, key=lambda item: float(item.get("total_tokens_mean") or 0.0), default=None)
    sc_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("sc_")]

    abstract: list[str] = []
    if best_accuracy_row is not None:
        abstract.append(
            f"总体准确率最高的方法是 `{best_accuracy_row['method_name']}`，准确率为 {format_float(best_accuracy_row.get('accuracy_mean'))}。"
        )
    if best_efficiency_row is not None:
        abstract.append(
            f"总体效率最高的方法是 `{best_efficiency_row['method_name']}`，每千 token 准确率为 {format_float(best_efficiency_row.get('acc_per_1k_tokens'), 6)}。"
        )
    if sc_rows:
        best_sc = max(sc_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0))
        abstract.append(
            f"在自洽类方法中，`{best_sc['method_name']}` 表现最佳，说明当前预算下自洽采样仍具有可观察收益。"
        )

    sections = [
        {
            "title": "研究问题与判读口径",
            "bullets": [
                "本实验把单智能体方法作为无通信比较锚点，核心问题是不同调用预算是否带来稳定的性能收益。",
                "主指标为准确率；成本指标采用平均总 token / 题；效率指标采用每千 token 准确率。",
                "对于自洽类方法，还额外关注预算扩张是否带来边际收益递减，以及 rerun 波动是否足够可控。",
            ],
        },
        {
            "title": "总体结果",
            "description": "表 1 汇总总体层面的准确率、成本和效率，便于直接选取强基线与等预算对照。",
            "table": {
                "headers": ["方法", "准确率", "平均总 token / 题", "每题调用数", "每千 token 准确率", "准确率标准差"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("accuracy_mean")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("acc_per_1k_tokens"), 6),
                        format_float(row.get("accuracy_std")),
                    ]
                    for row in sorted(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), reverse=True)
                ],
            },
        },
        {
            "title": "预算与稳定性分析",
            "bullets": [
                f"最强总体方法：`{best_accuracy_row['method_name']}`，准确率 {format_float(best_accuracy_row.get('accuracy_mean'))}，平均总 token {format_float(best_accuracy_row.get('total_tokens_mean'), 2)}。"
                if best_accuracy_row is not None else "",
                f"最高成本方法：`{most_expensive_row['method_name']}`，平均总 token {format_float(most_expensive_row.get('total_tokens_mean'), 2)}，可视为当前 phase 的预算上界。"
                if most_expensive_row is not None else "",
                "若某个自洽方法的准确率提升不再明显，而 token 成本持续上升，应优先把它作为等预算对照而非默认主方法。",
            ],
        },
        {
            "title": "分数据集表现",
            "tables": [
                {
                    "title": dataset,
                    "headers": ["方法", "准确率", "平均总 token / 题", "每题调用数", "每千 token 准确率"],
                    "rows": [
                        [
                            f"`{row['method_name']}`",
                            format_float(row.get("accuracy_mean")),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("calls_per_question_mean"), 2),
                            format_float(row.get("acc_per_1k_tokens"), 6),
                        ]
                        for row in sorted(
                            [item for item in summary_rows if item.get("dataset") == dataset],
                            key=lambda item: float(item.get("accuracy_mean") or 0.0),
                            reverse=True,
                        )
                    ],
                }
                for dataset in per_dataset
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "如果目标是构造强无通信基线，应同时记录“总体准确率最佳”和“单位成本效率最佳”两条结论，避免只看单一排序。",
                "若后续通信方法需要做等预算对照，应优先选取 token 规模最接近的 `cot_1` 或 `sc_k` 配置，而不是仅按准确率最高的方法对比。",
                "正式进入更大规模 phase 前，建议优先观察自洽类方法的收益是否已经饱和，以控制整体请求成本。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "本报告只反映当前 phase 的统计汇总，不直接等同于更大样本下的最终结论。",
                "单智能体报告不包含通信行为，因此无法解释多智能体收益来源，只能作为下游比较锚点。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`run_summary.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_tables.md`。",
                "图表与表格共享同一份 run 内数据源，便于后续复核和重渲染。",
            ],
        ),
    ]
    return render_scientific_report(
        title="单智能体科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase") or manifest.get("phase_name"))),
            ("Backbone", backbone),
            ("Prompt Version", str(manifest.get("prompt_version"))),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
