"""baseline_compare 的报告与导出逻辑。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research_experiments.core.io import write_json
from research_experiments.family_runtime.artifact_index import load_metrics_payload, resolve_run_artifact_index
from research_experiments.family_runtime.report_bundle import (
    render_family_report_bundle,
    render_family_scientific_report,
)
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload, load_jsonl_rows
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import format_float, render_run_reproducibility_section
from research_experiments.workspace.layout import default_reports_root


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    return load_metrics_payload(run_dir, family_name="baseline_compare")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics(run_dir))
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
    publish_dir = publish_dir or default_reports_root("baseline_compare")
    index = resolve_run_artifact_index(run_dir, family_name="baseline_compare")
    root = index.run_dir
    manifest = load_json_payload(index.manifest_path)
    metrics = load_metrics(root)
    prediction_rows = load_jsonl_rows(index.prediction_records_path)
    output_protocol_diagnostics = load_json_payload(root / "diagnostics" / "output_protocol_diagnostics.json")
    comparison_payload = _build_baseline_comparison_payload(manifest, metrics)
    comparison_path = root / "exports" / "baseline_comparison.json"
    write_json(comparison_path, comparison_payload)
    paper_summary_path = root / "exports" / "paper_summary.csv"
    _write_paper_summary(paper_summary_path, metrics.get("summary", []))

    base_markdown = _render_markdown(
        manifest,
        metrics,
        output_protocol_diagnostics,
        comparison_payload,
        prediction_rows,
        root,
    )
    payload = render_family_report_bundle(
        family_name="baseline_compare",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics),
    )
    payload["baseline_comparison"] = str(comparison_path)
    payload["paper_summary"] = str(paper_summary_path)
    return payload


def _build_figure_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = list(metrics.get("summary", []))
    figure_rows = [row for row in summary_rows if row.get("dataset") != "overall_micro"]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    mad_overall_rows = [row for row in overall_rows if row.get("method_type") == "mad"]
    dataset_count = len({str(row.get("dataset")) for row in figure_rows if row.get("dataset") != "overall"})
    method_count = len({str(row.get("method_name")) for row in overall_rows})
    return [
        build_frontier_figure_spec(
            figure_rows,
            title="Baseline 成本-性能前沿",
            caption=f"{method_count} 方法基准包在 overall macro 口径下的准确率与平均总 token 关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            figure_rows,
            title="Baseline 效率排序",
            caption=f"{method_count} 方法基准包在 overall macro 口径下的每千 token 准确率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            [row for row in figure_rows if row.get("dataset") != "overall"],
            title="Baseline 跨数据集表现",
            caption=f"{method_count} 方法基准包在 {dataset_count} 个 benchmark 上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="overall_relative_gains",
            title="总体相对增益",
            caption="overall macro 下，相对 cot_1、相对最佳 no-comm、以及 MAD 内部 debate 增益。",
            primary_metric="准确率差值",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "delta_vs_cot_1": float(row.get("accuracy_delta_vs_cot_1") or 0.0),
                    "delta_vs_best_no_comm": float(row.get("accuracy_delta_vs_best_no_comm") or 0.0),
                    "debate_gain": float(row.get("debate_gain_over_initial_vote") or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("delta_vs_cot_1", "vs cot_1"),
                ("delta_vs_best_no_comm", "vs best no-comm"),
                ("debate_gain", "debate 内部增益"),
            ],
            x_label="准确率差值",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="control 方法的 debate 内部增益为 0；MAD 方法可同时观察外部对比和内部增益。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="mad_debate_diagnostics",
            title="MAD 机制诊断",
            caption="overall macro 下，MAD 方法的纠正率、伤害率与 debate 内部准确率增益。",
            primary_metric="比率或差值",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "corrected_rate": float(row.get("corrected_rate") or 0.0),
                    "harmed_rate": float(row.get("harmed_rate") or 0.0),
                    "debate_gain": float(row.get("debate_gain_over_initial_vote") or 0.0),
                }
                for row in mad_overall_rows
            ],
            series=[
                ("corrected_rate", "纠正率"),
                ("harmed_rate", "伤害率"),
                ("debate_gain", "debate 内部增益"),
            ],
            x_label="比率或差值",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="当纠正率稳定高于伤害率，且 debate 内部增益为正时，说明多轮交流本身提供了额外价值。",
        ),
    ]


def _build_baseline_comparison_payload(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> dict[str, Any]:
    method_order = list(manifest.get("method_order") or [])
    dataset_order = list(manifest.get("dataset_order") or [])
    summary_rows = list(metrics.get("summary", []))
    overall_macro_rows = _ordered_rows(summary_rows, dataset="overall", method_order=method_order)
    overall_micro_rows = _ordered_rows(summary_rows, dataset="overall_micro", method_order=method_order)
    per_dataset_rows = {
        dataset: _ordered_rows(summary_rows, dataset=dataset, method_order=method_order) for dataset in dataset_order
    }
    leaders = {
        "overall_macro_best_accuracy": _leader_row(overall_macro_rows, "accuracy_mean"),
        "overall_macro_best_efficiency": _leader_row(overall_macro_rows, "accuracy_per_1k_tokens"),
        "overall_macro_best_no_comm": _leader_row(
            [row for row in overall_macro_rows if row.get("method_type") == "control"],
            "accuracy_mean",
        ),
        "overall_macro_best_mad": _leader_row(
            [row for row in overall_macro_rows if row.get("method_type") == "mad"],
            "accuracy_mean",
        ),
    }
    return {
        "run_dir": str(manifest.get("run_dir") or ""),
        "experiment": manifest.get("experiment"),
        "phase": manifest.get("phase"),
        "control_prompt_version": manifest.get("control_prompt_version"),
        "mad_prompt_version": manifest.get("mad_prompt_version"),
        "backbone": manifest.get("backbone"),
        "method_order": method_order,
        "control_method_names": list(manifest.get("control_method_names") or []),
        "dataset_order": dataset_order,
        "overall_macro_rows": overall_macro_rows,
        "overall_micro_rows": overall_micro_rows,
        "per_dataset_rows": per_dataset_rows,
        "leaders": leaders,
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    output_protocol_diagnostics: dict[str, Any],
    comparison_payload: dict[str, Any],
    prediction_rows: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    del prediction_rows
    backbone_name = resolve_manifest_model_name(manifest)
    overall_macro_rows = comparison_payload["overall_macro_rows"]
    overall_micro_rows = comparison_payload["overall_micro_rows"]
    leaders = comparison_payload["leaders"]
    summary = SummaryTableView.from_metrics_payload(metrics)
    dataset_count = len(comparison_payload["dataset_order"])
    method_count = len(comparison_payload["method_order"])

    best_accuracy = leaders["overall_macro_best_accuracy"]
    best_efficiency = leaders["overall_macro_best_efficiency"]
    best_no_comm = leaders["overall_macro_best_no_comm"]
    best_mad = leaders["overall_macro_best_mad"]

    abstract: list[str] = []
    if best_accuracy is not None:
        abstract.append(
            f"overall macro 下准确率最高的方法是 `{best_accuracy['method_name']}`，准确率为 {format_float(best_accuracy.get('accuracy_mean'))}。"
        )
    if best_efficiency is not None:
        abstract.append(
            f"overall macro 下效率最高的方法是 `{best_efficiency['method_name']}`，每千 token 准确率为 {format_float(best_efficiency.get('accuracy_per_1k_tokens'), 6)}。"
        )
    if best_no_comm is not None and best_mad is not None:
        abstract.append(
            f"最佳 no-comm 方法为 `{best_no_comm['method_name']}`；最佳 MAD 方法为 `{best_mad['method_name']}`，二者形成后续创新方法最直接的对齐锚点。"
        )

    sections = [
        {
            "title": "研究问题与基准口径",
            "bullets": [
                f"本 family 固定 {dataset_count} 个 benchmark、{method_count} 个方法与三档 phase，专门为后续创新方法提供稳定对照。",
                f"主报告使用 `overall` macro-average：{dataset_count} 个 benchmark 等权；同时保留 `overall_micro` 作为按题数加权的补充口径。",
                "相对指标统一输出为：`vs cot_1`、`vs best no-comm`、`debate 内部增益`、token 比率与调用比率。",
            ],
        },
        {
            "title": "总体主表（Macro）",
            "table": {
                "headers": [
                    "方法",
                    "类型",
                    "准确率",
                    "vs cot_1",
                    "vs best no-comm",
                    "初始 vote 准确率",
                    "debate 内部增益",
                    "总 token",
                    "调用数",
                    "每千 token 准确率",
                ],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row["method_type"]),
                        format_float(row.get("accuracy_mean")),
                        _signed_float(row.get("accuracy_delta_vs_cot_1")),
                        _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                        format_float(row.get("initial_vote_accuracy_mean")),
                        _signed_float(row.get("debate_gain_over_initial_vote")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("accuracy_per_1k_tokens"), 6),
                    ]
                    for row in overall_macro_rows
                ],
            },
        },
        {
            "title": "总体加权表（Micro）",
            "table": {
                "headers": [
                    "方法",
                    "准确率",
                    "vs cot_1",
                    "vs best no-comm",
                    "总 token",
                    "调用数",
                ],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("accuracy_mean")),
                        _signed_float(row.get("accuracy_delta_vs_cot_1")),
                        _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                    ]
                    for row in overall_micro_rows
                ],
            },
        },
        {
            "title": "分数据集对比",
            "tables": [
                {
                    "title": dataset,
                    "headers": [
                        "方法",
                        "准确率",
                        "vs cot_1",
                        "vs best no-comm",
                        "总 token",
                        "每千 token 准确率",
                    ],
                    "rows": [
                        [
                            f"`{row['method_name']}`",
                            format_float(row.get("accuracy_mean")),
                            _signed_float(row.get("accuracy_delta_vs_cot_1")),
                            _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("accuracy_per_1k_tokens"), 6),
                        ]
                        for row in comparison_payload["per_dataset_rows"][dataset]
                    ],
                }
                for dataset in comparison_payload["dataset_order"]
            ],
        },
        {
            "title": "MAD 机制诊断",
            "table": {
                "headers": [
                    "方法",
                    "纠正率",
                    "伤害率",
                    "翻票率",
                    "初始一致率",
                    "最终一致率",
                    "通信 token",
                ],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("corrected_rate")),
                        format_float(row.get("harmed_rate")),
                        format_float(row.get("flip_rate")),
                        format_float(row.get("initial_consensus_rate")),
                        format_float(row.get("final_consensus_rate")),
                        format_float(row.get("communication_tokens_mean"), 2),
                    ]
                    for row in overall_macro_rows
                    if row.get("method_type") == "mad"
                ],
            },
        },
        {
            "title": "输出协议诊断",
            "table": {
                "headers": ["方法", "协议失败率", "原因缺失率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("protocol_failure_rate")),
                        format_float(row.get("reason_missing_rate")),
                    ]
                    for row in output_protocol_diagnostics.get("rows", [])
                    if row.get("dataset") == "overall"
                ],
            },
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`cost_breakdown.json`、`debate_diagnostics.json`、`report.md`、`figure_manifest.json`、`exports/baseline_comparison.json`、`exports/paper_summary.csv`。",
                f"summary 行数：`{len(summary.rows)}`。",
            ],
        ),
    ]
    return render_family_scientific_report(
        title="Baseline Compare 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Control Prompt Version", str(manifest.get("control_prompt_version"))),
            ("MAD Prompt Version", str(manifest.get("mad_prompt_version"))),
            ("Control Output Protocol", str(manifest.get("control_output_protocol"))),
            ("MAD Initial Output Protocol", str(manifest.get("mad_initial_output_protocol"))),
            ("MAD Debate Output Protocol", str(manifest.get("mad_debate_output_protocol"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _write_paper_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "aggregate_kind",
        "model_name",
        "method_name",
        "method_type",
        "question_count",
        "accuracy_mean",
        "accuracy_delta_vs_cot_1",
        "accuracy_delta_vs_best_no_comm",
        "initial_vote_accuracy_mean",
        "debate_gain_over_initial_vote",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "accuracy_per_1k_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _ordered_rows(
    summary_rows: list[dict[str, Any]],
    *,
    dataset: str,
    method_order: list[str],
) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(method_order)}
    rows = [row for row in summary_rows if row.get("dataset") == dataset]
    return sorted(rows, key=lambda row: order.get(str(row.get("method_name")), 999))


def _leader_row(rows: list[dict[str, Any]], field: str) -> dict[str, Any] | None:
    if not rows:
        return None
    return max(rows, key=lambda row: float(row.get(field) or 0.0))


def _signed_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):+.4f}"
