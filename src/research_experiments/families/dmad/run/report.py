"""DMAD family 的报告渲染。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.shared.report_common import render_family_report_bundle, render_family_scientific_report
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics(run_dir))
    grouped = summary.grouped_by_dataset()
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": {dataset: [row.raw for row in rows] for dataset, rows in grouped.items()},
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    diagnostics = load_json_payload(root / "strategy_diagnostics.json")
    paper_tables = load_json_payload(root / "paper_tables.json")
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    base_markdown = _render_report_markdown(
        run_dir=root,
        manifest=manifest,
        summary_rows=summary_rows,
        diagnostics=diagnostics,
        paper_tables=paper_tables,
    )
    payload = render_family_report_bundle(
        family_name="dmad",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(summary_rows, diagnostics),
    )
    payload["evaluation_scope"] = manifest.get("evaluation_scope")
    return payload


def _render_report_markdown(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    paper_tables: dict[str, Any],
) -> str:
    overall_rows = {str(row["method_name"]): row for row in summary_rows if str(row.get("dataset")) == "overall"}
    evaluation_scope = str(manifest.get("evaluation_scope") or "paper_main")
    primary_method_name = "dmad_cot_sbp_pot" if evaluation_scope == "paper_main" else "dmad_cot_sbp_l2m"
    primary_row = overall_rows.get(primary_method_name)
    fixed_rows = [
        row
        for row in overall_rows.values()
        if str(row.get("method_name")) in {"mad_all_cot", "mad_all_sbp", "mad_all_pot", "mad_all_l2m"}
    ]
    best_fixed_row = max(fixed_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0)) if fixed_rows else None

    abstract: list[str] = []
    if primary_row is not None and best_fixed_row is not None:
        abstract.append(
            f"`{primary_method_name}` 在当前 run 中相对最佳固定-MAD 基线 `{best_fixed_row['method_name']}` 的总体准确率差值为 `{_delta(primary_row, best_fixed_row):+.4f}`。"
        )
    abstract.append(
        f"当前实验口径为 `{evaluation_scope}`，论文对齐版本为 `{manifest.get('paper_alignment_version')}`。"
    )
    paper_main_gap_rows = list(diagnostics.get("paper_main_gap_rows") or [])

    sections = [
        {
            "title": "实验口径",
            "bullets": [
                f"实验入口：`{manifest.get('experiment_name') or manifest.get('experiment')}`",
                f"评测范围：`{evaluation_scope}`",
                f"Backbone：`{resolve_manifest_model_name(manifest)}`",
                f"运行目录：`{run_dir.as_posix()}`",
                "当前主路径已移除 selector，MAD/DMAD 统一按最终轮自洽投票聚合。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "配置策略", "实际策略", "准确率", "总 token / 题", "通信 token / 题", "调用数 / 题", "纠正率", "退化率", "vs 最佳固定-MAD"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row.get("configured_strategy_name") or ""),
                        str(row.get("effective_strategy_name") or ""),
                        _fmt(row.get("accuracy_mean")),
                        _fmt(row.get("total_tokens_mean"), 2),
                        _fmt(row.get("communication_tokens_mean"), 2),
                        _fmt(row.get("calls_per_question_mean"), 2),
                        _fmt(row.get("correction_rate")),
                        _fmt(row.get("degradation_rate")),
                        _fmt_signed(row.get("gain_over_best_fixed_mad")),
                    ]
                    for row in overall_rows.values()
                ],
            },
        },
    ]

    grouped_section = _build_grouped_section(evaluation_scope, paper_tables)
    if grouped_section is not None:
        sections.append(grouped_section)
    if paper_main_gap_rows:
        sections.append(
            {
                "title": "剩余差距样本",
                "table": {
                    "headers": ["数据集", "样本", "DMAD", "最佳固定-MAD", "persona-D"],
                    "rows": [
                        [
                            f"`{item.get('dataset')}`",
                            f"`{item.get('sample_id')}`",
                            f"{item.get('dmad_prediction')} ({_fmt(item.get('dmad_score'))})",
                            (
                                f"{item.get('best_fixed_method_name')}: {item.get('best_fixed_prediction')} "
                                f"({_fmt(item.get('best_fixed_score'))})"
                            ),
                            (
                                ""
                                if item.get("persona_d_prediction") is None
                                else f"{item.get('persona_d_prediction')} ({_fmt(item.get('persona_d_score'))})"
                            ),
                        ]
                        for item in paper_main_gap_rows
                    ],
                },
            }
        )

    sections.append(
        {
            "title": "解释边界",
            "bullets": [
                "只有 `paper_main` 会被视作论文主结果口径；`paper_appendix` 与 `extended_validation` 不覆盖主结论。",
                "若 `dmad` 未达到不低于最佳固定-MAD 基线或未追平 `mad_persona_d`，应优先回到 prompting family、消息内容、PoT 执行协议与基线预算公平性继续排查。",
                "当前 run 只保证方法口径与趋势对齐，不直接等同于论文原模型上的绝对数值复现。",
            ],
        }
    )

    return render_family_scientific_report(
        title="DMAD 论文主线高保真复现报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment_name") or manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase_name") or manifest.get("phase"))),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("评测范围", evaluation_scope),
            ("对齐版本", str(manifest.get("paper_alignment_version") or "")),
        ],
        sections=sections,
    )


def _build_grouped_section(evaluation_scope: str, paper_tables: dict[str, Any]) -> dict[str, Any] | None:
    if evaluation_scope == "paper_main":
        rows: list[list[str]] = []
        for item in paper_tables.get("math_subject_rows", []):
            rows.append(
                [
                    "`competition_math`",
                    str(item.get("group_name") or "unknown"),
                    f"`{item.get('method_name')}`",
                    _fmt(item.get("accuracy_mean")),
                    str(item.get("question_count") or 0),
                ]
            )
        for item in paper_tables.get("gpqa_domain_rows", []):
            rows.append(
                [
                    "`gpqa_diamond`",
                    str(item.get("group_name") or "unknown"),
                    f"`{item.get('method_name')}`",
                    _fmt(item.get("accuracy_mean")),
                    str(item.get("question_count") or 0),
                ]
            )
        return {
            "title": "论文分组表",
            "table": {
                "headers": ["数据集", "分组", "方法", "准确率", "题数"],
                "rows": rows,
            },
        }
    if evaluation_scope == "paper_appendix":
        return {
            "title": "附录结果",
            "table": {
                "headers": ["方法", "准确率", "题数"],
                "rows": [
                    [
                        f"`{item.get('method_name')}`",
                        _fmt(item.get("accuracy_mean")),
                        str(item.get("question_count") or 0),
                    ]
                    for item in paper_tables.get("appendix_rows", [])
                ],
            },
        }
    if evaluation_scope == "extended_validation":
        return {
            "title": "扩展验证",
            "table": {
                "headers": ["方法", "准确率", "题数"],
                "rows": [
                    [
                        f"`{item.get('method_name')}`",
                        _fmt(item.get("accuracy_mean")),
                        str(item.get("question_count") or 0),
                    ]
                    for item in paper_tables.get("extended_dataset_rows", [])
                ],
            },
        }
    return None


def _build_figure_specs(summary_rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    overall_diagnostics = [row for row in diagnostics.get("rows", []) if str(row.get("dataset")) == "overall"]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="DMAD 成本-性能前沿",
            caption="总体层面比较各方法的准确率与平均总 token。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="DMAD 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="DMAD 跨数据集表现",
            caption="DMAD 与各基线在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="dmad_strategy_profile",
            title="DMAD 策略画像",
            caption="总体层面比较初始分歧、最终一致、纠正率与退化率。",
            primary_metric="比率",
            data=[
                {
                    "label": str(row.get("method_name") or ""),
                    "short_label": str(row.get("method_name") or ""),
                    "initial_disagreement_rate": float(row.get("initial_disagreement_rate") or 0.0),
                    "final_consensus_rate": float(row.get("final_consensus_rate") or 0.0),
                    "correction_rate": float(row.get("correction_rate") or 0.0),
                    "degradation_rate": float(row.get("degradation_rate") or 0.0),
                }
                for row in overall_diagnostics
            ],
            series=[
                ("initial_disagreement_rate", "初始分歧率"),
                ("final_consensus_rate", "最终一致率"),
                ("correction_rate", "纠正率"),
                ("degradation_rate", "退化率"),
            ],
            x_label="比率",
            source_kind="strategy_diagnostics",
            dataset_scope="overall",
            note="用于判断策略异质化是否真正改变了 debate 的信息动态。",
        ),
    ]


def _delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return float(left.get("accuracy_mean") or 0.0) - float(right.get("accuracy_mean") or 0.0)


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):.{digits}f}"


def _fmt_signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    return f"{float(value):+.{digits}f}"
