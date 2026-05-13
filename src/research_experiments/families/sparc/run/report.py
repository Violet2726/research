"""SPARC 报告与摘要。

报告层同时服务三类实验：内容消融、局部审计消融与 SPARC 主实验，
重点输出压缩比、触发策略收益、局部审计收益与整体成本-性能权衡。
"""

from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Any

from research_experiments.reporting.analysis_reports import (
    render_audit_diagnostic_report,
    render_frontier_report,
)
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.families.shared.report_common import render_family_report_bundle, render_family_scientific_report
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.report_views import SummaryRowView, SummaryTableView, load_json_payload, load_jsonl_rows
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
from research_experiments.workspace.layout import default_reports_root


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
    publish_dir = publish_dir or default_reports_root("sparc")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "metrics.json")
    diagnostics = load_json_payload(root / "diagnostics.json")
    predictions = load_jsonl_rows(root / "final_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    return render_family_report_bundle(
        family_name="sparc",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, diagnostics),
        supplemental_reports=[
            SupplementalReport(
                result_key="frontier_report",
                filename="frontier_report.md",
                content=render_frontier_report(metrics.get("summary", []), title="SPARC 前沿附录"),
            ),
            SupplementalReport(
                result_key="audit_diagnostic_report",
                filename="audit_diagnostics.md",
                content=render_audit_diagnostic_report(
                    metrics.get("summary", []),
                    title="SPARC 审计诊断附录",
                ),
            ),
        ],
    )


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall_rows = summary.overall_rows()
    variant_name = str(diagnostics.get("variant_name") or "")
    figure_specs = [
        build_frontier_figure_spec(
            rows,
            title="SPARC 成本-性能前沿",
            caption="SPARC 变体与对照方法在总体准确率和平均总 token 上的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="SPARC 效率排序",
            caption="按每千 token 准确率衡量的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="SPARC 跨数据集表现",
            caption="SPARC 变体与对照方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
    ]
    if variant_name == "content_ablation":
        figure_specs.append(
            build_scatter_figure_spec(
                figure_id="compression_tradeoff",
                title="压缩收益权衡",
                caption="内容消融变体在压缩率与总体准确率之间的权衡关系。",
                primary_metric="准确率",
                data=[
                    {
                        "label": str(row.display_name or row.method_name),
                        "short_label": str(row.display_name or row.method_name),
                        "x": float(row.compression_ratio_vs_full_cot or 0.0),
                        "y": float(row.accuracy_mean or 0.0),
                        "value": float(row.accuracy_mean or 0.0),
                    }
                    for row in overall_rows
                    if row.compression_ratio_vs_full_cot is not None
                ],
                x_label="相对 full CoT 的压缩率",
                y_label="准确率",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="压缩率越低表示相对 full CoT 的通信压缩越强。",
                reference_x=1.0,
            )
        )
    else:
        figure_specs.append(
            build_scatter_figure_spec(
                figure_id="audit_gain_vs_cost",
                title="审计收益与成本",
                caption="审计消融和端到端变体在审计 token 成本与总体准确率之间的关系。",
                primary_metric="准确率",
                data=[
                    {
                        "label": str(row.display_name or row.method_name),
                        "short_label": str(row.display_name or row.method_name),
                        "x": float(row.audit_tokens_mean or 0.0),
                        "y": float(row.accuracy_mean or 0.0),
                        "value": float(row.accuracy_mean or 0.0),
                    }
                    for row in overall_rows
                ],
                x_label="平均审计 token / 题",
                y_label="准确率",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="越靠左表示越少使用显式审计轮次。",
                reference_x=0.0,
            )
        )
    if diagnostics.get("trigger_selection") is not None:
        figure_specs.append(
            build_grouped_bar_figure_spec(
                figure_id="trigger_selection_profile",
                title="触发选择画像",
                caption="暴露 trigger 行为的 SPARC 变体在总体触发率和早退率上的对比。",
                primary_metric="比率",
                data=[
                    {
                        "label": str(row.display_name or row.method_name),
                        "short_label": str(row.display_name or row.method_name),
                        "trigger_rate": float(row.trigger_rate or 0.0),
                        "early_exit_rate": float(row.early_exit_rate or 0.0),
                    }
                    for row in overall_rows
                ],
                series=[("trigger_rate", "触发率"), ("early_exit_rate", "早退率")],
                x_label="比率",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="仅对 summary 中显式暴露 trigger 字段的变体展示该图。",
            )
        )
    return figure_specs


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    kind = str(manifest.get("variant_name"))
    if kind == "content_ablation":
        return _render_content_report(manifest, metrics, diagnostics, predictions, run_dir)
    if kind == "auditing_ablation":
        return _render_auditing_report(manifest, metrics, diagnostics, predictions, run_dir)
    return _render_sparc_report(manifest, metrics, diagnostics, predictions, run_dir)


def _render_content_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _rows_for_dataset(metrics, "overall")
    method_order = ["mv_3", "full_cot", "answer_only", "answer_confidence", "disagreement_step_only", "critical_evidence_only", "task_adaptive"]
    ordered_rows = _ordered_rows(overall_rows, method_order)
    disagreement_rows = _subset_summary(predictions, lambda row: bool(row.get("initial_disagreement")))
    oracle_rows = _subset_summary(predictions, lambda row: bool(row.get("oracle_positive")))
    recommendation = _summary_row_from_payload(diagnostics.get("recommended_next_default"))
    comparison_row = recommendation if recommendation else next((row for row in ordered_rows if row.method_name != "full_cot"), None)
    ci_text = _bootstrap_ci_text(predictions, comparison_row.method_name, "full_cot") if comparison_row else "未计算。"
    best_row = max(ordered_rows, key=lambda row: float(row.accuracy_mean or 0.0), default=None)
    abstract: list[str] = []
    if best_row is not None:
        abstract.append(
            f"总体准确率最高的方法是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。"
        )
    if comparison_row is not None and comparison_row.compression_ratio_vs_full_cot is not None:
        abstract.append(
            f"相对 `full_cot`，`{comparison_row.display_name or comparison_row.method_name}` 的压缩率为 {format_float(comparison_row.compression_ratio_vs_full_cot)}。"
        )
    if recommendation:
        abstract.append(f"当前推荐的下一轮默认消息模式为 `{recommendation.method_name}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验固定 trigger 与 auditing 机制，只比较消息内容本身对协作推理的影响。",
                "主指标为准确率；成本指标使用平均通信 token、平均总 token 和每千 token 准确率。",
                "重点回答的问题是：在不改变交互轮数的前提下，哪些内容压缩方式仍能保留大部分推理收益。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率", "相对 full CoT 压缩率"],
                "rows": [
                    [
                        f"`{row.display_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.calls_per_question_mean, 2),
                        format_float(row.acc_per_1k_tokens, 6),
                        format_float(row.compression_ratio_vs_full_cot),
                    ]
                    for row in ordered_rows
                ],
            },
        },
        {
            "title": "关键子集结果",
            "tables": [
                {
                    "title": "initial_disagreement = true",
                    "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题"],
                    "rows": [
                        [
                            f"`{row.display_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                        ]
                        for row in _ordered_rows(disagreement_rows, method_order)
                    ],
                },
                {
                    "title": "oracle_positive = true",
                    "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题"],
                    "rows": [
                        [
                            f"`{row.display_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                        ]
                        for row in _ordered_rows(oracle_rows, method_order)
                    ],
                },
            ],
            "bullets": [
                f"相对 `full_cot` 的总体准确率差值 95% bootstrap 区间：{ci_text}",
            ],
        },
        {
            "title": "下一轮建议",
            "bullets": [
                f"推荐默认消息模式：`{recommendation.method_name}`。" if recommendation else "当前未生成推荐消息模式。",
                (
                    f"推荐依据：总体准确率 `{format_float(recommendation.accuracy_mean)}`，平均总 token / 题 `{format_float(recommendation.total_tokens_mean, 2)}`。"
                    if recommendation
                    else "建议结合 frontier 图与压缩收益图继续筛选候选消息格式。"
                ),
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前区间估计属于探索性统计，不构成严格显著性结论。",
                "内容消融报告强调消息内容差异，不直接比较 trigger 和 auditing 机制的优劣。",
            ],
        },
        {
            **render_run_reproducibility_section(
                run_dir=run_dir,
                artifact_items=[
                    "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。",
                    "附录产物：`frontier_report.md`、`audit_diagnostics.md`。",
                ],
            )
        },
    ]
    failure_cases = _failure_case_section(
        predictions,
        primary_method=comparison_row.method_name if comparison_row else "full_cot",
        reference_method="full_cot",
    )
    if failure_cases:
        sections.insert(3, {"title": "失败案例", "cases": failure_cases})
    return render_family_scientific_report(
        title="SPARC 内容消融科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment") or "")),
            ("Phase", str(manifest.get("phase") or "")),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _render_auditing_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _ordered_rows(
        _rows_for_dataset(metrics, "overall"),
        ["majority_vote", "weighted_vote_fallback", "single_judge", "final_round_vote", "local_auditing"],
    )
    recommendation = _summary_row_from_payload(diagnostics.get("recommended_next_default"))
    ci_text = _bootstrap_ci_text(predictions, "local_auditing", "final_round_vote")
    best_row = max(overall_rows, key=lambda row: float(row.accuracy_mean or 0.0), default=None)
    abstract: list[str] = []
    if best_row is not None:
        abstract.append(
            f"总体准确率最高的聚合方式是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。"
        )
    abstract.append(f"`local_auditing` 相对 `final_round_vote` 的 95% bootstrap 区间为 {ci_text}。")
    if recommendation:
        abstract.append(f"当前推荐的下一轮默认聚合方式为 `{recommendation.method_name}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验固定消息内容，只比较最终聚合器和局部审计机制本身。",
                "除准确率外，重点跟踪解决率、弃权率、错误覆写率和 minority rescue 次数，以判断审计是否真正纠偏。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均审计 token / 题", "平均总 token / 题", "解决率", "弃权率", "错误覆写率", "Minority rescue 次数"],
                "rows": [
                    [
                        f"`{row.display_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.audit_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.resolve_rate),
                        format_float(row.abstain_rate),
                        format_float(row.wrong_overrule_rate),
                        str(row.minority_rescue_count),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "审计收益判读",
            "bullets": [
                f"`local_auditing` 相对 `final_round_vote` 的总体准确率差值 95% bootstrap 区间：{ci_text}",
                "若解决率提升而错误覆写率未同步上升，说明局部审计更可能在纠偏而非放大噪声。",
            ],
        },
        {
            "title": "下一轮建议",
            "bullets": [
                f"推荐默认聚合方式：`{recommendation.method_name}`。" if recommendation else "当前未生成推荐聚合方式。",
                (
                    f"推荐依据：总体准确率 `{format_float(recommendation.accuracy_mean)}`，平均总 token / 题 `{format_float(recommendation.total_tokens_mean, 2)}`。"
                    if recommendation
                    else "建议结合 audit_gain_vs_cost 图和错误覆写率再做最终取舍。"
                ),
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "局部审计的收益高度依赖于冲突检测质量，当前报告未单独拆出审计前置信号质量。",
                "探索性区间使用 bootstrap 估计，主要用于辅助比较而非正式显著性检验。",
            ],
        },
        {
            **render_run_reproducibility_section(
                run_dir=run_dir,
                artifact_items=[
                    "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。",
                    "附录产物：`frontier_report.md`、`audit_diagnostics.md`。",
                ],
            )
        },
    ]
    failure_cases = _failure_case_section(predictions, primary_method="local_auditing", reference_method="final_round_vote")
    if failure_cases:
        sections.insert(3, {"title": "失败案例", "cases": failure_cases})
    return render_family_scientific_report(
        title="SPARC 审计消融科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment") or "")),
            ("Phase", str(manifest.get("phase") or "")),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _render_sparc_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _ordered_rows(
        _rows_for_dataset(metrics, "overall"),
        ["mv_3", "always_communicate", "hybrid_trigger_baseline", "final_round_vote_baseline", "sparc_v1"],
    )
    trigger_selection = diagnostics.get("trigger_selection", {})
    recommendation = _summary_row_from_payload(diagnostics.get("recommended_next_default"))
    ci_text = _bootstrap_ci_text(predictions, "sparc_v1", "final_round_vote_baseline")
    best_row = max(overall_rows, key=lambda row: float(row.accuracy_mean or 0.0), default=None)
    abstract: list[str] = []
    if best_row is not None:
        abstract.append(
            f"总体准确率最高的方法是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。"
        )
    abstract.append(f"`sparc_v1` 相对 `final_round_vote_baseline` 的 95% bootstrap 区间为 {ci_text}。")
    if trigger_selection.get("selected_policy"):
        abstract.append(f"当前使用的 trigger 策略为 `{trigger_selection.get('selected_policy')}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "SPARC 主实验同时考察消息压缩、trigger 选择和局部审计三者的联合作用。",
                "主指标为准确率；成本侧同时记录通信 token、审计 token、总 token 与每题调用数。",
                "额外关注 trigger 率和早退率，用于判断策略是否把协作预算分配给真正需要的样本。",
            ],
        },
        {
            "title": "Trigger 选择记录",
            "bullets": [
                f"选中的 trigger 策略：`{trigger_selection.get('selected_policy', 'unknown')}`",
                f"选择原因：{trigger_selection.get('reason', '未记录')}",
                (
                    f"参考运行：`{trigger_selection.get('reference_run_dir')}`"
                    if trigger_selection.get("reference_run_dir")
                    else "未找到 trigger 参考运行，本轮使用默认策略。"
                ),
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均审计 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率", "触发率", "早退率"],
                "rows": [
                    [
                        f"`{row.display_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.audit_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.calls_per_question_mean, 2),
                        format_float(row.acc_per_1k_tokens, 6),
                        format_float(row.trigger_rate),
                        format_float(row.early_exit_rate),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "联合作用判读",
            "bullets": [
                f"`sparc_v1` 相对 `final_round_vote_baseline` 的总体准确率差值 95% bootstrap 区间：{ci_text}",
                "如果 trigger 率、早退率与总 token 同时下降，而准确率仍稳定，则说明 SPARC 的预算分配更有效。",
            ],
        },
        {
            "title": "下一轮建议",
            "bullets": [
                f"当前最佳 overall 方法：`{recommendation.method_name}`" if recommendation else "当前未生成下一轮建议。",
                (
                    f"trigger 选择记录：drop_questions=`{trigger_selection.get('drop_questions')}`，threshold=`{trigger_selection.get('threshold_questions')}`。"
                    if trigger_selection
                    else "建议回看 trigger_selection_profile 图，重新核对 trigger 阈值。"
                ),
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "SPARC 主实验同时耦合多种机制，单个部件的纯净因果贡献仍需结合内容消融和审计消融共同解释。",
                "探索性区间只反映当前样本下的经验不确定性，不等同于正式统计显著性判断。",
            ],
        },
        {
            **render_run_reproducibility_section(
                run_dir=run_dir,
                artifact_items=[
                    "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`final_predictions.jsonl`。",
                    "附录产物：`frontier_report.md`、`audit_diagnostics.md`。",
                ],
            )
        },
    ]
    failure_cases = _failure_case_section(predictions, primary_method="sparc_v1", reference_method="final_round_vote_baseline")
    if failure_cases:
        sections.insert(4, {"title": "失败案例", "cases": failure_cases})
    return render_family_scientific_report(
        title="SPARC 主实验科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment") or "")),
            ("Phase", str(manifest.get("phase") or "")),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _failure_case_section(
    predictions: list[dict[str, Any]],
    *,
    primary_method: str,
    reference_method: str,
) -> list[dict[str, Any]]:
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    cases: list[dict[str, Any]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary is None or reference is None:
            continue
        if float(primary["score"]) < float(reference["score"]):
            cases.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "question_preview": primary["question_preview"],
                    "gold": primary["gold"],
                    "primary_prediction": primary["prediction"],
                    "primary_score": primary["score"],
                    "reference_prediction": reference["prediction"],
                    "reference_score": reference["score"],
                    "reason": primary.get("note") or "主方法在该题上弱于参考方法。",
                }
            )
        if len(cases) >= 5:
            break
    return cases


def _subset_summary(
    predictions: list[dict[str, Any]],
    predicate,
) -> list[SummaryRowView]:
    rows = [row for row in predictions if predicate(row)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for (model_name, method_name), items in grouped.items():
        total_tokens_mean = _mean(float(item["total_tokens_per_question"]) for item in items)
        summary.append(
            {
                "dataset": "subset",
                "model_name": model_name,
                "method_name": method_name,
                "display_name": items[0]["display_name"],
                "accuracy_mean": _mean(float(item["score"]) for item in items),
                "communication_tokens_mean": _mean(float(item["communication_tokens_per_question"]) for item in items),
                "total_tokens_mean": total_tokens_mean,
            }
        )
    return list(SummaryTableView.from_rows(summary).rows)


def _summary_row_from_payload(payload: Any) -> SummaryRowView | None:
    if not isinstance(payload, dict):
        return None
    return SummaryRowView.from_row(payload)


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算。"
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


def _rows_for_dataset(metrics: dict[str, Any], dataset: str) -> list[Any]:
    return SummaryTableView.from_metrics_payload(metrics).dataset_rows(dataset)


def _ordered_rows(rows: list[Any], order: list[str]) -> list[Any]:
    return sorted(rows, key=lambda row: order.index(row.method_name) if row.method_name in order else 999)


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


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


