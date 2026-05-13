"""Free-MAD-lite 实验的科研报告与图资产生成。"""

from __future__ import annotations

from pathlib import Path
import math
import random
from typing import Any

from research_experiments.families.free_mad_lite.algorithms import METHOD_ORDER
from research_experiments.workspace.layout import default_reports_root
from research_experiments.reporting.analysis_reports import render_frontier_report
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.families.shared.report_common import render_family_report_bundle, render_family_scientific_report
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload, load_jsonl_rows
from research_experiments.reporting.run_figures import (
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


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("free_mad_lite")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "metrics.json")
    diagnostics = load_json_payload(root / "diagnostics.json")
    predictions = load_jsonl_rows(root / "final_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    return render_family_report_bundle(
        family_name="free_mad_lite",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics),
        supplemental_reports=[
            SupplementalReport(
                result_key="frontier_report",
                filename="frontier_report.md",
                content=render_frontier_report(metrics.get("summary", []), title="Free-MAD-lite 前沿附录"),
            )
        ],
    )


def _build_figure_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    summary = SummaryTableView.from_metrics_payload(metrics)
    rows = [row.raw for row in summary.rows]
    overall_rows = summary.overall_rows()
    return [
        build_frontier_figure_spec(
            rows,
            title="Free-MAD-lite 成本-性能前沿",
            caption="总体结果上，各 Free-MAD-lite 变体的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="Free-MAD-lite 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="Free-MAD-lite 跨数据集表现",
            caption="各 Free-MAD-lite 变体在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="trajectory_score_panel",
            title="轨迹裁决效果",
            caption="总体层面 changed / corrected / harmed 三类比例对比。",
            primary_metric="比例",
            data=[
                {
                    "label": row.method_name,
                    "short_label": row.method_name,
                    "changed_answer_rate": float(row.changed_answer_rate or 0.0),
                    "corrected_rate": (float(row.corrected_count or 0.0) / float(row.question_count or 1.0)),
                    "harmed_rate": (float(row.harmed_count or 0.0) / float(row.question_count or 1.0)),
                }
                for row in overall_rows
            ],
            series=[
                ("changed_answer_rate", "Changed answer"),
                ("corrected_rate", "Corrected rate"),
                ("harmed_rate", "Harmed rate"),
            ],
            x_label="比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="纠正和伤害都被标准化为题级比例，以便和答案变化率并列比较。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="judge_fallback_summary",
            title="Judge fallback 概览",
            caption="总体层面 judge fallback rate 对比。",
            primary_metric="Fallback rate",
            data=[
                {
                    "label": row.method_name,
                    "short_label": row.method_name,
                    "judge_fallback_rate": float(row.judge_fallback_rate or 0.0),
                }
                for row in overall_rows
            ],
            series=[("judge_fallback_rate", "Judge fallback rate")],
            x_label="比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="该比例越低，说明轨迹裁决器越少需要退回到基础投票结果。",
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
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = _ordered_rows(summary.overall_rows())
    best_row = summary.best_by("accuracy_mean", rows=overall_rows)
    best_efficiency_row = summary.best_by("acc_per_1k_tokens", rows=overall_rows)
    ci_text = _bootstrap_ci_text(predictions, "free_mad_lite_llm_trajectory", "vanilla_mad_r1_final_vote")

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row.method_name}`，准确率为 {format_float(best_row.accuracy_mean)}。")
    if best_efficiency_row is not None:
        abstract.append(f"总体效率最高的方法是 `{best_efficiency_row.method_name}`，每千 token 准确率为 {format_float(best_efficiency_row.acc_per_1k_tokens, 6)}。")
    abstract.append(f"`free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的总体准确率差异 bootstrap 95% CI 为 `{ci_text}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "Free-MAD-lite 关注单轮 anti-conformity 与 LLM trajectory judge 是否足以带来稳定收益，而不复现完整 score-model 训练流程。",
                "主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；机制指标重点是 changed answer、corrected、harmed 与 judge fallback rate。",
                "本实验固定比较 `mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote` 和 `free_mad_lite_llm_trajectory`，因此可以隔离轨迹裁决环节的贡献。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率", "Judge fallback rate", "Changed answer rate"],
                "rows": [
                    [
                        f"`{row.method_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.calls_per_question_mean, 2),
                        format_float(row.acc_per_1k_tokens, 6),
                        format_float(row.judge_fallback_rate),
                        format_float(row.changed_answer_rate),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "机制诊断",
            "bullets": [
                f"Judge fallback rate：`{diagnostics.get('judge_fallback_rate', 0.0)}`；Judge fallback count：`{diagnostics.get('judge_fallback_count', 0)}`。",
                f"Anti-conformity prompt hash：`{diagnostics.get('anti_conformity_prompt_hash', manifest.get('anti_conformity_prompt_hash', 'unknown'))}`。",
                "如果 changed answer rate 很高，但 corrected rate 没有同步提高，说明轨迹裁决更像是在频繁改写答案，而不是真正识别正确轨迹。",
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
                            f"`{row.method_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                            format_float(row.acc_per_1k_tokens, 6),
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
                "若 `free_mad_lite_llm_trajectory` 在总体准确率和每千 token 准确率上都优于 `vanilla_mad_r1_final_vote`，说明轨迹裁决在当前设置下具有独立价值。",
                "若 judge fallback rate 偏高，应优先增强 judge 的稳定性，再考虑扩大 anti-conformity 的使用范围。",
                "进入更大样本 phase 前，建议同时核对轨迹裁决效果图和 fallback 图，避免只看总体准确率结论。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前实现只验证单轮 anti-conformity 与 LLM trajectory judge，不包含论文中的完整 score-based 决策训练流程。",
                "本报告反映的是当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`trajectory_scores.jsonl`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。",
            ],
        ),
    ]
    return render_family_scientific_report(
        title="Free-MAD-lite 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _ordered_rows(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row.method_name) if row.method_name in METHOD_ORDER else 999)


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


