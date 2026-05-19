"""Scientific report rendering for DMAD."""

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
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_rows = {str(row["method_name"]): row for row in summary_rows if str(row.get("dataset")) == "overall"}
    by_dataset_rows = [row for row in summary_rows if str(row.get("dataset")) != "overall"]

    dmad_row = overall_rows.get("dmad_strategy_diverse_r1")
    vanilla_row = overall_rows.get("vanilla_mad_r1")
    gate_passed = bool(
        dmad_row
        and vanilla_row
        and float(dmad_row.get("accuracy_mean") or 0.0) >= float(vanilla_row.get("accuracy_mean") or 0.0)
        and float(dmad_row.get("calls_per_question_mean") or 0.0) <= float(vanilla_row.get("calls_per_question_mean") or 0.0)
    )
    base_markdown = _render_report_markdown(
        run_dir=root,
        manifest=manifest,
        overall_rows=overall_rows,
        by_dataset_rows=by_dataset_rows,
        diagnostics=diagnostics,
        gate_passed=gate_passed,
    )
    payload = render_family_report_bundle(
        family_name="dmad",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(summary_rows, diagnostics),
    )
    payload["gate_passed"] = gate_passed
    return payload


def _render_report_markdown(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    overall_rows: dict[str, dict[str, Any]],
    by_dataset_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    gate_passed: bool,
) -> str:
    dmad_row = overall_rows.get("dmad_strategy_diverse_r1")
    vanilla_row = overall_rows.get("vanilla_mad_r1")
    persona_row = overall_rows.get("persona_diverse_mad_r1")
    mv_row = overall_rows.get("mv_6")
    reflection_row = overall_rows.get("single_agent_reflection_r1")

    abstract: list[str] = []
    if dmad_row and vanilla_row:
        abstract.append(
            f"`dmad_strategy_diverse_r1` 相对 `vanilla_mad_r1` 的总体准确率差值为 `{_delta(dmad_row, vanilla_row):+.4f}`。"
        )
    if dmad_row and persona_row:
        abstract.append(
            f"`dmad_strategy_diverse_r1` 相对 `persona_diverse_mad_r1` 的总体准确率差值为 `{_delta(dmad_row, persona_row):+.4f}`。"
        )
    if dmad_row and vanilla_row:
        abstract.append(
            f"`count100` 升级 gate 当前为 `{'passed' if gate_passed else 'not_passed'}`，调用数比较为 `{dmad_row.get('calls_per_question_mean')}` vs `{vanilla_row.get('calls_per_question_mean')}`。"
        )
    if not abstract:
        abstract.append("当前 run 还没有足够的总体 summary 行用于生成主结论。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验固定比较 `single_agent_cot`、`single_agent_reflection_r1`、`mv_6`、`vanilla_mad_r1`、`persona_diverse_mad_r1` 和 `dmad_strategy_diverse_r1`。",
                "单智能体与投票基线复用仓内共享 control catalog，避免在 DMAD family 内部偷偷改采样参数。",
                "多智能体方法统一采用 best-solution selector 做最终聚合，以对齐原论文公开代码的“选择最佳候选解”逻辑，而不是只做简单多数投票。",
                "主问题不是“多智能体一定更强”，而是“策略异质化是否优于表面 persona 多样化”。",
            ],
        },
        {
            "title": "总体结果表",
            "table": {
                "headers": ["方法", "准确率", "总 token / 题", "通信 token / 题", "调用数 / 题", "纠正率", "退化率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        _fmt(row.get("accuracy_mean")),
                        _fmt(row.get("total_tokens_mean"), 2),
                        _fmt(row.get("communication_tokens_mean"), 2),
                        _fmt(row.get("calls_per_question_mean"), 2),
                        _fmt(row.get("correction_rate")),
                        _fmt(row.get("degradation_rate")),
                    ]
                    for row in overall_rows.values()
                ],
            },
        },
        {
            "title": "策略诊断",
            "table": {
                "headers": ["方法", "diversity_mode", "strategy_name", "初始分歧率", "最终一致率", "改答率", "gain vs vanilla", "gain vs persona"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row.get("diversity_mode") or ""),
                        str(row.get("strategy_name") or ""),
                        _fmt(row.get("initial_disagreement_rate")),
                        _fmt(row.get("final_consensus_rate")),
                        _fmt(row.get("changed_after_debate_rate")),
                        _fmt_signed(row.get("gain_over_vanilla_mad")),
                        _fmt_signed(row.get("gain_over_persona_diverse")),
                    ]
                    for row in diagnostics.get("rows", [])
                    if str(row.get("dataset")) == "overall"
                ],
            },
        },
        {
            "title": "分数据集结果",
            "table": {
                "headers": ["数据集", "方法", "准确率", "总 token / 题", "gain vs vanilla", "gain vs persona"],
                "rows": [
                    [
                        f"`{row['dataset']}`",
                        f"`{row['method_name']}`",
                        _fmt(row.get("accuracy_mean")),
                        _fmt(row.get("total_tokens_mean"), 2),
                        _fmt_signed(row.get("gain_over_vanilla_mad")),
                        _fmt_signed(row.get("gain_over_persona_diverse")),
                    ]
                    for row in by_dataset_rows
                    if str(row.get("method_name"))
                    in {"vanilla_mad_r1", "persona_diverse_mad_r1", "dmad_strategy_diverse_r1", "mv_6", "single_agent_reflection_r1"}
                ],
            },
        },
        {
            "title": "解释边界",
            "bullets": [
                "如果 `dmad_strategy_diverse_r1` 没有稳定高于 `vanilla_mad_r1`，应先视作 standalone diagnostic，而不是直接纳入更大矩阵。",
                "如果 `persona_diverse_mad_r1` 高于 `dmad_strategy_diverse_r1`，优先检查 prompting family、selector 聚合和角色分工是否仍与论文实现存在偏差。",
                "`mv_6` 与单智能体基线主要用于解释预算和 aggregation 差异，不替代 `DMAD vs vanilla vs persona` 这一主叙事。",
            ],
        },
    ]

    if mv_row is not None or reflection_row is not None:
        sections.insert(
            3,
            {
                "title": "辅助基线观察",
                "bullets": [
                    f"`mv_6` 总体准确率：{_fmt(mv_row.get('accuracy_mean'))}" if mv_row else "`mv_6` 总体结果缺失。",
                    (
                        f"`single_agent_reflection_r1` 总体准确率：{_fmt(reflection_row.get('accuracy_mean'))}"
                        if reflection_row
                        else "`single_agent_reflection_r1` 总体结果缺失。"
                    ),
                ],
            },
        )

    return render_family_scientific_report(
        title="DMAD 多样化多智能体辩论复现报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment_name") or manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase_name") or manifest.get("phase"))),
            ("Backbone", resolve_manifest_model_name(manifest)),
            ("Prompt Version", str(manifest.get("prompt_version") or "")),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _build_figure_specs(summary_rows: list[dict[str, Any]], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    overall_diagnostics = [row for row in diagnostics.get("rows", []) if str(row.get("dataset")) == "overall"]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="DMAD 成本-性能前沿",
            caption="总体层面对比各方法的准确率与平均总 token。",
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
            caption="总体层面对比分歧、一致、纠正与退化指标。",
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
            note="用于判断策略异质化是否真的改变了 debate 的信息动态。",
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
