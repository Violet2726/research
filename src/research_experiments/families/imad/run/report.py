"""iMAD 实验的科研报告与图资产生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.workspace.layout import default_reports_root
from research_experiments.families.shared.report_common import render_family_report_bundle
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取 iMAD 运行目录中的 `metrics.json`。"""

    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总 iMAD 运行摘要。"""

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
    """生成 iMAD 的中文科研报告。"""

    publish_dir = publish_dir or default_reports_root("imad")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    stability = load_json_payload(root / "stability_diagnostics.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]

    adaptive_row = _overall_row(overall_rows, "imad_adaptive")
    fixed_r1_row = _overall_row(overall_rows, "mad_fixed_r1")
    fixed_r2_row = _overall_row(overall_rows, "mad_fixed_r2")
    fixed_r3_row = _overall_row(overall_rows, "mad_fixed_r3")
    mv6_row = _overall_row(overall_rows, "mv_6")
    key_comparisons = _build_key_comparisons(
        adaptive_row=adaptive_row,
        fixed_r1_row=fixed_r1_row,
        fixed_r2_row=fixed_r2_row,
        fixed_r3_row=fixed_r3_row,
        mv6_row=mv6_row,
    )

    lines = [
        "# iMAD 自适应停止科研报告",
        "",
        "## 摘要",
        "",
    ]
    if adaptive_row is not None:
        lines.append(
            f"- `imad_adaptive` 的总体准确率为 `{float(adaptive_row.get('accuracy_mean') or 0.0):.4f}`，"
            f"平均执行轮数为 `{float(adaptive_row.get('executed_round_count_mean') or 0.0):.4f}`。"
        )
    if fixed_r1_row is not None and adaptive_row is not None:
        lines.append(
            f"- 相比 `mad_fixed_r1`，`imad_adaptive` 的准确率差值为 "
            f"`{_delta(adaptive_row, fixed_r1_row):+.4f}`，token 比例为 "
            f"`{_ratio(adaptive_row, fixed_r1_row, 'total_tokens_mean'):.4f}`。"
        )
    if fixed_r3_row is not None and adaptive_row is not None:
        lines.append(
            f"- 相比 `mad_fixed_r3`，`imad_adaptive` 的准确率差值为 "
            f"`{_delta(adaptive_row, fixed_r3_row):+.4f}`，token 比例为 "
            f"`{_ratio(adaptive_row, fixed_r3_row, 'total_tokens_mean'):.4f}`。"
        )
    if mv6_row is not None and fixed_r1_row is not None:
        lines.append(
            f"- `mv_6` 只作为 `mad_fixed_r1` 的同预算无通信 floor："
            f"二者准确率差值为 `{_delta(fixed_r1_row, mv6_row):+.4f}`。"
        )

    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 本实验复现 Adaptive Stopping / Efficient Debate 路线，只覆盖 same-context benchmark。",
            "- 核心比较对象是 `mad_fixed_r1 / r2 / r3` 与 `imad_adaptive`；其中 `mv_6` 只对应 `mad_fixed_r1` 的同预算无通信对照。",
            "- 自适应停止使用 Beta-Binomial 后验与相邻状态的 K-S 统计量判断稳定性，允许在第 1 轮 debate 后直接提前停止。",
            "",
            "## 总体结果表",
            "",
        ]
    )
    lines.extend(
        _render_table(
            overall_rows,
            [
                "method_name",
                "accuracy_mean",
                "total_tokens_mean",
                "calls_per_question_mean",
                "executed_round_count_mean",
                "stopped_early_rate",
                "stability_stop_rate",
                "matched_vote_control",
                "debate_gain_over_vote",
            ],
        )
    )
    lines.extend(["", "## 关键对照", ""])
    lines.extend(
        _render_table(
            key_comparisons,
            [
                "comparison",
                "accuracy_delta",
                "token_ratio",
                "calls_delta",
                "note",
            ],
        )
    )
    lines.extend(["", "## 稳定性诊断", ""])
    lines.extend(
        _render_table(
            stability.get("summary_rows", []),
            [
                "dataset",
                "method_name",
                "executed_round_count_mean",
                "stopped_early_rate",
                "stability_stop_rate",
                "max_round_reached_rate",
                "ks_statistic_last_mean",
                "posterior_mean_last_mean",
            ],
        )
    )
    lines.extend(["", "## 轮次级稳定性统计", ""])
    lines.extend(
        _render_table(
            stability.get("round_rows", []),
            [
                "dataset",
                "method_name",
                "round_index",
                "mean_support_rate",
                "mean_posterior_mean",
                "same_top_rate",
                "stability_gate_pass_rate",
                "mean_ks_statistic",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 结论与解释边界",
            "",
            "- `imad_adaptive` 的目标不是默认超过所有固定轮数方法，而是在不显著掉点的前提下减少不必要轮次。",
            "- 如果 `imad_adaptive` 仅能证明“优于 `mad_fixed_r3` 但劣于或不优于 `mad_fixed_r1`”，那么它更适合被解释为效率型 supporting 证据，而不是新的 headline 方法。",
            "- `mv_6` 不能被写成 `imad_adaptive` 的同预算对照；它只能作为 `mad_fixed_r1` 的无通信 floor，或作为更低成本 no-communication 下界。",
            "- 本报告只解释 same-context 复现结果，不外推到 split-context 或训练期辩论能力。",
        ]
    )

    payload = render_family_report_bundle(
        family_name="imad",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_figure_specs(metrics, stability),
    )
    payload["stability_summary"] = str(root / "stability_diagnostics.json")
    return payload


def _overall_row(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("method_name") == method_name), None)


def _build_key_comparisons(
    *,
    adaptive_row: dict[str, Any] | None,
    fixed_r1_row: dict[str, Any] | None,
    fixed_r2_row: dict[str, Any] | None,
    fixed_r3_row: dict[str, Any] | None,
    mv6_row: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if adaptive_row is not None and fixed_r1_row is not None:
        rows.append(
            {
                "comparison": "imad_adaptive vs mad_fixed_r1",
                "accuracy_delta": _delta(adaptive_row, fixed_r1_row),
                "token_ratio": _ratio(adaptive_row, fixed_r1_row, "total_tokens_mean"),
                "calls_delta": round(float(adaptive_row.get("calls_per_question_mean") or 0.0) - float(fixed_r1_row.get("calls_per_question_mean") or 0.0), 6),
                "note": "判断 adaptive 是否优于最便宜且有效的固定轮数基线。",
            }
        )
    if adaptive_row is not None and fixed_r2_row is not None:
        rows.append(
            {
                "comparison": "imad_adaptive vs mad_fixed_r2",
                "accuracy_delta": _delta(adaptive_row, fixed_r2_row),
                "token_ratio": _ratio(adaptive_row, fixed_r2_row, "total_tokens_mean"),
                "calls_delta": round(float(adaptive_row.get("calls_per_question_mean") or 0.0) - float(fixed_r2_row.get("calls_per_question_mean") or 0.0), 6),
                "note": "判断 adaptive 是否接近两轮固定 debate 的中间效率点。",
            }
        )
    if adaptive_row is not None and fixed_r3_row is not None:
        rows.append(
            {
                "comparison": "imad_adaptive vs mad_fixed_r3",
                "accuracy_delta": _delta(adaptive_row, fixed_r3_row),
                "token_ratio": _ratio(adaptive_row, fixed_r3_row, "total_tokens_mean"),
                "calls_delta": round(float(adaptive_row.get("calls_per_question_mean") or 0.0) - float(fixed_r3_row.get("calls_per_question_mean") or 0.0), 6),
                "note": "判断 adaptive 是否以更低成本逼近最充分的固定轮数 debate。",
            }
        )
    if fixed_r1_row is not None and mv6_row is not None:
        rows.append(
            {
                "comparison": "mad_fixed_r1 vs mv_6",
                "accuracy_delta": _delta(fixed_r1_row, mv6_row),
                "token_ratio": _ratio(fixed_r1_row, mv6_row, "total_tokens_mean"),
                "calls_delta": round(float(fixed_r1_row.get("calls_per_question_mean") or 0.0) - float(mv6_row.get("calls_per_question_mean") or 0.0), 6),
                "note": "唯一严格同预算的 debate vs no-communication 对照。",
            }
        )
    return rows


def _delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return round(float(left.get("accuracy_mean") or 0.0) - float(right.get("accuracy_mean") or 0.0), 6)


def _ratio(left: dict[str, Any], right: dict[str, Any], field: str) -> float:
    denominator = float(right.get(field) or 0.0)
    if denominator <= 0:
        return 0.0
    return round(float(left.get(field) or 0.0) / denominator, 6)


def _build_figure_specs(metrics: dict[str, Any], stability: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_mad_rows = [
        row
        for row in stability.get("summary_rows", [])
        if isinstance(row, dict) and row.get("dataset") == "overall"
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="iMAD 成本-性能前沿",
            caption="总体结果上，各 fixed-round 与 adaptive debate 方法的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="iMAD 效率排序",
            caption="按每千 token 准确率排序的总体效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="iMAD 跨数据集表现",
            caption="各 fixed-round 与 adaptive 方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="adaptive_stop_profile",
            title="自适应停止画像",
            caption="总体层面比较平均执行轮数、早停率与稳定停止率。",
            primary_metric="轮数或比率",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "executed_round_count_mean": float(row.get("executed_round_count_mean") or 0.0),
                    "stopped_early_rate": float(row.get("stopped_early_rate") or 0.0),
                    "stability_stop_rate": float(row.get("stability_stop_rate") or 0.0),
                }
                for row in overall_mad_rows
            ],
            series=[
                ("executed_round_count_mean", "平均执行轮数"),
                ("stopped_early_rate", "早停率"),
                ("stability_stop_rate", "稳定停止率"),
            ],
            x_label="轮数或比率",
            source_kind="stability_diagnostics",
            dataset_scope="overall",
            note="轮数与比率混合呈现，仅用于直观比较不同方法的停止行为，不替代正式统计检验。",
        ),
    ]


def _render_table(rows: list[dict[str, Any]], headers: list[str]) -> list[str]:
    if not rows:
        return ["No rows."]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        values = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                values.append(f"{value:.6f}")
            elif value is None:
                values.append("")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return lines
