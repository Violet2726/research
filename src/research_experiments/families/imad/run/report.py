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
    adaptive_row = next((row for row in overall_rows if row.get("method_name") == "imad_adaptive"), None)
    fixed_r3_row = next((row for row in overall_rows if row.get("method_name") == "mad_fixed_r3"), None)
    mv6_row = next((row for row in overall_rows if row.get("method_name") == "mv_6"), None)

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
    if adaptive_row is not None and fixed_r3_row is not None:
        delta = float(adaptive_row.get("accuracy_mean") or 0.0) - float(fixed_r3_row.get("accuracy_mean") or 0.0)
        lines.append(
            f"- 相比 `mad_fixed_r3`，`imad_adaptive` 的准确率差值为 `{delta:+.4f}`，"
            f"早停率为 `{float(adaptive_row.get('stopped_early_rate') or 0.0):.4f}`。"
        )
    if adaptive_row is not None and mv6_row is not None:
        delta = float(adaptive_row.get("accuracy_mean") or 0.0) - float(mv6_row.get("accuracy_mean") or 0.0)
        lines.append(f"- 相比同预算无通信控制 `mv_6`，`imad_adaptive` 的准确率差值为 `{delta:+.4f}`。")

    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 本实验复现 Adaptive Stopping / Efficient Debate 路线，只覆盖 same-context benchmark。",
            "- 核心比较对象是 `mad_fixed_r1 / r2 / r3` 与 `imad_adaptive`，以及共享的等预算无通信控制。",
            "- 自适应停止使用 Beta-Binomial 后验与相邻轮次 K-S 统计量判断稳定性，不引入训练期机制。",
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
                "debate_gain_over_vote",
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
            "- `imad_adaptive` 的目标是节省轮次与 token，而不是默认追求超过固定三轮 debate 的绝对最高分。",
            "- 若 adaptive 方法在 `count100` 以上阶段无法同时满足“省成本 + 不显著掉点”，应继续保留为 supporting family。",
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
