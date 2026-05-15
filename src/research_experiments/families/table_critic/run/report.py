"""Table-Critic 实验的科研报告与图资产生成。"""

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
    """读取 Table-Critic 运行目录中的 `metrics.json`。"""

    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总 Table-Critic 运行摘要。"""

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
    """生成 Table-Critic 的中文科研报告。"""

    publish_dir = publish_dir or default_reports_root("table_critic")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    error_analysis = load_json_payload(root / "error_analysis.json")
    template_tree = load_json_payload(root / "template_tree.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]
    non_overall_rows = [row.raw for row in summary.non_overall_rows()]

    paper_row = _overall_row(overall_rows, "table_critic_paper")
    chain_row = _overall_row(overall_rows, "chain_of_table")
    critic_row = _overall_row(overall_rows, "critic_cot")

    lines = [
        "# Table-Critic 原论文主线复现报告",
        "",
        "## 摘要",
        "",
    ]
    if paper_row is not None and chain_row is not None:
        lines.append(
            f"- `table_critic_paper` 相比 `chain_of_table` 的总体准确率差值为 `{_delta(paper_row, chain_row):+.4f}`，"
            f"token 比例为 `{_ratio(paper_row, chain_row, 'total_tokens_mean'):.4f}`。"
        )
    if paper_row is not None and critic_row is not None:
        lines.append(
            f"- 相比 `critic_cot`，`table_critic_paper` 的总体准确率差值为 `{_delta(paper_row, critic_row):+.4f}`，"
            f"degradation rate 差值为 `{float(paper_row.get('degradation_rate') or 0.0) - float(critic_row.get('degradation_rate') or 0.0):+.4f}`。"
        )
    if paper_row is not None:
        lines.append(
            f"- `table_critic_paper` 的平均 refinement 轮数为 `{float(paper_row.get('refinement_round_count_mean') or 0.0):.4f}`，"
            f"template reuse rate 为 `{float(paper_row.get('template_reuse_rate') or 0.0):.4f}`。"
        )

    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 当前 `table_critic_main` 是结构化表推理的平行复现主线，不并入 `faithful_matrix`。",
            "- canonical benchmark 固定为 `WikiTQ / TabFact`。",
            "- canonical 方法固定为 `end_to_end_qa / few_shot_qa / chain_of_table / critic_cot / table_critic_paper`。",
            "- `table_critic_paper` 复现 `Chain-of-Table 初答 -> Judge -> Critic -> Refiner -> Curator`，并维护 self-evolving template tree。",
            "- 当前不纳入 `Binder / Dater`，避免把程序执行栈引入成新的主变量。",
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
                "refinement_round_count_mean",
                "correction_rate",
                "degradation_rate",
                "template_reuse_rate",
                "gain_over_chain_of_table",
            ],
        )
    )
    lines.extend(["", "## 分数据集结果", ""])
    lines.extend(
        _render_table(
            non_overall_rows,
            [
                "dataset",
                "method_name",
                "accuracy_mean",
                "refinement_round_count_mean",
                "correction_rate",
                "degradation_rate",
                "template_reuse_rate",
                "gain_over_chain_of_table",
            ],
        )
    )
    lines.extend(["", "## 错误分析", ""])
    lines.extend(
        _render_table(
            error_analysis.get("summary_rows", []),
            [
                "dataset",
                "method_name",
                "judge_error_detected_rate",
                "correction_rate",
                "degradation_rate",
                "template_reuse_rate",
                "top_error_step",
            ],
        )
    )
    lines.extend(["", "## 模板树叶节点摘要", ""])
    lines.extend(
        _render_table(
            _template_leaf_rows(template_tree),
            [
                "path",
                "template_id",
                "template_title",
                "pattern_summary",
                "usage_count",
                "success_count",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 结论与解释边界",
            "",
            "- 若 `table_critic_paper` 在 `WikiTQ` 上稳定优于 `chain_of_table` 与 `critic_cot`，且在 `TabFact` 上不系统性落后于 `critic_cot`，则视为趋势通过。",
            "- 若 `count300` 只体现出微弱增益且 token 成本失衡，就应冻结为 supporting reproduction，而不是继续扩任务面。",
            "- 当前 `dog_graph_main` 继续保留为 supporting reproduction，本机口径下不再作为下一条主攻实验。",
        ]
    )

    return render_family_report_bundle(
        family_name="table_critic",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_figure_specs(metrics, error_analysis),
    )


def _overall_row(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("method_name") == method_name), None)


def _delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return round(float(left.get("accuracy_mean") or 0.0) - float(right.get("accuracy_mean") or 0.0), 6)


def _ratio(left: dict[str, Any], right: dict[str, Any], field: str) -> float:
    denominator = float(right.get(field) or 0.0)
    if denominator <= 0:
        return 0.0
    return round(float(left.get(field) or 0.0) / denominator, 6)


def _build_figure_specs(metrics: dict[str, Any], error_analysis: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            overall_rows,
            title="Table-Critic 成本-性能前沿",
            caption="总体结果上，各表推理方法的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            overall_rows,
            title="Table-Critic 效率排序",
            caption="按每千 token 准确率排序的总体效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="Table-Critic 跨数据集表现",
            caption="各方法在不同表推理数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="table_critic_error_profile",
            title="Table-Critic 修正画像",
            caption="总体层面比较 correction、degradation 与 template reuse 三个关键比率。",
            primary_metric="比率",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "correction_rate": float(row.get("correction_rate") or 0.0),
                    "degradation_rate": float(row.get("degradation_rate") or 0.0),
                    "template_reuse_rate": float(row.get("template_reuse_rate") or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("correction_rate", "修正率"),
                ("degradation_rate", "退化率"),
                ("template_reuse_rate", "模板复用率"),
            ],
            x_label="比率",
            source_kind="error_analysis",
            dataset_scope="overall",
            note="用于观察批判-修正链是否真的带来净改善，而不是单纯增加 token。",
        ),
    ]


def _template_leaf_rows(tree: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    def _walk(node: dict[str, Any]) -> None:
        for item in node.get("summaries", []):
            rows.append(
                {
                    "path": item.get("path"),
                    "template_id": item.get("template_id"),
                    "template_title": item.get("template_title"),
                    "pattern_summary": item.get("pattern_summary"),
                    "usage_count": item.get("usage_count"),
                    "success_count": item.get("success_count"),
                }
            )
        for child in node.get("children", {}).values():
            _walk(child)

    if tree.get("datasets"):
        for dataset, node in sorted(tree["datasets"].items()):
            before = len(rows)
            _walk(node)
            for row in rows[before:]:
                row["path"] = f"{dataset}: {row['path']}"
    elif tree:
        _walk(tree)
    return rows


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
