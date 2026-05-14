"""DoG 实验的科研报告与图资产生成。"""

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
    """读取 DoG 运行目录中的 `metrics.json`。"""

    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总 DoG 运行摘要。"""

    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    summary = SummaryTableView.from_metrics_payload(load_metrics(root))
    grouped = summary.grouped_by_dataset()
    return {
        "run_dir": str(root),
        "experiment_kind": str(manifest.get("experiment_kind") or "static"),
        "row_count": len(summary.rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": {dataset: [row.raw for row in rows] for dataset, rows in grouped.items()},
    }


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """生成 DoG 的中文科研报告。"""

    publish_dir = publish_dir or default_reports_root("dog_graph")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    experiment_kind = str(manifest.get("experiment_kind") or "static")
    if experiment_kind == "paper":
        return _render_paper_report(root, publish_dir, manifest)
    return _render_static_report(root, publish_dir, manifest)


def _render_paper_report(root: Path, publish_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = load_metrics(root)
    graph_diagnostics = load_json_payload(root / "graph_diagnostics.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]
    baseline_row = _overall_row(overall_rows, "tog_iterative_baseline")
    dog_row = _overall_row(overall_rows, "dog_graph_paper")
    by_dataset_rows = [row.raw for row in summary.rows if row.raw.get("dataset") != "overall"]

    lines = [
        "# DoG 原论文高保真复现报告",
        "",
        "## 摘要",
        "",
    ]
    if dog_row is not None and baseline_row is not None:
        lines.append(
            f"- `dog_graph_paper` 相比 `tog_iterative_baseline` 的总体准确率差值为 `{_delta(dog_row, baseline_row):+.4f}`，"
            f"token 比例为 `{_ratio(dog_row, baseline_row, 'total_tokens_mean'):.4f}`。"
        )
    if dog_row is not None:
        lines.append(
            f"- `dog_graph_paper` 的问题简化成功率为 `{float(dog_row.get('simplification_success_rate') or 0.0):.4f}`，"
            f"direct fallback 率为 `{float(dog_row.get('direct_fallback_rate') or 0.0):.4f}`。"
        )
    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 当前 `dog_graph_main` 只代表 DoG 原论文高保真主线，不再代表静态子图消融。",
            "- canonical 对照只有两条：`tog_iterative_baseline` 与 `dog_graph_paper`。",
            "- Freebase 任务固定走：动态关系检索 -> enough-answer 判断 -> 三角色顺序问题简化 -> direct fallback。",
            "- MetaQA 任务固定走：动态图展开 + 三角色顺序问题简化，不强行套入 Freebase 的 enough-answer / fallback 分支。",
            "- 主指标 `accuracy_mean` 已回到论文式准确率口径，不再使用 alias-set token-F1 作为 canonical 主结果。",
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
                "retrieval_hops_mean",
                "simplification_success_rate",
                "direct_fallback_rate",
                "gain_over_baseline",
            ],
        )
    )
    lines.extend(["", "## 分数据集结果", ""])
    lines.extend(
        _render_table(
            by_dataset_rows,
            [
                "dataset",
                "method_name",
                "accuracy_mean",
                "retrieval_hops_mean",
                "simplification_success_rate",
                "direct_fallback_rate",
                "gain_over_baseline",
            ],
        )
    )
    lines.extend(["", "## 过程诊断", ""])
    lines.extend(
        _render_table(
            graph_diagnostics.get("summary_rows", []),
            [
                "dataset",
                "method_name",
                "accuracy_mean",
                "retrieval_hops_mean",
                "simplification_success_rate",
                "direct_fallback_rate",
                "false_positive_relation_rate",
                "enough_answer_yes_rate",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 结论与解释边界",
            "",
            "- 本报告只解释官方论文级流程是否跑通，以及 `dog_graph_paper` 相对 `tog_iterative_baseline` 的趋势关系。",
            "- 如果某个任务没有 enough-answer 或 direct fallback 分支，那是因为官方实现本身对 MetaQA 与 Freebase 任务不完全同构。",
            "- 只要 `dog_graph_paper` 在 `WebQuestions` 与 `GrailQA` 上稳定优于 `tog_iterative_baseline`，且其余任务不系统性落后，就可视为趋势通过。",
            "- 当前报告不把 legacy 静态子图结果混入 canonical 论文结论。",
        ]
    )

    payload = render_family_report_bundle(
        family_name="dog_graph",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_paper_figure_specs(metrics, graph_diagnostics),
    )
    payload["graph_diagnostics"] = str(root / "graph_diagnostics.json")
    return payload


def _render_static_report(root: Path, publish_dir: str | Path, manifest: dict[str, Any]) -> dict[str, Any]:
    metrics = load_metrics(root)
    graph_diagnostics = load_json_payload(root / "graph_diagnostics.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]

    lines = [
        "# DoG 静态子图消融报告",
        "",
        "## 摘要",
        "",
        "- 当前实验线只用于 legacy 静态候选子图对照，不代表 DoG 原论文主结果。",
        "",
        "## 总体结果表",
        "",
    ]
    lines.extend(
        _render_table(
            overall_rows,
            [
                "method_name",
                "accuracy_mean",
                "total_tokens_mean",
                "calls_per_question_mean",
                "subgraph_node_count_mean",
                "subgraph_edge_count_mean",
                "grounded_communication_rate",
                "debate_gain_over_vote",
            ],
        )
    )
    lines.extend(["", "## 图诊断摘要", ""])
    lines.extend(
        _render_table(
            graph_diagnostics.get("summary_rows", []),
            [
                "dataset",
                "method_name",
                "accuracy_mean",
                "grounded_communication_rate",
                "subgraph_node_count_mean",
                "subgraph_edge_count_mean",
                "evidence_triple_count_mean",
                "answer_path_length_mean",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 解释边界",
            "",
            "- 该实验线只解释静态候选子图上的图求解、投票与显式图辩论消融。",
            "- canonical 论文主线请改看 `dog_graph_main` 的高保真复现结果。",
        ]
    )
    payload = render_family_report_bundle(
        family_name="dog_graph",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_static_figure_specs(metrics, graph_diagnostics),
    )
    payload["graph_diagnostics"] = str(root / "graph_diagnostics.json")
    return payload


def _overall_row(rows: list[dict[str, Any]], method_name: str) -> dict[str, Any] | None:
    return next((row for row in rows if row.get("method_name") == method_name), None)


def _delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    return round(float(left.get("accuracy_mean") or 0.0) - float(right.get("accuracy_mean") or 0.0), 6)


def _ratio(left: dict[str, Any], right: dict[str, Any], field: str) -> float:
    denominator = float(right.get(field) or 0.0)
    if denominator <= 0:
        return 0.0
    return round(float(left.get(field) or 0.0) / denominator, 6)


def _build_paper_figure_specs(metrics: dict[str, Any], graph_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_graph_rows = [
        row
        for row in graph_diagnostics.get("summary_rows", [])
        if isinstance(row, dict) and row.get("dataset") == "overall"
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="DoG 论文主线成本-性能前沿",
            caption="canonical 论文主线中，DoG 与迭代基线在总体准确率与平均总 token 上的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="DoG 论文主线效率排序",
            caption="按每千 token 准确率排序的总体效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="DoG 论文主线跨数据集表现",
            caption="各方法在不同论文任务上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="dog_paper_process_profile",
            title="DoG 过程画像",
            caption="比较 simplification success、fallback 与 false-positive relation 压制情况。",
            primary_metric="比率",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "simplification_success_rate": float(row.get("simplification_success_rate") or 0.0),
                    "direct_fallback_rate": float(row.get("direct_fallback_rate") or 0.0),
                    "false_positive_relation_rate": float(row.get("false_positive_relation_rate") or 0.0),
                }
                for row in overall_graph_rows
            ],
            series=[
                ("simplification_success_rate", "问题简化成功率"),
                ("direct_fallback_rate", "direct fallback 率"),
                ("false_positive_relation_rate", "空扩展关系率"),
            ],
            x_label="比率",
            source_kind="graph_diagnostics",
            dataset_scope="overall",
            note="用于观察 DoG 是否真的通过问题简化与关系压缩改善检索链路。",
        ),
    ]


def _build_static_figure_specs(metrics: dict[str, Any], graph_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_graph_rows = [
        row
        for row in graph_diagnostics.get("summary_rows", [])
        if isinstance(row, dict) and row.get("dataset") == "overall"
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="DoG 静态消融成本-性能前沿",
            caption="静态候选子图消融中的总体准确率与平均总 token 关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="DoG 静态消融效率排序",
            caption="按每千 token 准确率排序的静态消融效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="DoG 静态消融跨数据集表现",
            caption="各静态图方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="graph_grounding_profile",
            title="DoG 静态图通信画像",
            caption="总体层面比较 grounded communication rate、证据三元组数量与答案路径长度。",
            primary_metric="比率或长度",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "grounded_communication_rate": float(row.get("grounded_communication_rate") or 0.0),
                    "evidence_triple_count_mean": float(row.get("evidence_triple_count_mean") or 0.0),
                    "answer_path_length_mean": float(row.get("answer_path_length_mean") or 0.0),
                }
                for row in overall_graph_rows
            ],
            series=[
                ("grounded_communication_rate", "Grounded communication rate"),
                ("evidence_triple_count_mean", "证据三元组数量"),
                ("answer_path_length_mean", "答案路径长度"),
            ],
            x_label="比率或长度",
            source_kind="graph_diagnostics",
            dataset_scope="overall",
            note="只用于观察静态子图通信是否真正引用了可追踪证据。",
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
