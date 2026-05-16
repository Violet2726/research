"""ECON 实验的科研报告与图资产生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.workspace.layout import default_reports_root
from research_experiments.families.shared.report_common import render_family_report_bundle
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload, load_jsonl_rows
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取 ECON 运行目录中的 `metrics.json`。"""

    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总 ECON 运行摘要。"""

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
    """生成 ECON 的中文科研报告。"""

    publish_dir = publish_dir or default_reports_root("econ")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    belief_rows = load_jsonl_rows(root / "belief_trace.jsonl")
    equilibrium_rows = load_jsonl_rows(root / "equilibrium_trace.jsonl")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]
    non_overall_rows = [row.raw for row in summary.non_overall_rows()]

    vote_row = _overall_row(overall_rows, "vote_mv3")
    full_row = _overall_row(overall_rows, "econ_full_comm_r1")
    bne_row = _overall_row(overall_rows, "econ_bne_main")

    lines = [
        "# ECON 低通信协调复现报告",
        "",
        "## 摘要",
        "",
    ]
    if bne_row is not None and vote_row is not None:
        lines.append(
            f"- `econ_bne_main` 相比 `vote_mv3` 的总体准确率差值为 `{_delta(bne_row, vote_row):+.4f}`。"
        )
    if bne_row is not None and full_row is not None:
        lines.append(
            f"- `econ_bne_main` 相比 `econ_full_comm_r1` 的 token 比例为 `{_ratio(bne_row, full_row, 'total_tokens_mean'):.4f}`。"
        )
    if bne_row is not None:
        lines.append(
            f"- `econ_bne_main` 的 correction rate 为 `{float(bne_row.get('correction_rate') or 0.0):.4f}`，"
            f"degradation rate 为 `{float(bne_row.get('degradation_rate') or 0.0):.4f}`。"
        )
    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 当前 `econ_same_context_main` 是 same-context faithful 主线中的 supporting 论文复现 family。",
            "- canonical benchmark 固定为 `GSM8K / StrategyQA / HotpotQA`。",
            "- canonical 方法固定为 `single_agent_cot / vote_mv3 / econ_full_comm_r1 / econ_bne_main`。",
            "- `econ_bne_main` 复现 `独立求解 -> belief state -> 有限动作 equilibrium 选择 -> 一次受控 belief update`。",
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
                "communication_tokens_mean",
                "calls_per_question_mean",
                "correction_rate",
                "degradation_rate",
                "gain_over_vote_mv3",
                "token_ratio_over_full_comm",
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
                "correction_rate",
                "degradation_rate",
                "gain_over_vote_mv3",
                "keep_local_rate",
                "query_best_peer_rate",
                "query_two_peers_rate",
            ],
        )
    )
    lines.extend(["", "## 信念与动作诊断", ""])
    lines.extend(
        _render_table(
            _belief_summary_rows(belief_rows, equilibrium_rows),
            [
                "dataset",
                "method_name",
                "selected_action",
                "belief_score",
                "expected_gain",
                "communication_cost",
                "agreement_ratio",
                "rationale_conflict",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 结论与解释边界",
            "",
            "- 若 `econ_bne_main` 总体高于 `vote_mv3`，且不系统性低于 `econ_full_comm_r1`，则说明低通信协调主叙事成立。",
            "- 若收益只来自更高 token，而不能明显省于 `econ_full_comm_r1`，这条线只能保留为 supporting reproduction。",
            "- 当前 `dog_graph_main` 继续保留为 supporting reproduction；`table_critic_main` 暂停在有效 `count100`。",
        ]
    )
    return render_family_report_bundle(
        family_name="econ",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_figure_specs(metrics),
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


def _belief_summary_rows(
    belief_rows: list[dict[str, Any]],
    equilibrium_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    equilibrium_lookup = {
        (str(row.get("dataset") or ""), str(row.get("sample_id") or "")): row
        for row in equilibrium_rows
    }
    rows: list[dict[str, Any]] = []
    for row in belief_rows:
        key = (str(row.get("dataset") or ""), str(row.get("sample_id") or ""))
        equilibrium = equilibrium_lookup.get(key, {})
        rows.append(
            {
                "dataset": row.get("dataset"),
                "method_name": row.get("method_name"),
                "selected_action": equilibrium.get("selected_action"),
                "belief_score": equilibrium.get("belief_score"),
                "expected_gain": equilibrium.get("expected_gain"),
                "communication_cost": equilibrium.get("communication_cost"),
                "agreement_ratio": row.get("agreement_ratio"),
                "rationale_conflict": row.get("rationale_conflict"),
            }
        )
    return rows


def _build_figure_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="ECON 成本-性能前沿",
            caption="总体结果上，ECON 各方法的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="ECON 效率排序",
            caption="按每千 token 准确率排序的总体效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="ECON 跨数据集表现",
            caption="各方法在不同 same-context 数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="econ_action_mix",
            title="ECON 动作画像",
            caption="总体层面比较 keep_local / adopt_vote / query_best_peer / query_two_peers 的动作占比。",
            primary_metric="动作比例",
            data=[
                {
                    "label": str(row.get("method_name") or ""),
                    "short_label": str(row.get("method_name") or ""),
                    "keep_local_rate": float(row.get("keep_local_rate") or 0.0),
                    "adopt_vote_rate": float(row.get("adopt_vote_rate") or 0.0),
                    "query_best_peer_rate": float(row.get("query_best_peer_rate") or 0.0),
                    "query_two_peers_rate": float(row.get("query_two_peers_rate") or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("keep_local_rate", "keep_local"),
                ("adopt_vote_rate", "adopt_vote"),
                ("query_best_peer_rate", "query_best_peer"),
                ("query_two_peers_rate", "query_two_peers"),
            ],
            x_label="动作比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="观察 belief-driven 协调是否真的把预算花在更有价值的动作上。",
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
        values: list[str] = []
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

