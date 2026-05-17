"""ColMAD 实验的科研报告与图资产生成。"""

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
    """读取 ColMAD 运行目录中的 `metrics.json`。"""

    return load_json_payload(Path(run_dir) / "metrics.json")


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总 ColMAD 运行摘要。"""

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
    """生成 ColMAD 的中文科研报告。"""

    publish_dir = publish_dir or default_reports_root("colmad")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_metrics(root)
    protocol_diagnostics = load_json_payload(root / "protocol_diagnostics.json")
    summary = SummaryTableView.from_metrics_payload(metrics)
    overall_rows = [row.raw for row in summary.overall_rows()]
    non_overall_rows = [row.raw for row in summary.non_overall_rows()]

    single_row = _overall_row(overall_rows, "single_agent_detector")
    competitive_row = _overall_row(overall_rows, "copmad_competitive")
    collaborative_row = _overall_row(overall_rows, "colmad_collaborative")

    lines = [
        "# ColMAD 协作监督式多智能体辩论复现报告",
        "",
        "## 摘要",
        "",
    ]
    if collaborative_row is not None and competitive_row is not None:
        lines.append(
            f"- `colmad_collaborative` 相比 `copmad_competitive` 的总体准确率差值为 `{_delta(collaborative_row, competitive_row):+.4f}`，"
            f"token 比例为 `{_ratio(collaborative_row, competitive_row, 'total_tokens_mean'):.4f}`。"
        )
    if collaborative_row is not None and single_row is not None:
        lines.append(
            f"- `colmad_collaborative` 相比 `single_agent_detector` 的总体准确率差值为 `{_delta(collaborative_row, single_row):+.4f}`。"
        )
    if collaborative_row is not None:
        lines.append(
            f"- `colmad_collaborative` 的 `correct_to_wrong_shift_rate` 为 `{float(collaborative_row.get('correct_to_wrong_shift_rate') or 0.0):.4f}`，"
            f"`supportive_critique_rate` 为 `{float(collaborative_row.get('supportive_critique_rate') or 0.0):.4f}`。"
        )

    lines.extend(
        [
            "",
            "## 研究问题与实验设计",
            "",
            "- 当前 `colmad_realmistake_main` 是平行论文复现支线，进入 `reproduction_matrix`，但不进入 `faithful_matrix`。",
            "- canonical benchmark 固定为 `ReaLMistake` 的 `math_problem_generation / fine_grained_fact_verification / answerability_classification` 三类错误检测任务。",
            "- canonical 方法固定为 `single_agent_detector / copmad_competitive / colmad_collaborative`。",
            "- `copmad_competitive` 复现竞争性零和辩论；`colmad_collaborative` 复现支持性批评与证据互补的协作监督协议。",
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
                "correct_to_wrong_shift_rate",
                "wrong_to_correct_shift_rate",
                "competitive_hacking_rate",
                "supportive_critique_rate",
            ],
        )
    )
    lines.extend(["", "## 分任务结果", ""])
    lines.extend(
        _render_table(
            non_overall_rows,
            [
                "dataset",
                "method_name",
                "accuracy_mean",
                "gain_over_single_agent",
                "gain_over_competitive",
                "correct_to_wrong_shift_rate",
                "wrong_to_correct_shift_rate",
                "supportive_critique_rate",
            ],
        )
    )
    lines.extend(["", "## 协议诊断", ""])
    lines.extend(
        _render_table(
            protocol_diagnostics.get("summary_rows", []),
            [
                "dataset",
                "method_name",
                "competitive_hacking_rate",
                "supportive_critique_rate",
                "correct_to_wrong_shift_rate",
                "wrong_to_correct_shift_rate",
                "judge_disagreement_rate",
                "evidence_complementarity_rate",
                "fake_evidence_rate",
                "overconfident_claim_rate",
                "fallacious_argument_rate",
            ],
        )
    )
    lines.extend(
        [
            "",
            "## 结论与解释边界",
            "",
            "- 这条线的核心不是一般推理准确率竞赛，而是 error detection / scalable oversight 场景下的竞争式协议与协作式协议对照。",
            "- 若 `colmad_collaborative` 稳定优于 `copmad_competitive`，且相比单智能体仍保留非平凡增益，则说明协作监督式协议在该场景下更符合论文趋势。",
            "- 若增益只依赖更高 token，或 `correct_to_wrong_shift_rate` 没有明显下降，则应冻结为 supporting reproduction，而不是继续扩新协议。",
        ]
    )

    return render_family_report_bundle(
        family_name="colmad",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown="\n".join(lines) + "\n",
        figure_specs=_build_figure_specs(metrics, protocol_diagnostics),
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


def _build_figure_specs(metrics: dict[str, Any], protocol_diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    diagnostics_rows = [row for row in protocol_diagnostics.get("summary_rows", []) if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            overall_rows,
            title="ColMAD 成本-性能前沿",
            caption="总体结果上，各错误检测协议的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            overall_rows,
            title="ColMAD 效率排序",
            caption="按每千 token 准确率排序的总体效率比较。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="ColMAD 跨任务表现",
            caption="各错误检测协议在不同 ReaLMistake 任务上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="colmad_protocol_profile",
            title="ColMAD 协议画像",
            caption="总体层面比较 hacking、supportive critique 与 shift 三类关键比率。",
            primary_metric="比率",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "competitive_hacking_rate": float(row.get("competitive_hacking_rate") or 0.0),
                    "supportive_critique_rate": float(row.get("supportive_critique_rate") or 0.0),
                    "correct_to_wrong_shift_rate": float(row.get("correct_to_wrong_shift_rate") or 0.0),
                    "wrong_to_correct_shift_rate": float(row.get("wrong_to_correct_shift_rate") or 0.0),
                }
                for row in diagnostics_rows
            ],
            series=[
                ("competitive_hacking_rate", "竞争式黑客率"),
                ("supportive_critique_rate", "支持性批评率"),
                ("correct_to_wrong_shift_rate", "正确转错误率"),
                ("wrong_to_correct_shift_rate", "错误转正确率"),
            ],
            x_label="比率",
            source_kind="protocol_diagnostics",
            dataset_scope="overall",
            note="用于观察协议差异究竟来自协作增益，还是来自更高成本与说服偏差。",
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
        rendered = []
        for header in headers:
            value = row.get(header)
            if isinstance(value, float):
                rendered.append(f"{value:.6f}")
            elif isinstance(value, list):
                rendered.append(", ".join(str(item) for item in value))
            elif value is None:
                rendered.append("")
            else:
                rendered.append(str(value))
        lines.append("| " + " | ".join(rendered) + " |")
    return lines
