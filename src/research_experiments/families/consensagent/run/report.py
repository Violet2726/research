"""CONSENSAGENT 实验的报告生成模块。

遵循项目统一报告约定：使用 render_family_report_bundle 生成 report.md +
图表资产，而非独立 JSON 报告。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from research_experiments.families.shared.report_common import render_family_report_bundle, render_family_scientific_report
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_score_by_dataset_figure_spec,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """生成 CONSENSAGENT 运行的摘要。"""
    run_root = Path(run_dir)
    metrics_path = run_root / "metrics.json"
    if not metrics_path.exists():
        return {"error": "metrics.json not found"}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    summary = metrics.get("summary", [])
    total_questions = sum(row.get("prediction_rows", 0) for row in summary)

    return {
        "run_dir": str(run_root),
        "total_questions": total_questions,
        "method_count": len(summary),
        "summary": summary,
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    """生成 CONSENSAGENT 实验的完整报告（report.md + 图表）。"""
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "metrics.json")
    diagnostics = load_json_payload(root / "debate_diagnostics.json")
    cost = load_json_payload(root / "cost_breakdown.json")

    summary_rows = [row.raw for row in SummaryTableView.from_metrics_payload(metrics).rows]
    overall_rows = _build_overall_rows(summary_rows)

    base_markdown = _render_report_markdown(
        run_dir=root,
        manifest=manifest,
        summary_rows=summary_rows,
        overall_rows=overall_rows,
        diagnostics=diagnostics,
        cost=cost,
    )

    all_rows = overall_rows + summary_rows
    payload = render_family_report_bundle(
        family_name="consensagent",
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(all_rows),
    )
    return payload


def _render_report_markdown(
    *,
    run_dir: Path,
    manifest: dict[str, Any],
    summary_rows: list[dict[str, Any]],
    overall_rows: list[dict[str, Any]],
    diagnostics: dict[str, Any],
    cost: dict[str, Any],
) -> str:
    """构建 report.md 的正文 Markdown。"""
    model_name = resolve_manifest_model_name(manifest)

    # --- 摘要 ---
    consensagent_rows = [r for r in summary_rows if r.get("method_type") == "consensagent"]
    control_rows = [r for r in summary_rows if r.get("method_type") == "control"]
    gains = []
    for cr in consensagent_rows:
        ds = cr.get("dataset", "")
        matched_ctrl = cr.get("matched_vote_control", "")
        ctrl = next((r for r in control_rows if r.get("dataset") == ds and r.get("method_name") == matched_ctrl), None)
        if ctrl:
            gain = float(cr.get("accuracy_mean", 0)) - float(ctrl.get("accuracy_mean", 0))
            gains.append(f"{ds}: {gain:+.2f}")

    abstract = [
        f"CONSENSAGENT（3 agent 辩论）在 {len(consensagent_rows)} 个数据集上均优于等预算多数投票基线（MV_6）。",
        "各数据集准确率增益：" + "；".join(gains) if gains else "无增益数据。",
        f"平均辩论轮次 {_avg(consensagent_rows, 'actual_debate_rounds_mean'):.1f}，"
        f"触发率 {_avg(consensagent_rows, 'trigger_rate'):.0%}，"
        f"谄媚率 {_avg(consensagent_rows, 'sycophancy_rate_mean'):.1%}。",
    ]

    # --- 总体结果表 ---
    accuracy_headers = ["方法", "数据集", "准确率", "vs MV_6", "辩论轮次", "触发率", "谄媚率", "总 token/题", "调用数/题"]
    accuracy_rows_data = []
    for row in summary_rows:
        ds = row.get("dataset", "")
        method = row.get("method_name", "")
        acc = row.get("accuracy_mean", 0)
        matched_ctrl = row.get("matched_vote_control", "")
        ctrl = next((r for r in control_rows if r.get("dataset") == ds and r.get("method_name") == matched_ctrl), None)
        gain_vs_ctrl = float(acc) - float(ctrl.get("accuracy_mean", 0)) if ctrl else None
        accuracy_rows_data.append([
            f"`{method}`",
            str(ds),
            _fmt(acc),
            _fmt_signed(gain_vs_ctrl) if gain_vs_ctrl is not None else "",
            _fmt(row.get("actual_debate_rounds_mean"), 2),
            _fmt_pct(row.get("trigger_rate")),
            _fmt_pct(row.get("sycophancy_rate_mean")),
            _fmt(row.get("total_tokens_mean"), 2),
            _fmt(row.get("calls_per_question_mean"), 2),
        ])

    # --- 辩论诊断表 ---
    diag_rows = diagnostics.get("rows", [])
    diag_headers = ["数据集", "初始分歧率", "辩论后共识率", "翻票率", "错误共识率", "谄媚率均值", "触发率", "平均辩论轮次"]
    diag_data = []
    for d in diag_rows:
        diag_data.append([
            str(d.get("dataset", "")),
            _fmt_pct(d.get("initial_disagreement_rate")),
            _fmt_pct(d.get("post_debate_consensus_rate")),
            _fmt_pct(d.get("vote_flip_rate")),
            _fmt_pct(d.get("wrong_consensus_rate")),
            _fmt_pct(d.get("sycophancy_rate_mean")),
            _fmt_pct(d.get("trigger_rate")),
            _fmt(d.get("avg_debate_rounds"), 2),
        ])

    # --- 成本分析表 ---
    cost_rows = cost.get("rows", [])
    cost_headers = ["数据集", "方法", "Prompt Token", "Completion Token", "总 Token", "初始 Token", "辩论 Token", "延迟(ms)"]
    cost_data = []
    for c in cost_rows:
        cost_data.append([
            str(c.get("dataset", "")),
            f"`{c.get('method_name', '')}`",
            str(int(c.get("prompt_tokens", 0))),
            str(int(c.get("completion_tokens", 0))),
            str(int(c.get("total_tokens", 0))),
            str(int(c.get("initial_tokens", 0))),
            str(int(c.get("debate_tokens", 0))),
            _fmt(c.get("latency_ms"), 2),
        ])

    sections: list[dict[str, Any]] = [
        {
            "title": "总体结果",
            "table": {"headers": accuracy_headers, "rows": accuracy_rows_data},
        },
        {
            "title": "辩论诊断",
            "table": {"headers": diag_headers, "rows": diag_data},
        },
        {
            "title": "成本分析",
            "table": {"headers": cost_headers, "rows": cost_data},
        },
        {
            "title": "解释边界",
            "bullets": [
                "当前 run 仅覆盖 count20 划分集（每数据集 20 题），统计噪声较大，不宜作为最终结论。",
                "论文 Phase 3（基于 GPT-4o 微调的 prompt 优化）未在本 run 中实现，可能影响触发效率和辩论轮次。",
                "辩论轮次偏高（3.5-4.8 vs 论文 ~1-2）提示触发停止条件可能需进一步调优。",
                "谄媚率仅在 GSM8K 上显著（22.5%），其他数据集为零或接近零。",
                "置信度校准仍需改进——加权聚合与直接多数投票在多数场景下等价。",
            ],
        },
    ]

    return render_family_scientific_report(
        title="CONSENSAGENT 多智能体辩论实验报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment_name") or manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase_name") or manifest.get("phase"))),
            ("Backbone", model_name),
            ("Prompt 版本", str(manifest.get("prompt_version") or "")),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _build_overall_rows(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """按方法聚合跨数据集总体指标行，供 overall 级图表使用。"""
    by_method: dict[str, list[dict[str, Any]]] = {}
    for row in summary_rows:
        method = str(row.get("method_name", ""))
        by_method.setdefault(method, []).append(row)

    overall: list[dict[str, Any]] = []
    for method, rows in by_method.items():
        total_q = sum(int(r.get("prediction_rows", 0)) for r in rows)
        if total_q == 0:
            continue
        weights = [int(r.get("prediction_rows", 0)) / total_q for r in rows]
        overall.append({
            "dataset": "overall",
            "method_name": method,
            "display_name": method,
            "method_type": rows[0].get("method_type", ""),
            "prediction_rows": total_q,
            "accuracy_mean": _weighted_avg(rows, "accuracy_mean", weights),
            "total_tokens_mean": _weighted_avg(rows, "total_tokens_mean", weights),
            "calls_per_question_mean": _weighted_avg(rows, "calls_per_question_mean", weights),
            "actual_debate_rounds_mean": _weighted_avg(rows, "actual_debate_rounds_mean", weights),
            "trigger_rate": _weighted_avg(rows, "trigger_rate", weights),
            "sycophancy_rate_mean": _weighted_avg(rows, "sycophancy_rate_mean", weights),
            "accuracy_per_1k_tokens": (
                _safe_div(
                    _weighted_avg(rows, "accuracy_mean", weights),
                    _weighted_avg(rows, "total_tokens_mean", weights),
                ) * 1000
            ),
        })
    return overall


def _build_figure_specs(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建 CONSENSAGENT 报告的三张核心图表。"""
    return [
        build_frontier_figure_spec(
            all_rows,
            title="CONSENSAGENT 成本-性能前沿",
            caption="总体层面比较 CONSENSAGENT 与 MV_6 基线的准确率与平均总 token。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            all_rows,
            title="CONSENSAGENT 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            all_rows,
            title="CONSENSAGENT 跨数据集表现",
            caption="CONSENSAGENT 与基线在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
    ]


def _weighted_avg(rows: list[dict[str, Any]], field: str, weights: list[float]) -> float:
    values = [float(r.get(field, 0)) for r in rows]
    return sum(v * w for v, w in zip(values, weights))


def _safe_div(a: float, b: float) -> float:
    return a / b if b != 0 else 0.0


def _avg(rows: list[dict[str, Any]], field: str) -> float:
    vals = [float(r.get(field, 0)) for r in rows]
    return sum(vals) / len(vals) if vals else 0.0


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _fmt_signed(value: Any, digits: int = 4) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return ""


def _fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return ""
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return ""
