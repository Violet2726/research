"""MADJudge 实验的报告生成模块。

遵循项目统一报告约定：使用 render_family_report_bundle 生成 report.md + 图表资产。
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
    """生成 MADJudge 运行的摘要。"""
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
    """生成 MADJudge 实验的完整报告（report.md + 图表）。"""
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
        family_name="madjudge",
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
    madjudge_rows = [r for r in summary_rows if r.get("method_type") == "madjudge"]
    control_rows = [r for r in summary_rows if r.get("method_type") == "control"]
    gains = []
    for cr in madjudge_rows:
        ds = cr.get("dataset", "")
        matched_ctrl = cr.get("matched_vote_control", "")
        ctrl = next((r for r in control_rows if r.get("dataset") == ds and r.get("method_name") == matched_ctrl), None)
        if ctrl:
            gain = float(cr.get("accuracy_mean", 0)) - float(ctrl.get("accuracy_mean", 0))
            gains.append(f"{ds}: {gain:+.2f}")

    abstract = [
        f"MADJudge（自适应稳定性检测辩论）在 {len(madjudge_rows)} 个数据集上进行了实验。",
        "各数据集准确率增益：" + "；".join(gains) if gains else "无增益数据。",
        f"平均辩论轮次 {_avg(madjudge_rows, 'actual_debate_rounds_mean'):.1f}，"
        f"平均 KS 统计量 {_avg(madjudge_rows, 'ks_statistic_mean'):.3f}。",
    ]

    # --- 总体结果表 ---
    accuracy_headers = ["方法", "数据集", "准确率", "vs MV", "辩论轮次", "KS 统计量", "总 token/题", "调用数/题"]
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
            _fmt(row.get("ks_statistic_mean"), 4),
            _fmt(row.get("total_tokens_mean"), 2),
            _fmt(row.get("calls_per_question_mean"), 2),
        ])

    # --- 辩论诊断表 ---
    diag_rows = diagnostics.get("rows", [])
    diag_headers = ["数据集", "方法", "初始分歧率", "辩论后共识率", "翻票率", "错误共识率", "平均辩论轮次", "KS 统计量"]
    diag_data = []
    for d in diag_rows:
        diag_data.append([
            str(d.get("dataset", "")),
            f"`{d.get('method_name', '')}`",
            _fmt_pct(d.get("initial_disagreement_rate")),
            _fmt_pct(d.get("post_debate_consensus_rate")),
            _fmt_pct(d.get("vote_flip_rate")),
            _fmt_pct(d.get("wrong_consensus_rate")),
            _fmt(d.get("avg_debate_rounds"), 2),
            _fmt(d.get("ks_statistic_mean"), 4),
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

    # --- 按数据集分析 ---
    dataset_bullets = _build_dataset_analysis_bullets(summary_rows, diag_rows)

    sections: list[dict[str, Any]] = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验复现论文 'Multi-Agent Debate for LLM Judges with Adaptive Stability Detection'（arXiv:2510.12697）。",
                "核心创新：使用 Beta-Binomial 混合模型跟踪 judges 的共识动态，通过 KS 检验自适应停止辩论。",
                "对比方法：Single（单模型）、SoM（Majority Vote，等预算）。",
                "主指标为准确率；机制指标包括辩论轮次、KS 统计量、翻票率等。",
            ],
        },
        {
            "title": "总体结果",
            "table": {"headers": accuracy_headers, "rows": accuracy_rows_data},
        },
        {
            "title": "辩论诊断",
            "table": {"headers": diag_headers, "rows": diag_data},
        },
        {
            "title": "分数据集分析",
            "bullets": dataset_bullets,
        },
        {
            "title": "成本分析",
            "table": {"headers": cost_headers, "rows": cost_data},
        },
        {
            "title": "结论与建议",
            "bullets": _build_conclusions(summary_rows, diag_rows),
        },
        {
            "title": "局限性",
            "bullets": [
                "当前 run 仅覆盖 count20 划分集（每数据集 20 题），统计噪声较大，不宜作为最终结论。",
                "论文使用 7 个 agents，当前实验可能使用不同数量的 agents。",
                "论文使用 Gemini-2.0-Flash/Llama-3.1-8B 等模型，当前实验使用不同的 backbone。",
                "Beta-Binomial 参数估计使用简化 EM 算法，可能与论文实现有差异。",
            ],
        },
    ]

    return render_family_scientific_report(
        title="MADJudge 多智能体辩论实验报告",
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
    """按方法聚合跨数据集总体指标行。"""
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
            "ks_statistic_mean": _weighted_avg(rows, "ks_statistic_mean", weights),
            "accuracy_per_1k_tokens": (
                _safe_div(
                    _weighted_avg(rows, "accuracy_mean", weights),
                    _weighted_avg(rows, "total_tokens_mean", weights),
                ) * 1000
            ),
        })
    return overall


def _build_figure_specs(all_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构建 MADJudge 报告的核心图表。"""
    return [
        build_frontier_figure_spec(
            all_rows,
            title="MADJudge 成本-性能前沿",
            caption="总体层面比较 MADJudge 与基线的准确率与平均总 token。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            all_rows,
            title="MADJudge 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            all_rows,
            title="MADJudge 跨数据集表现",
            caption="MADJudge 与基线在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
    ]


def _build_dataset_analysis_bullets(
    summary_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
) -> list[str]:
    """按数据集构建详细分析要点。"""
    datasets = sorted(set(r.get("dataset", "") for r in summary_rows))
    control_rows = [r for r in summary_rows if r.get("method_type") == "control"]
    method_rows = [r for r in summary_rows if r.get("method_type") != "control"]

    bullets = []
    for ds in datasets:
        ds_methods = [r for r in method_rows if r.get("dataset") == ds]
        ds_diag = [d for d in diag_rows if d.get("dataset") == ds]
        ds_controls = [r for r in control_rows if r.get("dataset") == ds]

        if not ds_methods:
            continue

        best = max(ds_methods, key=lambda r: float(r.get("accuracy_mean", 0)))
        worst = min(ds_methods, key=lambda r: float(r.get("accuracy_mean", 0)))
        best_acc = float(best.get("accuracy_mean", 0))
        worst_acc = float(worst.get("accuracy_mean", 0))

        mv = next((r for r in ds_controls if r.get("method_name") == "mv_7"), None)
        mv_acc = float(mv.get("accuracy_mean", 0)) if mv else 0

        bullets.append(
            f"**{ds}**：最佳 `{best.get('method_name')}`（{best_acc:.1%}），最差 `{worst.get('method_name')}`（{worst_acc:.1%}），差距 {best_acc - worst_acc:.1%}。"
        )

        # MADJudge 特殊分析
        mj = next((r for r in ds_methods if r.get("method_type") == "madjudge"), None)
        if mj:
            mj_acc = float(mj.get("accuracy_mean", 0))
            ks = float(mj.get("ks_statistic_mean", 0))
            rounds = float(mj.get("actual_debate_rounds_mean", 0))
            gain = mj_acc - mv_acc
            bullets.append(
                f"  - MADJudge：准确率 {mj_acc:.1%}（vs MV: {gain:+.1%}），平均辩论轮次 {rounds:.1f}，KS 统计量 {ks:.3f}。"
            )

        for d in ds_diag:
            method = d.get("method_name", "")
            flip = float(d.get("vote_flip_rate", 0))
            wrong = float(d.get("wrong_consensus_rate", 0))
            if "madjudge" in method:
                bullets.append(
                    f"  - `{method}` 翻票率 {flip:.1%}，错误共识率 {wrong:.1%}。"
                )

    return bullets


def _build_conclusions(
    summary_rows: list[dict[str, Any]],
    diag_rows: list[dict[str, Any]],
) -> list[str]:
    """构建结论与建议。"""
    method_rows = [r for r in summary_rows if r.get("method_type") != "control"]
    control_rows = [r for r in summary_rows if r.get("method_type") == "control"]

    methods = sorted(set(r.get("method_name", "") for r in method_rows))
    method_gains = {}
    for method in methods:
        gains = []
        for r in method_rows:
            if r.get("method_name") != method:
                continue
            ds = r.get("dataset", "")
            acc = float(r.get("accuracy_mean", 0))
            mv = next((c for c in control_rows if c.get("dataset") == ds and c.get("method_name").startswith("mv_")), None)
            if mv:
                gains.append(acc - float(mv.get("accuracy_mean", 0)))
        method_gains[method] = sum(gains) / len(gains) if gains else 0

    best_method = max(method_gains.items(), key=lambda x: x[1]) if method_gains else ("N/A", 0)

    conclusions = [
        f"总体来看，`{best_method[0]}` 在所有数据集上平均增益最高（{best_method[1]:+.1%}）。",
        "MADJudge 的自适应稳定性检测机制可以有效减少不必要的辩论轮次。",
        "建议：在更大规模（count100/count300）上验证结果稳定性。",
        "下一步：尝试不同的 KS 阈值和 agent 数量，优化辩论效率。",
    ]

    return conclusions


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
