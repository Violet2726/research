"""CRED-V 报告渲染。"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from research_experiments.core.io import write_json
from research_experiments.family_runtime.artifact_index import load_metrics_payload, resolve_run_artifact_index
from research_experiments.family_runtime.report_bundle import (
    render_family_report_bundle,
    render_family_scientific_report,
)
from research_experiments.reporting.report_views import SummaryTableView, load_json_payload
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import format_float, render_run_reproducibility_section
from research_experiments.workspace.layout import default_reports_root


def load_metrics(run_dir: str | Path, *, family_name: str = "cred_v") -> dict[str, Any]:
    return load_metrics_payload(run_dir, family_name=family_name)


def summarize_run(run_dir: str | Path, *, family_name: str = "cred_v") -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics(run_dir, family_name=family_name))
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
    *,
    family_name: str = "cred_v",
    display_name: str = "CRED-V",
) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root(family_name)
    index = resolve_run_artifact_index(run_dir, family_name=family_name)
    root = index.run_dir
    manifest = load_json_payload(index.manifest_path)
    metrics = load_metrics(root, family_name=family_name)
    router_eval = load_json_payload(root / "diagnostics" / "router_eval.json")
    debate_diagnostics = load_json_payload(root / "diagnostics" / "debate_diagnostics.json")
    output_protocol_diagnostics = load_json_payload(root / "diagnostics" / "output_protocol_diagnostics.json")
    comparison = _build_comparison_payload(manifest, metrics)
    comparison_path = root / "exports" / "cred_comparison.json"
    write_json(comparison_path, comparison)
    paper_summary_path = root / "exports" / "paper_summary.csv"
    _write_paper_summary(paper_summary_path, metrics.get("summary", []))
    base_markdown = _render_markdown(
        manifest,
        metrics,
        router_eval,
        debate_diagnostics,
        output_protocol_diagnostics,
        comparison,
        root,
        display_name=display_name,
    )
    payload = render_family_report_bundle(
        family_name=family_name,
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, display_name=display_name),
    )
    payload["cred_comparison"] = str(comparison_path)
    payload["paper_summary"] = str(paper_summary_path)
    return payload


def _build_figure_specs(metrics: dict[str, Any], *, display_name: str = "CRED-V") -> list[dict[str, Any]]:
    summary_rows = [row for row in metrics.get("summary", []) if row.get("dataset") != "overall_micro"]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    cred_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("cred_")]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title=f"{display_name} 成本-性能前沿",
            caption="CRED-V 与 no-comm 控制组在 overall macro 口径下的准确率与平均 token。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title=f"{display_name} 效率排序",
            caption="overall macro 下每千 token 准确率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            [row for row in summary_rows if row.get("dataset") != "overall"],
            title=f"{display_name} 跨数据集准确率",
            caption="各方法在每个 benchmark 上的准确率。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="cred_mechanism_diagnostics",
            title=f"{display_name} 机制诊断",
            caption="CRED 方法的触发率、纠正率、伤害率与 debate 内部增益。",
            primary_metric="比率或差值",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "trigger_rate": float(row.get("trigger_rate") or 0.0),
                    "corrected_rate": float(row.get("corrected_rate") or 0.0),
                    "harmed_rate": float(row.get("harmed_rate") or 0.0),
                    "debate_gain": float(row.get("debate_gain_over_initial_vote") or 0.0),
                }
                for row in cred_rows
            ],
            series=[
                ("trigger_rate", "触发率"),
                ("corrected_rate", "纠正率"),
                ("harmed_rate", "伤害率"),
                ("debate_gain", "debate 增益"),
            ],
            x_label="比率或差值",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="CRED 的核心验收是纠正率高于伤害率，并在成本受控下取得正向 debate gain。",
        ),
    ]


def _build_comparison_payload(manifest: dict[str, Any], metrics: dict[str, Any]) -> dict[str, Any]:
    method_order = list(manifest.get("method_order") or [])
    dataset_order = list(manifest.get("dataset_order") or [])
    summary_rows = list(metrics.get("summary", []))
    return {
        "experiment": manifest.get("experiment"),
        "phase": manifest.get("phase"),
        "method_order": method_order,
        "dataset_order": dataset_order,
        "overall_macro_rows": _ordered_rows(summary_rows, dataset="overall", method_order=method_order),
        "overall_micro_rows": _ordered_rows(summary_rows, dataset="overall_micro", method_order=method_order),
        "per_dataset_rows": {
            dataset: _ordered_rows(summary_rows, dataset=dataset, method_order=method_order)
            for dataset in dataset_order
        },
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    router_eval: dict[str, Any],
    debate_diagnostics: dict[str, Any],
    output_protocol_diagnostics: dict[str, Any],
    comparison: dict[str, Any],
    run_dir: Path,
    *,
    display_name: str = "CRED-V",
) -> str:
    del debate_diagnostics
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = comparison["overall_macro_rows"]
    cred_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("cred_")]
    best = max(overall_rows, key=lambda row: float(row.get("accuracy_mean") or 0.0)) if overall_rows else None
    abstract = [
        f"{display_name} 将多候选预提交、分裂投票路由、定向反驳验证和生存分数聚合串联起来，目标是验证交互是否带来低伤害的正向纠错。",
    ]
    if best is not None:
        abstract.append(f"本 run overall macro 最高准确率方法为 `{best['method_name']}`，准确率 {format_float(best.get('accuracy_mean'))}。")

    sections = [
        {
            "title": "理论假设",
            "bullets": [
                "固定轮次 MAD 的收益常被初始投票解释；CRED-V 只在 router 判定有高价值错误风险时触发反驳。",
                "反驳必须给出可证伪攻击；聚合只在挑战者具备足够 survival margin 和具体证据时覆盖初始赢家。",
                "主指标同时观察 accuracy、token、trigger rate、corrected rate、harmed rate 与 debate gain。",
            ],
        },
        {
            "title": "总体主表（Macro）",
            "table": {
                "headers": ["方法", "类型", "准确率", "vs cot_1", "vs best no-comm", "初始 vote", "debate 增益", "触发率", "总 token", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row["method_type"]),
                        format_float(row.get("accuracy_mean")),
                        _signed_float(row.get("accuracy_delta_vs_cot_1")),
                        _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                        format_float(row.get("initial_vote_accuracy_mean")),
                        _signed_float(row.get("debate_gain_over_initial_vote")),
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("accuracy_per_1k_tokens"), 6),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "CRED 机制诊断",
            "table": {
                "headers": ["方法", "纠正率", "伤害率", "翻票率", "通信 token", "调用数"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("corrected_rate")),
                        format_float(row.get("harmed_rate")),
                        format_float(row.get("flip_rate")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                    ]
                    for row in cred_rows
                ],
            },
        },
        {
            "title": "Router 诊断",
            "table": {
                "headers": ["数据集", "题数", "触发率", "平均风险数", "平均证据质量", "平均反驳调用"],
                "rows": [
                    [
                        str(row.get("dataset")),
                        str(row.get("question_count")),
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("avg_risk_count")),
                        format_float(row.get("avg_evidence_quality")),
                        format_float(row.get("refutation_calls_mean")),
                    ]
                    for row in router_eval.get("summary_rows", [])
                ],
            },
        },
        {
            "title": "输出协议诊断",
            "table": {
                "headers": ["方法/阶段", "协议失败率", "原因缺失率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("protocol_failure_rate")),
                        format_float(row.get("reason_missing_rate")),
                    ]
                    for row in output_protocol_diagnostics.get("rows", [])
                    if row.get("dataset") == "overall"
                ],
            },
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`router_eval.json`、`debate_diagnostics.json`、`cred_comparison.json`、`paper_summary.csv`。",
                f"summary 行数：`{len(SummaryTableView.from_metrics_payload(metrics).rows)}`。",
            ],
        ),
    ]
    return render_family_scientific_report(
        title=f"{display_name} 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("CRED Output Protocol", str(manifest.get("cred_output_protocol"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _write_paper_summary(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "dataset",
        "aggregate_kind",
        "model_name",
        "method_name",
        "method_type",
        "question_count",
        "accuracy_mean",
        "accuracy_delta_vs_cot_1",
        "accuracy_delta_vs_best_no_comm",
        "initial_vote_accuracy_mean",
        "debate_gain_over_initial_vote",
        "trigger_rate",
        "corrected_rate",
        "harmed_rate",
        "communication_tokens_mean",
        "total_tokens_mean",
        "calls_per_question_mean",
        "accuracy_per_1k_tokens",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _ordered_rows(summary_rows: list[dict[str, Any]], *, dataset: str, method_order: list[str]) -> list[dict[str, Any]]:
    order = {name: index for index, name in enumerate(method_order)}
    return sorted([row for row in summary_rows if row.get("dataset") == dataset], key=lambda row: order.get(str(row.get("method_name")), 999))


def _signed_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):+.4f}"
