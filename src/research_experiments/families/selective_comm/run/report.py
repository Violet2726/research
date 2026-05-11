"""选择性通信实验的科研报告与图资产生成。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.core.foundation.workspace import default_reports_root
from research_experiments.reporting.analysis_reports import (
    render_frontier_report,
    render_trigger_diagnostic_report,
)
from research_experiments.reporting.report_pipeline import SupplementalReport, render_report_bundle
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.report_views import (
    DiagnosticTableView,
    SummaryTableView,
    load_json_payload,
    load_jsonl_rows,
)
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


METHOD_ORDER = [
    "mv_3",
    "always_communicate",
    "disagreement_triggered",
    "confidence_triggered",
    "hybrid_trigger",
    "voc_trigger_v2",
    "mv_6",
    "sc_6",
]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_json_payload(Path(run_dir) / "policy_metrics.json"))
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
    publish_dir = publish_dir or default_reports_root("selective_comm")
    root = Path(run_dir)
    manifest = load_json_payload(root / "manifest.json")
    metrics = load_json_payload(root / "policy_metrics.json")
    diagnostics = load_json_payload(root / "policy_diagnostics.json")
    oracle = load_json_payload(root / "oracle_trigger_eval.json")
    predictions = load_jsonl_rows(root / "policy_predictions.jsonl")
    base_markdown = _render_markdown(manifest, metrics, diagnostics, oracle, predictions, root)
    return render_report_bundle(
        run_dir=root,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, diagnostics),
        supplemental_reports=[
            SupplementalReport(
                result_key="frontier_report",
                filename="frontier_report.md",
                content=render_frontier_report(metrics.get("summary", []), title="选择性通信前沿附录"),
            ),
            SupplementalReport(
                result_key="trigger_diagnostic_report",
                filename="trigger_diagnostics.md",
                content=render_trigger_diagnostic_report(
                    _analysis_trigger_rows(diagnostics),
                    title="选择性通信触发诊断附录",
                ),
            ),
        ],
    )


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary = SummaryTableView.from_metrics_payload(metrics)
    policy_rows = DiagnosticTableView.from_rows(diagnostics.get("policy_rows", []))
    shared_prefix_rows = DiagnosticTableView.from_rows(diagnostics.get("shared_prefix_rows", []))
    summary_rows = [row.raw for row in summary.rows]
    overall_policy_rows = policy_rows.overall_rows()
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="选择性通信成本-性能前沿",
            caption="总体结果上，各策略的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="选择性通信效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="选择性通信跨数据集表现",
            caption="各策略在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_scatter_figure_spec(
            figure_id="trigger_tradeoff",
            title="触发率权衡",
            caption="总体触发率相对于准确率的变化。",
            primary_metric="准确率",
            data=[
                {
                    "label": row.name,
                    "short_label": row.name,
                    "x": float(row.trigger_rate or 0.0),
                    "y": float(row.accuracy_mean or 0.0),
                    "value": float(row.accuracy_mean or 0.0),
                }
                for row in overall_policy_rows
            ],
            x_label="触发率",
            y_label="准确率",
            source_kind="policy_diagnostics",
            dataset_scope="overall",
            note="左上区域的策略代表在较低触发频率下维持较高准确率。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="shared_prefix_savings",
            title="共享前缀节省比",
            caption="共享前缀执行相对于独立重跑的 token 节省比例。",
            primary_metric="节省比例",
            data=[
                {
                    "label": row.dataset or "unknown",
                    "short_label": row.dataset or "unknown",
                    "shared_prefix_savings_ratio": float(row.shared_prefix_savings_ratio or 0.0),
                }
                for row in shared_prefix_rows.rows
            ],
            series=[("shared_prefix_savings_ratio", "节省比例")],
            x_label="节省比例",
            source_kind="policy_diagnostics",
            dataset_scope="per_dataset",
            note="值越高说明同一份 Stage A / Stage B 被更多策略复用。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="oracle_alignment",
            title="Oracle 对齐情况",
            caption="总体 Oracle 精确率与召回率对比。",
            primary_metric="比率",
            data=[
                {
                    "label": row.name,
                    "short_label": row.name,
                    "precision": float(row.precision or 0.0),
                    "recall": float(row.recall or 0.0),
                }
                for row in overall_policy_rows
            ],
            series=[("precision", "精确率"), ("recall", "召回率")],
            x_label="比率",
            source_kind="policy_diagnostics",
            dataset_scope="overall",
            note="精确率与召回率共同衡量策略是否把通信机会分配给真正有收益的样本。",
        ),
    ]


def _analysis_trigger_rows(diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in diagnostics.get("policy_rows", []):
        rows.append(
            {
                "dataset": row.get("dataset"),
                "policy_name": row.get("policy_name"),
                "method_name": row.get("policy_name"),
                "trigger_rate": row.get("trigger_rate"),
                "early_exit_rate": row.get("early_exit_rate"),
                "oracle_precision": row.get("precision"),
                "oracle_recall": row.get("recall"),
                "false_trigger_rate": row.get("false_trigger_rate"),
                "missed_beneficial_comm_rate": row.get("missed_beneficial_comm_rate"),
                "communication_tokens_mean": row.get("communication_tokens_mean"),
            }
        )
    return rows


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    oracle: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    summary = SummaryTableView.from_metrics_payload(metrics)
    policy_rows = DiagnosticTableView.from_rows(diagnostics.get("policy_rows", []))
    voc_policy_rows = DiagnosticTableView.from_rows(diagnostics.get("voc_policy_rows", []))
    shared_prefix_rows = DiagnosticTableView.from_rows(diagnostics.get("shared_prefix_rows", []))
    recommendation = diagnostics.get("recommended_next_default_policy", {})
    overall_main_rows = _ordered_rows(summary.overall_rows())
    per_dataset_rows = {
        dataset: _ordered_rows(summary.dataset_rows(dataset))
        for dataset in summary.dataset_names()
    }
    best_row = summary.best_by("accuracy_mean", rows=overall_main_rows)
    best_policy_row = policy_rows.best_by("accuracy_mean", rows=policy_rows.overall_rows())
    failure_cases = _select_failure_cases(oracle.get("sample_rows", []), predictions)

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。")
    if best_policy_row is not None:
        abstract.append(f"触发策略中表现最佳的是 `{best_policy_row.name}`。")
    if shared_prefix_rows.rows:
        mean_savings = sum(float(row.shared_prefix_savings_ratio or 0.0) for row in shared_prefix_rows.rows) / len(shared_prefix_rows.rows)
        abstract.append(f"共享前缀机制的平均 token 节省比例约为 {format_float(mean_savings)}。")
    abstract.append(f"当前推荐的下一轮默认策略为 `{recommendation.get('selected_policy', 'hybrid_trigger')}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验回答三件事：是否真的通过共享前缀节省了请求成本；哪些 trigger 策略更接近通信收益 Oracle；默认策略该如何选择。",
                "主指标为准确率；成本指标采用平均总 token / 题和平均通信 token / 题；策略诊断重点是触发率、早退率、误触发率和漏掉有益通信率。",
                "所有 trigger 策略共享相同的 Stage A 与触发后的 Stage B，因此结论可直接归因于策略本身，而不是额外请求差异。",
            ],
        },
        {
            "title": "共享前缀节省情况",
            "bullets": [
                f"`{row.dataset}`：共享执行实际 token=`{format_float(row.shared_actual_tokens, 2)}`，独立重跑 token=`{format_float(row.naive_independent_tokens, 2)}`，节省比例=`{format_float(row.shared_prefix_savings_ratio)}`。"
                for row in shared_prefix_rows.rows
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row.display_name}`",
                        format_float(row.accuracy_mean),
                        format_float(row.communication_tokens_mean, 2),
                        format_float(row.total_tokens_mean, 2),
                        format_float(row.acc_per_1k_tokens, 6),
                    ]
                    for row in overall_main_rows
                ],
            },
        },
        {
            "title": "Trigger 诊断",
            "table": {
                "headers": ["策略", "触发率", "早退率", "Oracle 精确率", "Oracle 召回率", "误触发率", "漏掉有益通信率"],
                "rows": [
                    [
                        f"`{row.name}`",
                        format_float(row.trigger_rate),
                        format_float(row.early_exit_rate),
                        format_float(row.precision),
                        format_float(row.recall),
                        format_float(row.false_trigger_rate),
                        format_float(row.missed_beneficial_comm_rate),
                    ]
                    for row in _ordered_policy_rows(policy_rows.overall_rows())
                ],
            },
        },
        {
            "title": "VoC 诊断",
            "table": {
                "headers": ["策略", "Helpful Recall", "Harmful Trigger Rate", "Neutral Waste Rate", "触发率", "平均通信 token / 题"],
                "rows": [
                    [
                        f"`{row.name}`",
                        format_float(row.number("helpful_recall")),
                        format_float(row.number("harmful_trigger_rate")),
                        format_float(row.number("neutral_waste_rate")),
                        format_float(row.trigger_rate),
                        format_float(row.communication_tokens_mean, 2),
                    ]
                    for row in _ordered_policy_rows(voc_policy_rows.overall_rows())
                ],
            },
            "bullets": [
                f"推荐默认策略：`{recommendation.get('selected_policy', 'hybrid_trigger')}`。",
                f"相对 `always_communicate` 的准确率下降：`{recommendation.get('accuracy_drop_vs_always', 0.0)}`；总 token 降低比例：`{recommendation.get('token_drop_ratio_vs_always', 0.0)}`。",
            ],
        },
        {
            "title": "分数据集表现",
            "tables": [
                {
                    "title": dataset,
                    "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每千 token 准确率"],
                    "rows": [
                        [
                            f"`{row.display_name}`",
                            format_float(row.accuracy_mean),
                            format_float(row.communication_tokens_mean, 2),
                            format_float(row.total_tokens_mean, 2),
                            format_float(row.acc_per_1k_tokens, 6),
                        ]
                        for row in rows
                    ],
                }
                for dataset, rows in per_dataset_rows.items()
            ],
        },
        {
            "title": "典型案例",
            "cases": [
                {
                    "数据集": case["dataset"],
                    "样本 ID": case["sample_id"],
                    "问题预览": case["question_preview"],
                    "金标": case["gold"],
                    "mv_3": f"{case['mv_3_prediction']} / {case['mv_3_score']}",
                    "always_communicate": f"{case['always_prediction']} / {case['always_score']}",
                    "解释": case["reason"],
                }
                for case in failure_cases[:5]
            ],
            "bullets": ["当前阶段未收集到足够稳定的失败案例。"] if not failure_cases else [],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "若某策略的 Oracle 召回率高但误触发率也高，说明它捕捉到了不少有益通信机会，但仍需进一步压缩无效通信。",
                "若共享前缀节省比例稳定较高，则后续多策略评估应继续保持统一前缀，而不是拆成独立请求。",
                "默认策略不应只看总体准确率，还应联合考察 Oracle 精确率、召回率和总 token 降幅。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前 Oracle 仍是基于控制组结果构造的近似标签，更适合作为工程决策辅助，而非最终机制证明。",
                "本报告反映的是当前 phase 的黑盒策略效果，不直接等同于更长周期、更大样本下的最终主结论。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`oracle_trigger_eval.json`、`report.md`、`figure_manifest.json`、`figures/`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核和后续论文引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="选择性通信科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", backbone_name),
            ("Prompt Version", str(manifest.get("prompt_version"))),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _select_failure_cases(
    oracle_rows: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    pred_lookup: dict[tuple[str, str, str], dict[str, Any]] = {}
    for row in predictions:
        pred_lookup[(row["dataset"], row["sample_id"], row["method_name"])] = row

    cases: list[dict[str, Any]] = []
    for row in oracle_rows:
        dataset = row["dataset"]
        sample_id = row["sample_id"]
        mv_3_row = pred_lookup.get((dataset, sample_id, "mv_3"))
        always_row = pred_lookup.get((dataset, sample_id, "always_communicate"))
        disagreement_row = pred_lookup.get((dataset, sample_id, "disagreement_triggered"))
        hybrid_row = pred_lookup.get((dataset, sample_id, "hybrid_trigger"))
        voc_row = pred_lookup.get((dataset, sample_id, "voc_trigger_v2"))
        if mv_3_row is None or always_row is None:
            continue
        reason = None
        if row.get("oracle_label") == "helpful" and voc_row and not voc_row.get("triggered"):
            reason = "always_communicate 能纠错，但 voc_trigger_v2 在该题 early exit，漏掉了有益通信。"
        elif row.get("oracle_label") == "harmful" and voc_row and voc_row.get("triggered"):
            reason = "通信会把答案从对带错，但 voc_trigger_v2 仍然触发了通信。"
        elif row.get("oracle_label") == "neutral" and disagreement_row and disagreement_row.get("triggered"):
            reason = "通信没有带来正确性变化，但 disagreement_triggered 仍然进入了通信。"
        elif row.get("beneficial_communication") and hybrid_row and not hybrid_row.get("triggered"):
            reason = "always_communicate 能纠错，但 hybrid_trigger 在该题 early exit，漏掉了有益通信。"
        elif float(always_row["score"]) < float(mv_3_row["score"]):
            reason = "always_communicate 比无通信的 `mv_3` 更差，说明该题存在通信伤害。"
        if reason is None:
            continue
        cases.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "question_preview": row["question_preview"],
                "gold": mv_3_row["gold"],
                "mv_3_prediction": mv_3_row["prediction"],
                "mv_3_score": mv_3_row["score"],
                "always_prediction": always_row["prediction"],
                "always_score": always_row["score"],
                "reason": reason,
            }
        )
    return cases[:5]


def _ordered_rows(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row.method_name) if row.method_name in METHOD_ORDER else 999)


def _ordered_policy_rows(rows: list[Any]) -> list[Any]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row.method_name) if row.method_name in METHOD_ORDER else 999)
