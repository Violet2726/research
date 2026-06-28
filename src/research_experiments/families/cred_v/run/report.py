"""CRED-V 任务验证报告渲染。"""

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
    verification_diagnostics = load_json_payload(root / "diagnostics" / "debate_diagnostics.json")
    output_protocol_diagnostics = load_json_payload(root / "diagnostics" / "output_protocol_diagnostics.json")
    comparison = _build_comparison_payload(manifest, metrics)
    comparison_path = root / "exports" / "cred_comparison.json"
    write_json(comparison_path, comparison)
    paper_summary_path = root / "exports" / "paper_summary.csv"
    _write_paper_summary(paper_summary_path, metrics.get("summary", []))
    base_markdown = _render_rfs_markdown(
        manifest,
        metrics,
        router_eval,
        verification_diagnostics,
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
        figure_specs=_build_rfs_figure_specs(metrics, display_name=display_name),
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
            figure_id="cred_v_mechanism_diagnostics",
            title=f"{display_name} 任务验证机制诊断",
            caption="CRED-V 方法的触发率、纠正率、伤害率与验证增益。",
            primary_metric="比率或差值",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "trigger_rate": float(row.get("trigger_rate") or 0.0),
                    "corrected_rate": float(row.get("corrected_rate") or 0.0),
                    "harmed_rate": float(row.get("harmed_rate") or 0.0),
                    "promotion_precision": float(row.get("promotion_precision") or 0.0),
                    "verification_gain": float(row.get("verification_gain_over_initial_vote") or row.get("debate_gain_over_initial_vote") or 0.0),
                }
                for row in cred_rows
            ],
            series=[
                ("trigger_rate", "触发率"),
                ("corrected_rate", "纠正率"),
                ("harmed_rate", "伤害率"),
                ("promotion_precision", "promotion precision"),
                ("verification_gain", "验证增益"),
            ],
            x_label="比率或差值",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="CRED-V 的核心验收是验证触发子集上纠正多于伤害，并在成本受控下取得正向净增益。",
        ),
    ]


def _build_rfs_figure_specs(metrics: dict[str, Any], *, display_name: str = "CRED-V") -> list[dict[str, Any]]:
    summary_rows = [row for row in metrics.get("summary", []) if row.get("dataset") != "overall_micro"]
    overall_rows = [row for row in summary_rows if row.get("dataset") == "overall"]
    cred_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("cred_")]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title=f"{display_name} 成本-性能前沿",
            caption="CRED-RFS 与 no-comm 控制组在 overall macro 口径下的准确率与平均 token。",
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
            figure_id="cred_v_mechanism_diagnostics",
            title=f"{display_name} RFS 机制诊断",
            caption="CRED-RFS 方法的触发率、纠正率、伤害率、promotion precision 与选择增益。",
            primary_metric="比率或差值",
            data=[
                {
                    "label": row["method_name"],
                    "short_label": row["method_name"],
                    "trigger_rate": float(row.get("trigger_rate") or 0.0),
                    "corrected_rate": float(row.get("corrected_rate") or 0.0),
                    "harmed_rate": float(row.get("harmed_rate") or 0.0),
                    "promotion_precision": float(row.get("promotion_precision") or 0.0),
                    "selection_gain": float(row.get("debate_gain_over_initial_vote") or row.get("verification_gain_over_initial_vote") or 0.0),
                }
                for row in cred_rows
            ],
            series=[
                ("trigger_rate", "触发率"),
                ("corrected_rate", "纠正率"),
                ("harmed_rate", "伤害率"),
                ("promotion_precision", "promotion precision"),
                ("selection_gain", "选择增益"),
            ],
            x_label="比率或差值",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="CRED-RFS 的核心验收是弱分裂子集上的纠正多于伤害，并在 paired comparison 中相对 sc_5 取得正向显著性。",
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
        "per_dataset_rows": {dataset: _ordered_rows(summary_rows, dataset=dataset, method_order=method_order) for dataset in dataset_order},
        "paired_comparisons": list(metrics.get("paired_comparisons", [])),
    }


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    router_eval: dict[str, Any],
    verification_diagnostics: dict[str, Any],
    output_protocol_diagnostics: dict[str, Any],
    comparison: dict[str, Any],
    run_dir: Path,
    *,
    display_name: str = "CRED-V",
) -> str:
    del verification_diagnostics
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = comparison["overall_macro_rows"]
    cred_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("cred_")]
    best = max(overall_rows, key=lambda row: float(row.get("accuracy_mean") or 0.0)) if overall_rows else None
    abstract = [
        f"{display_name} 将五路 SC 对齐候选、弱分歧路由、单次任务验证器和晋级证书聚合串联起来，目标是检验少数正确候选能否低伤害晋级。",
    ]
    if best is not None:
        abstract.append(f"本 run overall macro 最高准确率方法为 `{best['method_name']}`，准确率 {format_float(best.get('accuracy_mean'))}。")

    sections = [
        {
            "title": "理论假设",
            "bullets": [
                "Stage A 已经达到 `sc_5` 候选质量，CRED-V 不再用开放辩论增加噪声。",
                "Router 只在 5 路候选低于强多数时触发验证，主要攻击弱分歧样本里的错误多数。",
                "Verifier 输出 promotion certificate；只有分数差、置信度和具体证据同时过门槛时，challenger 才能覆盖初始赢家。",
                "主指标同时观察 accuracy、token、trigger rate、corrected rate、harmed rate 与 verification gain。",
            ],
        },
        {
            "title": "总体主表",
            "table": {
                "headers": ["方法", "类型", "准确率", "vs cot_1", "vs best no-comm", "初始 vote", "验证增益", "触发率", "总 token", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row["method_type"]),
                        format_float(row.get("accuracy_mean")),
                        _signed_float(row.get("accuracy_delta_vs_cot_1")),
                        _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                        format_float(row.get("initial_vote_accuracy_mean")),
                        _signed_float(row.get("verification_gain_over_initial_vote") or row.get("debate_gain_over_initial_vote")),
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("accuracy_per_1k_tokens"), 6),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "CRED-V 机制诊断",
            "table": {
                "headers": [
                    "method",
                    "corrected_rate",
                    "harmed_rate",
                    "flip_rate",
                    "promotion_precision",
                    "harm_per_correction",
                    "actual_gain",
                    "stage_candidate_oracle",
                    "candidate_pool_oracle",
                    "selection_recall",
                    "expansion_oracle_gain",
                    "expansion_trigger_rate",
                    "false_consensus_triggers",
                    "false_consensus_recovered",
                    "safe_repair_count",
                    "hetero_agreement_count",
                    "math_repair_count",
                    "hotpot_span_repair_count",
                    "pairwise_duels",
                    "pairwise_wins",
                    "gpqa_unanimous_duels",
                    "safe_corrected",
                    "safe_harmed",
                    "blocked_2of3",
                    "blocked_mmlu",
                    "blocked_strategyqa",
                    "method_expansion_calls",
                    "shadow_corrected",
                    "shadow_harmed",
                    "shadow_precision",
                    "shadow_net_gain",
                    "shadow_possible_gain",
                    "shadow_gate_passed",
                    "duel_invalid",
                    "duel_retry_recoverable",
                    "minority_probes",
                    "non_answer_blocked",
                    "validator_pass_count",
                    "single_pro_blocked",
                    "verification_tokens",
                    "calls",
                ],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("corrected_rate")),
                        format_float(row.get("harmed_rate")),
                        format_float(row.get("flip_rate")),
                        format_float(row.get("promotion_precision")),
                        format_float(row.get("harm_per_correction"), 2),
                        _signed_float(row.get("actual_gain")),
                        format_float(row.get("stage_candidate_oracle_accuracy")),
                        format_float(row.get("candidate_pool_oracle_accuracy")),
                        format_float(row.get("selection_recall_on_pool_correct")),
                        _signed_float(row.get("expansion_oracle_gain")),
                        format_float(row.get("expansion_trigger_rate")),
                        str(row.get("false_consensus_trigger_count") or 0),
                        str(row.get("false_consensus_recovered_count") or 0),
                        str(row.get("safe_repair_count") or 0),
                        str(row.get("hetero_agreement_count") or 0),
                        str(row.get("math_repair_count") or 0),
                        str(row.get("hotpot_span_repair_count") or 0),
                        str(row.get("pairwise_duel_count") or 0),
                        str(row.get("pairwise_duel_win_count") or 0),
                        str(row.get("gpqa_unanimous_duel_count") or 0),
                        str(row.get("safe_selector_corrected_count") or 0),
                        str(row.get("safe_selector_harmed_count") or 0),
                        str(row.get("blocked_2of3_pairwise_count") or 0),
                        str(row.get("blocked_mmlu_pairwise_count") or 0),
                        str(row.get("blocked_strategyqa_probe_count") or 0),
                        str(row.get("method_expansion_call_count") or 0),
                        str(row.get("shadow_counterfactual_corrected_count") or 0),
                        str(row.get("shadow_counterfactual_harmed_count") or 0),
                        format_float(row.get("shadow_precision")),
                        _signed_float(row.get("shadow_net_gain")),
                        _signed_float(row.get("shadow_possible_gain")),
                        str(row.get("shadow_gate_passed_count", row.get("shadow_gate_passed")) or 0),
                        str(row.get("duel_invalid_count") or 0),
                        str(row.get("duel_retry_recoverable_count") or 0),
                        str(row.get("minority_probe_count") or 0),
                        str(row.get("non_answer_candidate_blocked_count") or 0),
                        str(row.get("validator_pass_count") or 0),
                        str(row.get("single_pro_promotion_blocked_count") or 0),
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
                "headers": ["数据集", "题数", "触发率", "平均风险数", "平均证据质量", "平均验证调用"],
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
            ("CRED Stage A Protocol", str(manifest.get("cred_stage_a_output_protocol") or manifest.get("cred_output_protocol"))),
            ("CRED Verification Protocol", str(manifest.get("cred_verification_output_protocol") or manifest.get("cred_output_protocol"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _render_rfs_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    router_eval: dict[str, Any],
    verification_diagnostics: dict[str, Any],
    output_protocol_diagnostics: dict[str, Any],
    comparison: dict[str, Any],
    run_dir: Path,
    *,
    display_name: str = "CRED-V",
) -> str:
    del verification_diagnostics
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = comparison["overall_macro_rows"]
    cred_rows = [row for row in overall_rows if str(row.get("method_name") or "").startswith("cred_")]
    paired_rows = [
        row
        for row in comparison.get("paired_comparisons", [])
        if row.get("dataset") == "overall" and str(row.get("method_name") or "").startswith("cred_")
    ]
    best = max(overall_rows, key=lambda row: float(row.get("accuracy_mean") or 0.0)) if overall_rows else None
    abstract = [
        f"{display_name} 当前主线采用 CRED-RFS：先生成自由推理候选，再对弱分裂样本做选择性算力扩展，最后用保守门控聚合。",
        "本报告把 candidate gain、selection loss、repair gain 和 adaptive compute gain 分开统计，避免把普通投票收益误写成验证收益。",
    ]
    if best is not None:
        abstract.append(f"本 run overall macro 最优方法为 `{best['method_name']}`，准确率 {format_float(best.get('accuracy_mean'))}。")

    sections = [
        {
            "title": "理论假设",
            "bullets": [
                "Stage A 必须保持自由推理；结构化 JSON 只用于旁路候选或报告，不压缩主求解过程。",
                "强多数默认锁定，只有数学等价或 HotpotQA context span 这类确定性修复可以覆盖。",
                "弱分裂样本追加自由推理候选；MC 任务可追加 choice shuffle 候选，但单个 pro 候选没有翻票权。",
                "主结论必须同时观察 accuracy、selection loss、promotion precision、harm_per_correction、token 和 paired significance。",
            ],
        },
        {
            "title": "总体主表",
            "table": {
                "headers": ["方法", "类型", "准确率", "vs cot_1", "vs best no-comm", "初始 vote", "选择增益", "触发率", "总 token", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        str(row["method_type"]),
                        format_float(row.get("accuracy_mean")),
                        _signed_float(row.get("accuracy_delta_vs_cot_1")),
                        _signed_float(row.get("accuracy_delta_vs_best_no_comm")),
                        format_float(row.get("initial_vote_accuracy_mean")),
                        _signed_float(row.get("debate_gain_over_initial_vote") or row.get("verification_gain_over_initial_vote")),
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("accuracy_per_1k_tokens"), 6),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "RFS 机制诊断",
            "table": {
                "headers": [
                    "method",
                    "corrected_rate",
                    "harmed_rate",
                    "promotion_precision",
                    "harm_per_correction",
                    "actual_gain",
                    "stage_candidate_oracle",
                    "candidate_pool_oracle",
                    "selection_loss",
                    "adaptive_trigger_rate",
                    "math_repair",
                    "hotpot_repair",
                    "shuffle_agreement",
                    "single_pro_blocked",
                    "strong_locked",
                    "gpqa_unanimous_duels",
                    "safe_corrected",
                    "safe_harmed",
                    "blocked_2of3",
                    "method_expansion_calls",
                    "shadow_corrected",
                    "shadow_harmed",
                    "shadow_precision",
                    "shadow_net_gain",
                    "duel_invalid",
                    "adaptive_tokens",
                    "calls",
                ],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("corrected_rate")),
                        format_float(row.get("harmed_rate")),
                        format_float(row.get("promotion_precision")),
                        format_float(row.get("harm_per_correction"), 2),
                        _signed_float(row.get("actual_gain")),
                        format_float(row.get("stage_candidate_oracle_accuracy")),
                        format_float(row.get("candidate_pool_oracle_accuracy")),
                        format_float(row.get("selection_loss")),
                        format_float(row.get("adaptive_trigger_rate") or row.get("expansion_trigger_rate")),
                        str(row.get("math_repair_count") or 0),
                        str(row.get("hotpot_span_repair_count") or 0),
                        str(row.get("choice_shuffle_agreement_count") or 0),
                        str(row.get("single_pro_promotion_blocked_count") or 0),
                        str(row.get("strong_majority_locked_count") or 0),
                        str(row.get("gpqa_unanimous_duel_count") or 0),
                        str(row.get("safe_selector_corrected_count") or 0),
                        str(row.get("safe_selector_harmed_count") or 0),
                        str(row.get("blocked_2of3_pairwise_count") or 0),
                        str(row.get("method_expansion_call_count") or 0),
                        str(row.get("shadow_counterfactual_corrected_count") or 0),
                        str(row.get("shadow_counterfactual_harmed_count") or 0),
                        format_float(row.get("shadow_precision")),
                        _signed_float(row.get("shadow_net_gain")),
                        str(row.get("duel_invalid_count") or 0),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                    ]
                    for row in cred_rows
                ],
            },
        },
        {
            "title": "Paired 显著性",
            "table": {
                "headers": ["method", "reference", "delta", "wins", "losses", "ties", "McNemar p", "bootstrap CI", "positive"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        f"`{row['reference_method']}`",
                        _signed_float(row.get("accuracy_delta")),
                        str(row.get("wins") or 0),
                        str(row.get("losses") or 0),
                        str(row.get("ties") or 0),
                        format_float(row.get("mcnemar_p"), 6),
                        f"[{_signed_float(row.get('bootstrap_ci_low'))}, {_signed_float(row.get('bootstrap_ci_high'))}]",
                        "yes" if row.get("bootstrap_significant_positive") else "no",
                    ]
                    for row in paired_rows
                ],
            },
        },
        {
            "title": "Router 诊断",
            "table": {
                "headers": ["数据集", "题数", "触发率", "平均风险数", "平均证据质量", "平均扩展调用"],
                "rows": [
                    [
                        str(row.get("dataset")),
                        str(row.get("question_count")),
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("avg_risk_count")),
                        format_float(row.get("avg_evidence_quality")),
                        format_float(row.get("expansion_calls_mean")),
                    ]
                    for row in router_eval.get("summary_rows", [])
                ],
            },
        },
        {
            "title": "输出协议诊断",
            "table": {
                "headers": ["方法/阶段", "协议失败率", "reason 缺失率"],
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
            ("CRED Stage A Protocol", str(manifest.get("cred_stage_a_output_protocol") or manifest.get("cred_output_protocol"))),
            ("CRED Verification Protocol", str(manifest.get("cred_verification_output_protocol") or manifest.get("cred_output_protocol"))),
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
        "oracle_accuracy_mean",
        "oracle_gap",
        "stage_candidate_oracle_accuracy",
        "candidate_pool_oracle_accuracy",
        "selection_loss",
        "expansion_oracle_accuracy",
        "expansion_oracle_gain",
        "target_precision_on_wrong_majority",
        "promotion_recall_on_wrong_majority",
        "selection_recall_on_pool_correct",
        "actual_gain",
        "verification_gain_over_initial_vote",
        "debate_gain_over_initial_vote",
        "accuracy_delta_vs_sc5",
        "token_ratio_vs_sc5",
        "base_vote_delta_vs_sc5",
        "trigger_rate",
        "expansion_trigger_rate",
        "adaptive_trigger_rate",
        "false_consensus_trigger_count",
        "false_consensus_recovered_count",
        "corrected_rate",
        "harmed_rate",
        "promotion_precision",
        "harm_per_correction",
        "safe_repair_count",
        "hetero_agreement_count",
        "math_repair_count",
        "hotpot_span_repair_count",
        "validator_pass_count",
        "choice_shuffle_agreement_count",
        "pairwise_duel_count",
        "pairwise_duel_win_count",
        "pairwise_duel_precision",
        "safe_selector_corrected_count",
        "safe_selector_harmed_count",
        "gpqa_unanimous_duel_count",
        "blocked_2of3_pairwise_count",
        "blocked_mmlu_pairwise_count",
        "blocked_strategyqa_probe_count",
        "method_expansion_call_count",
        "shadow_counterfactual_corrected_count",
        "shadow_counterfactual_harmed_count",
        "shadow_precision",
        "shadow_net_gain",
        "shadow_possible_gain",
        "shadow_gate_passed_count",
        "duel_invalid_count",
        "duel_retry_recoverable_count",
        "minority_probe_count",
        "minority_probe_precision",
        "non_answer_candidate_blocked_count",
        "single_pro_promotion_blocked_count",
        "strong_majority_locked_count",
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
