"""A-SMAD 报告渲染入口。"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from research_experiments.families.adaptive_sparse_mad.config import ADAPTIVE_POLICY_METHODS
from research_experiments.family_runtime.artifact_index import (
    load_metrics_payload,
    named_diagnostic_paths,
    resolve_run_artifact_index,
)
from research_experiments.family_runtime.report_bundle import render_family_report_bundle
from research_experiments.reporting.report_views import DiagnosticTableView, SummaryTableView, load_json_payload
from research_experiments.reporting.reporting_utils import resolve_manifest_model_name
from research_experiments.reporting.run_figures import (
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_score_by_dataset_figure_spec,
)
from research_experiments.reporting.scientific_report import format_float
from research_experiments.workspace.layout import default_reports_root

METHOD_ORDER = [
    "cot_1",
    "mv_3",
    "sc_5",
    "hetero_vote_3",
    "ega_only_v4",
    "adaptive_gate_v4",
    "adaptive_dual_open_v5",
    "adaptive_counterfactual_v1",
]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    summary = SummaryTableView.from_metrics_payload(load_metrics_payload(run_dir, family_name="adaptive_sparse_mad"))
    grouped = summary.grouped_by_dataset()
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(summary.rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": {dataset: [row.raw for row in rows] for dataset, rows in grouped.items()},
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("adaptive_sparse_mad")
    index = resolve_run_artifact_index(run_dir, family_name="adaptive_sparse_mad")
    manifest = load_json_payload(index.manifest_path)
    metrics = load_json_payload(index.metrics_view_path)
    diagnostics = named_diagnostic_paths(index.run_dir, family_name="adaptive_sparse_mad")
    router_eval = load_json_payload(_diagnostic_path(diagnostics, index.run_dir, "router_eval.json"))
    policy_diagnostics = load_json_payload(_diagnostic_path(diagnostics, index.run_dir, "policy_diagnostics.json"))
    stage_a_resolver_breakdown = load_json_payload(
        _diagnostic_path(diagnostics, index.run_dir, "stage_a_resolver_breakdown.json")
    )
    stage_a_error_buckets = load_json_payload(_diagnostic_path(diagnostics, index.run_dir, "stage_a_error_buckets.json"))
    stage_a_solver_contributions = load_json_payload(
        _diagnostic_path(diagnostics, index.run_dir, "stage_a_solver_contributions.json")
    )
    base_markdown = _render_markdown(
        manifest,
        metrics,
        router_eval,
        policy_diagnostics,
        stage_a_resolver_breakdown,
        stage_a_error_buckets,
        stage_a_solver_contributions,
        index.run_dir,
    )
    return render_family_report_bundle(
        family_name="adaptive_sparse_mad",
        run_dir=index.run_dir,
        publish_dir=publish_dir,
        manifest=manifest,
        base_markdown=base_markdown,
        figure_specs=_build_figure_specs(metrics, policy_diagnostics),
    )


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    del diagnostics
    summary = SummaryTableView.from_metrics_payload(metrics)
    summary_rows = [row.raw for row in summary.rows]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="A-SMAD 成本-性能前沿",
            caption="总体结果上，各方法的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="A-SMAD 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="A-SMAD 跨数据集表现",
            caption="各方法在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
    ]

def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    router_eval: dict[str, Any],
    policy_diagnostics: dict[str, Any],
    stage_a_resolver_breakdown: dict[str, Any],
    stage_a_error_buckets: dict[str, Any],
    stage_a_solver_contributions: dict[str, Any],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    summary = SummaryTableView.from_metrics_payload(metrics)
    grouped_by_dataset = {
        dataset: _ordered_rows(summary.dataset_rows(dataset))
        for dataset in summary.dataset_names()
    }
    overall_rows = _ordered_rows(summary.overall_rows())
    policy_rows = DiagnosticTableView.from_rows(policy_diagnostics.get("policy_rows", []))
    overall_policy_rows = [row.raw for row in policy_rows.overall_rows()]
    pairwise_rows = [
        row
        for row in policy_diagnostics.get("pairwise_rows", [])
    ]
    promotion_gate = policy_diagnostics.get("promotion_gate", {})
    promotion_gate_rows = list(promotion_gate.get("candidate_rows", []))
    mainline_gate = policy_diagnostics.get("mainline_gate", {})
    mainline_gate_rows = list(mainline_gate.get("candidate_rows", []))
    best_row = summary.best_by("accuracy_mean", rows=overall_rows)
    recommended = policy_diagnostics.get("recommended_next_default_policy", {})
    resolver_breakdown_rows = stage_a_resolver_breakdown.get("summary_rows", [])
    overall_resolver_rows = [row for row in resolver_breakdown_rows if row.get("dataset") == "overall"]
    resolver_examples = stage_a_resolver_breakdown.get("example_rows", [])
    error_summary = stage_a_error_buckets.get("summary", {})
    error_dataset_rows = stage_a_error_buckets.get("dataset_rows", [])
    error_examples = stage_a_error_buckets.get("example_rows", [])
    contribution_rows = stage_a_solver_contributions.get("summary_rows", [])
    overall_contribution = next((row for row in contribution_rows if row.get("dataset") == "overall"), {})

    abstract = []
    if best_row is not None:
        abstract.append(
            f"总体准确率最高的方法是 `{best_row.display_name}`，准确率为 {format_float(best_row.accuracy_mean)}。"
        )
    abstract.append(f"当前建议继续默认跟踪的方法是 `{recommended.get('selected_policy', 'hetero_vote_3')}`。")

    lines = [
        "# A-SMAD same-context 科研报告",
        "",
        "## 摘要",
        "",
    ]
    lines.extend(f"- {line}" for line in abstract)
    lines.extend(
        [
            "",
            "## 实验概览",
            "",
            f"- 实验名：`{manifest.get('experiment_name', manifest.get('experiment', 'same_context_main'))}`",
            f"- Phase：`{manifest.get('phase_name', manifest.get('phase', 'unknown'))}`",
            f"- Backbone：`{backbone_name}`",
            f"- Prompt Version：`{manifest.get('prompt_version', 'adaptive_sparse_mad_v2_task_schema')}`",
            f"- Stage A Prompt：`{manifest.get('stage_a_prompt_version', manifest.get('prompt_version', 'adaptive_sparse_mad_v2_task_schema'))}`",
            f"- Adaptive Prompt：`{manifest.get('adaptive_prompt_version', manifest.get('prompt_version', 'adaptive_sparse_mad_v2_task_schema'))}`",
            f"- 运行目录：`{run_dir.as_posix()}`",
            "",
            "## 研究问题与实验设计",
            "",
            "- 当前快速默认实验只比较 `hetero_vote_3` 与 `cot_1 / mv_3 / sc_5`，先确认异质 Stage A 是否能稳定打败强 no-comm 基线。",
            "- Stage A 固定为 `solver_cot / solver_l2m / solver_skeptic` 三个异质 solver，Stage B / judge 默认不进入对比。",
            "- 当前主线重点是先把 Stage A 做强，再根据误差分桶结果决定是否值得引入更强聚合或轻量仲裁。",
            "",
            "## 总体结果",
            "",
            "| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in overall_rows:
        lines.append(
            f"| `{row.display_name}` | {format_float(row.accuracy_mean)} | {format_float(row.communication_tokens_mean)} | "
            f"{format_float(row.total_tokens_mean)} | {format_float(row.calls_per_question_mean)} | {format_float(row.acc_per_1k_tokens)} |"
        )

    lines.extend(
        [
            "",
            "## Stage A Error Buckets",
            "",
            f"- `error_count`: `{int(error_summary.get('error_count', 0) or 0)}`",
            f"- `all_three_wrong`: `{int(error_summary.get('all_three_wrong', 0) or 0)}`",
            f"- `clean_pseudo_majority`: `{int(error_summary.get('clean_pseudo_majority', 0) or 0)}`",
            f"- `confidence_miscalibration`: `{int(error_summary.get('confidence_miscalibration', 0) or 0)}`",
            f"- `constraint_mismatch`: `{int(error_summary.get('constraint_mismatch', 0) or 0)}`",
            "",
            "| Dataset | Error Count | all_three_wrong | clean_pseudo_majority | confidence_miscalibration | constraint_mismatch |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in error_dataset_rows:
        lines.append(
            f"| `{row.get('dataset', 'unknown')}` | {int(row.get('error_count', 0) or 0)} | "
            f"{int(row.get('all_three_wrong', 0) or 0)} | {int(row.get('clean_pseudo_majority', 0) or 0)} | "
            f"{int(row.get('confidence_miscalibration', 0) or 0)} | {int(row.get('constraint_mismatch', 0) or 0)} |"
        )
    if error_examples:
        lines.extend(
            [
                "",
                "| Example Sample | Bucket | Predicted Answer | Correct Candidate In Stage A |",
                "| --- | --- | --- | --- |",
            ]
        )
        for row in error_examples[:8]:
            lines.append(
                f"| `{row.get('dataset', 'unknown')}:{row.get('sample_id', '')}` | `{row.get('bucket', 'unknown')}` | "
                f"`{row.get('predicted_answer', 'unknown')}` | `{str(bool(row.get('correct_in_stage_a'))).lower()}` |"
            )

    if overall_contribution:
        lines.extend(
            [
                "",
                "## Stage A Solver Contributions",
                "",
                "| Solver | any_correct | solo_correct | majority_wrong_but_solver_right |",
                "| --- | --- | --- | --- |",
            ]
        )
        for solver_name in ("solver_cot", "solver_l2m", "solver_skeptic"):
            lines.append(
                f"| `{solver_name}` | {int(overall_contribution.get(f'any_correct_{solver_name}', 0) or 0)} | "
                f"{int(overall_contribution.get(f'solo_correct_{solver_name}', 0) or 0)} | "
                f"{int(overall_contribution.get(f'majority_wrong_but_solver_right_{solver_name}', 0) or 0)} |"
            )

    if overall_resolver_rows:
        lines.extend(
            [
                "",
                "## Stage A Resolver Breakdown",
                "",
                "| Resolver | Total | Correct | Wrong | Accuracy |",
                "| --- | --- | --- | --- | --- |",
            ]
        )
        for row in overall_resolver_rows:
            lines.append(
                f"| `{row.get('resolver', 'unknown')}` | {int(row.get('total', 0) or 0)} | "
                f"{int(row.get('correct', 0) or 0)} | {int(row.get('wrong', 0) or 0)} | "
                f"{format_float(float(row.get('accuracy_mean', 0.0) or 0.0))} |"
            )
        if resolver_examples:
            lines.extend(
                [
                    "",
                    "| Resolver Example | Prediction | Correct Answers | Score |",
                    "| --- | --- | --- | --- |",
                ]
            )
            for row in resolver_examples[:8]:
                lines.append(
                    f"| `{row.get('resolver', 'unknown')}:{row.get('dataset', 'unknown')}:{row.get('sample_id', '')}` | "
                    f"`{row.get('prediction', 'unknown')}` | `{','.join(row.get('correct_answers', []))}` | "
                    f"{format_float(float(row.get('score', 0.0) or 0.0))} |"
                )

    router_title = "## Stage A-only 诊断"
    router_note = "- 当前 A-SMAD 主线不再运行 pair critique、judge 或其他 Stage B 通信策略。"
    if any(str(row.raw.get("method_name") or "") in ADAPTIVE_POLICY_METHODS for row in overall_rows):
        router_title = "## Adaptive Policy 诊断"
        router_note = "- 当前主线使用 Stage A 后的轻量触发式验证，不恢复 pair critique 或 judge。"
    lines.extend(
        [
            "",
            router_title,
            "",
            router_note,
            f"- `router_eval.summary_rows`: `{len(router_eval.get('summary_rows', []))}`",
            f"- `policy_diagnostics.policy_rows`: `{len(overall_policy_rows)}`",
        ]
    )
    interesting_pairwise_rows = [
        row
        for row in pairwise_rows
        if str(row.get("method_name") or "") in {"ega_only_v4", "adaptive_gate_v4", "adaptive_dual_open_v5", "adaptive_counterfactual_v1"}
        and str(row.get("baseline_method_name") or "") in {"hetero_vote_3", "sc_5"}
    ]
    if interesting_pairwise_rows:
        lines.extend(
            [
                "",
                "## Paired Comparison",
                "",
                "| Dataset | Method | Baseline | Delta Acc | 95% CI | Corrected | Harmed | McNemar p | Holm p |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in interesting_pairwise_rows:
            lines.append(
                f"| `{row.get('dataset', 'unknown')}` | `{row.get('method_name', 'unknown')}` | `{row.get('baseline_method_name', 'unknown')}` | "
                f"{format_float(float(row.get('accuracy_delta', 0.0) or 0.0))} | "
                f"`[{format_float(float(row.get('bootstrap_ci_low', 0.0) or 0.0))}, {format_float(float(row.get('bootstrap_ci_high', 0.0) or 0.0))}]` | "
                f"{int(row.get('corrected_count', 0) or 0)} | {int(row.get('harmed_count', 0) or 0)} | "
                f"{format_float(float(row.get('exact_mcnemar_p', 1.0) or 1.0))} | "
                f"{format_float(float(row.get('holm_adjusted_p', 1.0) or 1.0))} |"
            )

    if promotion_gate_rows:
        lines.extend(
            [
                "",
                "## Promotion Gate",
                "",
                "| Method | Promote To Count100 | Mainline Ready | Net Corrected | Positive Datasets | Negative Datasets | Verdict |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in promotion_gate_rows:
            lines.append(
                f"| `{row.get('method_name', 'unknown')}` | `{str(bool(row.get('promote_to_count100'))).lower()}` | "
                f"`{str(bool(row.get('mainline_ready_signal'))).lower()}` | "
                f"{int(row.get('net_corrected', 0) or 0)} | "
                f"`{','.join(row.get('positive_datasets', [])) or 'none'}` | "
                f"`{','.join(row.get('negative_datasets', [])) or 'none'}` | "
                f"`{row.get('verdict_reason', 'unknown')}` |"
            )

    if mainline_gate_rows:
        lines.extend(
            [
                "",
                "## Mainline Gate",
                "",
                "| Method | Eligible | Mainline Ready | Delta Acc | 95% CI | Holm p | Core Categories | Negative Datasets | Verdict |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in mainline_gate_rows:
            lines.append(
                f"| `{row.get('method_name', 'unknown')}` | "
                f"`{str(bool(row.get('eligible_for_mainline_assessment'))).lower()}` | "
                f"`{str(bool(row.get('mainline_ready'))).lower()}` | "
                f"{format_float(float(row.get('overall_accuracy_delta', 0.0) or 0.0))} | "
                f"`[{format_float(float(row.get('bootstrap_ci_low', 0.0) or 0.0))}, {format_float(float(row.get('bootstrap_ci_high', 0.0) or 0.0))}]` | "
                f"{format_float(float(row.get('holm_adjusted_p', 1.0) or 1.0))} | "
                f"{int(row.get('core_category_positive_count', 0) or 0)} | "
                f"`{','.join(row.get('negative_datasets', [])) or 'none'}` | "
                f"`{row.get('verdict_reason', 'unknown')}` |"
            )

    lines.extend(["", "## 分数据集表现", ""])
    for dataset, rows in grouped_by_dataset.items():
        lines.extend(
            [
                f"### {dataset}",
                "",
                "| 方法 | 准确率 | 平均通信 token / 题 | 平均总 token / 题 | 每题调用数 | 每千 token 准确率 |",
                "| --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in rows:
            lines.append(
                f"| `{row.display_name}` | {format_float(row.accuracy_mean)} | {format_float(row.communication_tokens_mean)} | "
                f"{format_float(row.total_tokens_mean)} | {format_float(row.calls_per_question_mean)} | {format_float(row.acc_per_1k_tokens)} |"
            )
        lines.append("")

    lines.extend(
        [
            "## 结论与建议",
            "",
            "- 如果 `hetero_vote_3` 稳定强于 `cot_1 / mv_3 / sc_5`，说明异质 Stage A 仍然是当前最值得继续加码的主线。",
            "- 如果 `adaptive_gate_v4` 或 `adaptive_dual_open_v5` 在 `all_three_wrong` 高发的数据集上仍能保持净正改正，就说明分歧触发 + 证据重排仍是值得继续扩展的机制主线。",
            "- `Promotion Gate` 面向 `count20 -> count100` 的晋级决策；只有跨多个数据集保持净正收益的方法才值得继续上大样本验证。",
            "- 如果误差主要集中在 `all_three_wrong`，优先改 solver 本身；如果主要集中在 `clean_pseudo_majority`，再优先改聚合。",
            "- 只有在 Stage A 已经足够强时，才值得恢复更昂贵的通信或 judge 消融。",
            "",
            "## 复现与产物说明",
            "",
            f"- 运行目录：`{run_dir.as_posix()}`",
            "- 关键产物：`views/metrics.json`、`views/predictions.jsonl`、`diagnostics/router_eval.json`、`diagnostics/policy_diagnostics.json`、`diagnostics/stage_a_error_buckets.json`、`diagnostics/stage_a_solver_contributions.json`、`report.md`、`figure_manifest.json`、`figures/`。",
            "- 本地报告与图表共用同一套 run 内数据源，便于后续复核和 rerender。",
        ]
    )
    return "\n".join(lines).strip() + "\n"


def _ordered_rows(rows: list[Any]) -> list[Any]:
    order = {name: idx for idx, name in enumerate(METHOD_ORDER)}
    return sorted(rows, key=lambda row: order.get(row.method_name, 999))


def _diagnostic_path(
    diagnostics: dict[str, Path],
    run_dir: Path,
    filename: str,
) -> Path:
    return diagnostics.get(filename, run_dir / "diagnostics" / filename)
