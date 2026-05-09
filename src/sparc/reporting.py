"""SPARC 报告与摘要。

报告层同时服务三类实验：内容消融、局部审计消融与 SPARC 主实验，
重点输出压缩比、触发策略收益、局部审计收益与整体成本-性能权衡。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from experiment_core.reporting.analysis_reports import (
    render_audit_diagnostic_report,
    render_frontier_report,
    write_report,
)
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_scatter_figure_spec,
    build_score_by_dataset_figure_spec,
    write_figure_bundle,
)
from experiment_core.foundation.workspace import default_reports_root


REPORT_NAME_BY_KIND = {
    "content_ablation": "content_ablation_report.md",
    "auditing_ablation": "auditing_ablation_report.md",
    "sparc_v1": "sparc_v1_report.md",
}


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = _load_json(Path(run_dir) / "metrics.json")
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["dataset"]].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("sparc")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics, diagnostics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="SPARC Frontier"))
    write_report(root / "audit_diagnostics.md", render_audit_diagnostic_report(metrics.get("summary", []), title="SPARC Audit Diagnostics"))
    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )
    return {
        "run_dir": str(root),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "frontier_report": str(root / "frontier_report.md"),
        "audit_diagnostic_report": str(root / "audit_diagnostics.md"),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("summary", [])
    overall_rows = [row for row in rows if row.get("dataset") == "overall"]
    experiment_kind = str(diagnostics.get("experiment_kind") or "")
    figure_specs = [
        build_frontier_figure_spec(
            rows,
            title="SPARC frontier",
            caption="Overall accuracy versus average total tokens across SPARC variants and controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="SPARC efficiency ranking",
            caption="Overall efficiency ranking measured by accuracy per 1K tokens.",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="Accuracy per 1K tokens",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="SPARC score by dataset",
            caption="Per-dataset accuracy map across SPARC variants and controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
        ),
    ]
    if experiment_kind == "content_ablation":
        figure_specs.append(
            build_scatter_figure_spec(
                figure_id="compression_tradeoff",
                title="Compression tradeoff",
                caption="Overall compression ratio versus accuracy across content-ablation variants.",
                primary_metric="Accuracy",
                data=[
                    {
                        "label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "short_label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "x": float(row.get("compression_ratio_vs_full_cot") or 0.0),
                        "y": float(row.get("accuracy_mean") or 0.0),
                        "value": float(row.get("accuracy_mean") or 0.0),
                    }
                    for row in overall_rows
                    if row.get("compression_ratio_vs_full_cot") is not None
                ],
                x_label="Compression ratio vs full CoT",
                y_label="Accuracy",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="Lower compression ratios indicate stronger communication compression relative to full CoT.",
                reference_x=1.0,
            )
        )
    else:
        figure_specs.append(
            build_scatter_figure_spec(
                figure_id="audit_gain_vs_cost",
                title="Audit gain versus cost",
                caption="Overall audit-token cost versus accuracy across auditing and end-to-end variants.",
                primary_metric="Accuracy",
                data=[
                    {
                        "label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "short_label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "x": float(row.get("audit_tokens_mean") or 0.0),
                        "y": float(row.get("accuracy_mean") or 0.0),
                        "value": float(row.get("accuracy_mean") or 0.0),
                    }
                    for row in overall_rows
                ],
                x_label="Average audit tokens per question",
                y_label="Accuracy",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="The left edge corresponds to variants that avoid explicit audit passes.",
                reference_x=0.0,
            )
        )
    if diagnostics.get("trigger_selection") is not None:
        figure_specs.append(
            build_grouped_bar_figure_spec(
                figure_id="trigger_selection_profile",
                title="Trigger selection profile",
                caption="Overall trigger and early-exit rates across SPARC variants that expose trigger behavior.",
                primary_metric="Rate",
                data=[
                    {
                        "label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "short_label": str(row.get("display_name") or row.get("method_name") or "unknown"),
                        "trigger_rate": float(row.get("trigger_rate") or 0.0),
                        "early_exit_rate": float(row.get("early_exit_rate") or 0.0),
                    }
                    for row in overall_rows
                ],
                series=[("trigger_rate", "Trigger rate"), ("early_exit_rate", "Early-exit rate")],
                x_label="Rate",
                source_kind="metrics.summary",
                dataset_scope="overall",
                note="Trigger and early-exit behavior is shown for variants that expose these fields in the SPARC summary.",
            )
        )
    return figure_specs


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    kind = str(manifest.get("experiment_kind"))
    if kind == "content_ablation":
        return _render_content_report(manifest, metrics, diagnostics, predictions, run_dir)
    if kind == "auditing_ablation":
        return _render_auditing_report(manifest, metrics, diagnostics, predictions, run_dir)
    return _render_sparc_report(manifest, metrics, diagnostics, predictions, run_dir)


def _render_content_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _rows_for_dataset(metrics, "overall")
    method_order = ["mv_3", "full_cot", "answer_only", "answer_confidence", "disagreement_step_only", "critical_evidence_only", "task_adaptive"]
    ordered_rows = _ordered_rows(overall_rows, method_order)
    disagreement_rows = _subset_summary(predictions, lambda row: bool(row.get("initial_disagreement")))
    oracle_rows = _subset_summary(predictions, lambda row: bool(row.get("oracle_positive")))
    recommendation = diagnostics.get("recommended_next_default")
    comparison_row = recommendation if recommendation else next((row for row in ordered_rows if row["method_name"] != "full_cot"), None)
    ci_text = _bootstrap_ci_text(predictions, comparison_row["method_name"], "full_cot") if comparison_row else "未计算。"
    lines = [
        "# SPARC 内容消融报告",
        "",
        "## 1. 实验范围与公平性说明",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{resolve_manifest_model_name(manifest)}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 本轮固定 `always_communicate`，仅比较消息内容，不混入 trigger 和 auditing 变量。",
        "- 数据集固定为 `GSM8K + StrategyQA + HotpotQA` 的 `smoke20_seed42`。",
        "",
        "## 2. 主结果表",
        "",
        "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Compression vs Full CoT |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in ordered_rows:
        compression = row.get("compression_ratio_vs_full_cot")
        compression_text = f"{compression:.4f}" if compression is not None else "-"
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | {row['acc_per_1k_tokens']:.6f} | {compression_text} |"
        )

    lines.extend(
        [
            "",
            "## 3. 条件子集",
            "",
            "### initial_disagreement = true",
            "",
            "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in _ordered_rows(disagreement_rows, method_order):
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "### oracle_positive = true",
            "",
            "| Method | Accuracy | Avg Comm Tokens | Avg Total Tokens |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for row in _ordered_rows(oracle_rows, method_order):
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## 4. 探索性区间",
            "",
            f"- 相对 `full_cot` 的 overall accuracy delta 95% bootstrap CI：{ci_text}",
            "",
        ]
    )
    lines.extend(_failure_case_section(predictions, primary_method=comparison_row["method_name"] if comparison_row else "full_cot", reference_method="full_cot"))
    lines.extend(
        [
            "## 6. 下一轮建议",
            "",
            f"- 推荐默认消息模式：`{recommendation['method_name']}`" if recommendation else "- 当前未生成推荐模式。",
            f"- 推荐依据：overall accuracy=`{recommendation['accuracy_mean']:.4f}`，total tokens=`{recommendation['total_tokens_mean']:.2f}`。" if recommendation else "",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None) + "\n"


def _render_auditing_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _ordered_rows(
        _rows_for_dataset(metrics, "overall"),
        ["majority_vote", "weighted_vote_fallback", "single_judge", "final_round_vote", "local_auditing"],
    )
    recommendation = diagnostics.get("recommended_next_default")
    ci_text = _bootstrap_ci_text(predictions, "local_auditing", "final_round_vote")
    lines = [
        "# SPARC 审计消融报告",
        "",
        "## 1. 实验范围与公平性说明",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{resolve_manifest_model_name(manifest)}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "- 本轮固定消息内容，只比较最终聚合与局部审计方式。",
        "",
        "## 2. 主结果表",
        "",
        "| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Resolve Rate | Abstain Rate | Wrong Overrule Rate | Minority Rescue Count |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['audit_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} | {row['resolve_rate']:.4f} | "
            f"{row['abstain_rate']:.4f} | {row['wrong_overrule_rate']:.4f} | {row['minority_rescue_count']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 探索性区间",
            "",
            f"- `local_auditing` 相对 `final_round_vote` 的 overall accuracy delta 95% bootstrap CI：{ci_text}",
            "",
        ]
    )
    lines.extend(_failure_case_section(predictions, primary_method="local_auditing", reference_method="final_round_vote"))
    lines.extend(
        [
            "## 5. 下一轮建议",
            "",
            f"- 推荐默认聚合方式：`{recommendation['method_name']}`" if recommendation else "- 当前未生成推荐聚合方式。",
            f"- 推荐依据：overall accuracy=`{recommendation['accuracy_mean']:.4f}`，total tokens=`{recommendation['total_tokens_mean']:.2f}`。" if recommendation else "",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None) + "\n"


def _render_sparc_report(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    overall_rows = _ordered_rows(
        _rows_for_dataset(metrics, "overall"),
        ["mv_3", "always_communicate", "hybrid_trigger_baseline", "final_round_vote_baseline", "sparc_v1"],
    )
    trigger_selection = diagnostics.get("trigger_selection", {})
    recommendation = diagnostics.get("recommended_next_default")
    ci_text = _bootstrap_ci_text(predictions, "sparc_v1", "final_round_vote_baseline")
    lines = [
        "# SPARC v1 Smoke 报告",
        "",
        "## 1. 实验范围与公平性说明",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Backbone：`{resolve_manifest_model_name(manifest)}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        f"- 选中的 trigger 策略：`{trigger_selection.get('selected_policy')}`",
        f"- trigger 选择原因：`{trigger_selection.get('reason')}`",
        f"- trigger 参考运行：`{trigger_selection.get('reference_run_dir')}`" if trigger_selection.get("reference_run_dir") else "- 未找到 trigger 参考运行，使用默认策略。",
        "",
        "## 2. 主结果表",
        "",
        "| Method | Accuracy | Avg Comm Tokens | Avg Audit Tokens | Avg Total Tokens | Calls / Q | Acc / 1K Tokens | Trigger Rate | Early Exit Rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in overall_rows:
        lines.append(
            f"| `{row['display_name']}` | {row['accuracy_mean']:.4f} | {row['communication_tokens_mean']:.2f} | "
            f"{row['audit_tokens_mean']:.2f} | {row['total_tokens_mean']:.2f} | {row['calls_per_question_mean']:.2f} | "
            f"{row['acc_per_1k_tokens']:.6f} | {row['trigger_rate']:.4f} | {row['early_exit_rate']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 3. 探索性区间",
            "",
            f"- `sparc_v1` 相对 `final_round_vote_baseline` 的 overall accuracy delta 95% bootstrap CI：{ci_text}",
            "",
        ]
    )
    lines.extend(_failure_case_section(predictions, primary_method="sparc_v1", reference_method="final_round_vote_baseline"))
    lines.extend(
        [
            "## 5. 下一轮建议",
            "",
            f"- 当前最佳 overall 方法：`{recommendation['method_name']}`" if recommendation else "- 当前未生成下一轮建议。",
            f"- trigger 选择记录：drop_questions=`{trigger_selection.get('drop_questions')}`，threshold=`{trigger_selection.get('threshold_questions')}`。" if trigger_selection else "",
            "",
        ]
    )
    return "\n".join(line for line in lines if line is not None) + "\n"


def _failure_case_section(
    predictions: list[dict[str, Any]],
    *,
    primary_method: str,
    reference_method: str,
) -> list[str]:
    lines = ["## 5. 失败案例", ""]
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    cases: list[dict[str, Any]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary is None or reference is None:
            continue
        if float(primary["score"]) < float(reference["score"]):
            cases.append(
                {
                    "dataset": dataset,
                    "sample_id": sample_id,
                    "question_preview": primary["question_preview"],
                    "gold": primary["gold"],
                    "primary_prediction": primary["prediction"],
                    "primary_score": primary["score"],
                    "reference_prediction": reference["prediction"],
                    "reference_score": reference["score"],
                    "reason": primary.get("note") or "主方法在该题上弱于参考方法。",
                }
            )
        if len(cases) >= 5:
            break
    if not cases:
        lines.append("- 当前 smoke20 下没有稳定失败案例。")
        lines.append("")
        return lines
    for index, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"### Case {index}",
                "",
                f"- 数据集：`{case['dataset']}`",
                f"- 样本：`{case['sample_id']}`",
                f"- 问题预览：{case['question_preview']}",
                f"- 金标：`{case['gold']}`",
                f"- 主方法：`{case['primary_prediction']}` / score=`{case['primary_score']}`",
                f"- 参考方法：`{case['reference_prediction']}` / score=`{case['reference_score']}`",
                f"- 说明：{case['reason']}",
                "",
            ]
        )
    return lines


def _subset_summary(
    predictions: list[dict[str, Any]],
    predicate,
) -> list[dict[str, Any]]:
    rows = [row for row in predictions if predicate(row)]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["model_name"], row["method_name"]), []).append(row)
    summary = []
    for (model_name, method_name), items in grouped.items():
        total_tokens_mean = _mean(float(item["total_tokens_per_question"]) for item in items)
        summary.append(
            {
                "dataset": "subset",
                "model_name": model_name,
                "method_name": method_name,
                "display_name": items[0]["display_name"],
                "accuracy_mean": _mean(float(item["score"]) for item in items),
                "communication_tokens_mean": _mean(float(item["communication_tokens_per_question"]) for item in items),
                "total_tokens_mean": total_tokens_mean,
            }
        )
    return summary


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算。"
    deltas = _bootstrap_accuracy_delta(paired, iterations=2000, seed=42)
    return f"[{_quantile(deltas, 0.025):.6f}, {_quantile(deltas, 0.975):.6f}]（探索性）"


def _paired_rows(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> list[tuple[float, float]]:
    lookup = {
        (row["dataset"], row["sample_id"], row["method_name"]): row
        for row in predictions
    }
    paired: list[tuple[float, float]] = []
    sample_keys = sorted({(row["dataset"], row["sample_id"]) for row in predictions})
    for dataset, sample_id in sample_keys:
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary is None or reference is None:
            continue
        paired.append((float(primary["score"]), float(reference["score"])))
    return paired


def _bootstrap_accuracy_delta(
    paired_scores: list[tuple[float, float]],
    *,
    iterations: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    rows = list(paired_scores)
    samples: list[float] = []
    for _ in range(iterations):
        picked = [rows[rng.randrange(len(rows))] for _ in range(len(rows))]
        primary_acc = sum(primary for primary, _ in picked) / len(picked)
        reference_acc = sum(reference for _, reference in picked) / len(picked)
        samples.append(round(primary_acc - reference_acc, 6))
    return samples


def _rows_for_dataset(metrics: dict[str, Any], dataset: str) -> list[dict[str, Any]]:
    return [row for row in metrics.get("summary", []) if row.get("dataset") == dataset]


def _ordered_rows(rows: list[dict[str, Any]], order: list[str]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: order.index(row["method_name"]) if row["method_name"] in order else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "sparc")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = resolve_manifest_model_name(manifest).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)

