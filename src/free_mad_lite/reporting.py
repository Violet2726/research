"""Free-MAD-lite 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from free_mad_lite.logic import METHOD_ORDER
from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.analysis_reports import render_frontier_report, write_report
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_score_by_dataset_figure_spec,
    write_figure_bundle,
)
from experiment_core.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = _load_json(Path(run_dir) / "metrics.json")
    rows = metrics.get("summary", [])
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset"))].append(row)
    return {
        "run_dir": str(Path(run_dir)),
        "row_count": len(rows),
        "datasets": sorted(grouped),
        "summary_by_dataset": grouped,
    }


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("free_mad_lite")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")
    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report = root / "report.md"
    local_report.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="Free-MAD-lite 前沿附录"))
    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )
    return {
        "run_dir": str(root),
        "local_report": str(local_report),
        "published_report": str(publish_path),
        "frontier_report": str(root / "frontier_report.md"),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("summary", [])
    overall_rows = [row for row in rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            rows,
            title="Free-MAD-lite 成本-性能前沿",
            caption="总体结果上，各 Free-MAD-lite 变体的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="Free-MAD-lite 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="Free-MAD-lite 跨数据集表现",
            caption="各 Free-MAD-lite 变体在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="trajectory_score_panel",
            title="轨迹裁决效果",
            caption="总体层面 changed / corrected / harmed 三类比例对比。",
            primary_metric="比例",
            data=[
                {
                    "label": str(row.get("method_name") or "unknown"),
                    "short_label": str(row.get("method_name") or "unknown"),
                    "changed_answer_rate": float(row.get("changed_answer_rate") or 0.0),
                    "corrected_rate": (float(row.get("corrected_count") or 0.0) / float(row.get("question_count") or 1.0)),
                    "harmed_rate": (float(row.get("harmed_count") or 0.0) / float(row.get("question_count") or 1.0)),
                }
                for row in overall_rows
            ],
            series=[
                ("changed_answer_rate", "Changed answer"),
                ("corrected_rate", "Corrected rate"),
                ("harmed_rate", "Harmed rate"),
            ],
            x_label="比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="纠正和伤害都被标准化为题级比例，以便和答案变化率并列比较。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="judge_fallback_summary",
            title="Judge fallback 概览",
            caption="总体层面 judge fallback rate 对比。",
            primary_metric="Fallback rate",
            data=[
                {
                    "label": str(row.get("method_name") or "unknown"),
                    "short_label": str(row.get("method_name") or "unknown"),
                    "judge_fallback_rate": float(row.get("judge_fallback_rate") or 0.0),
                }
                for row in overall_rows
            ],
            series=[("judge_fallback_rate", "Judge fallback rate")],
            x_label="比例",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="该比例越低，说明轨迹裁决器越少需要退回到基础投票结果。",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    best_row = max(overall_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), default=None)
    best_efficiency_row = max(overall_rows, key=lambda item: float(item.get("acc_per_1k_tokens") or 0.0), default=None)
    ci_text = _bootstrap_ci_text(predictions, "free_mad_lite_llm_trajectory", "vanilla_mad_r1_final_vote")

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row['method_name']}`，准确率为 {format_float(best_row.get('accuracy_mean'))}。")
    if best_efficiency_row is not None:
        abstract.append(f"总体效率最高的方法是 `{best_efficiency_row['method_name']}`，每千 token 准确率为 {format_float(best_efficiency_row.get('acc_per_1k_tokens'), 6)}。")
    abstract.append(f"`free_mad_lite_llm_trajectory` 相对 `vanilla_mad_r1_final_vote` 的总体准确率差异 bootstrap 95% CI 为 `{ci_text}`。")

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "Free-MAD-lite 关注单轮 anti-conformity 与 LLM trajectory judge 是否足以带来稳定收益，而不复现完整 score-model 训练流程。",
                "主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；机制指标重点是 changed answer、corrected、harmed 与 judge fallback rate。",
                "本实验固定比较 `mv_3_initial`、`vanilla_mad_r1_final_vote`、`anti_conformity_final_vote` 和 `free_mad_lite_llm_trajectory`，因此可以隔离轨迹裁决环节的贡献。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率", "Judge fallback rate", "Changed answer rate"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("accuracy_mean")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("acc_per_1k_tokens"), 6),
                        format_float(row.get("judge_fallback_rate")),
                        format_float(row.get("changed_answer_rate")),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "机制诊断",
            "bullets": [
                f"Judge fallback rate：`{diagnostics.get('judge_fallback_rate', 0.0)}`；Judge fallback count：`{diagnostics.get('judge_fallback_count', 0)}`。",
                f"Anti-conformity prompt hash：`{diagnostics.get('anti_conformity_prompt_hash', manifest.get('anti_conformity_prompt_hash', 'unknown'))}`。",
                "如果 changed answer rate 很高，但 corrected rate 没有同步提高，说明轨迹裁决更像是在频繁改写答案，而不是真正识别正确轨迹。",
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
                            f"`{row['method_name']}`",
                            format_float(row.get("accuracy_mean")),
                            format_float(row.get("communication_tokens_mean"), 2),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("acc_per_1k_tokens"), 6),
                        ]
                        for row in _ordered_rows([item for item in metrics.get("summary", []) if item.get("dataset") == dataset])
                    ],
                }
                for dataset in sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"})
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "若 `free_mad_lite_llm_trajectory` 在总体准确率和每千 token 准确率上都优于 `vanilla_mad_r1_final_vote`，说明轨迹裁决在当前设置下具有独立价值。",
                "若 judge fallback rate 偏高，应优先增强 judge 的稳定性，再考虑扩大 anti-conformity 的使用范围。",
                "进入更大样本 phase 前，建议同时核对轨迹裁决效果图和 fallback 图，避免只看总体准确率结论。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前实现只验证单轮 anti-conformity 与 LLM trajectory judge，不包含论文中的完整 score-based 决策训练流程。",
                "本报告反映的是当前 phase 的机制验证结果，不直接等同于更大样本上的最终结论。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`trajectory_scores.jsonl`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="Free-MAD-lite 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row.get("method_name") in METHOD_ORDER else 999)


def _bootstrap_ci_text(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> str:
    paired = _paired_rows(predictions, primary_method, reference_method)
    if not paired:
        return "未计算"
    rng = random.Random(42)
    deltas: list[float] = []
    for _ in range(2000):
        picked = [paired[rng.randrange(len(paired))] for _ in range(len(paired))]
        deltas.append(round(sum(a - b for a, b in picked) / len(picked), 6))
    return f"[{_quantile(deltas, 0.025):.6f}, {_quantile(deltas, 0.975):.6f}]"


def _paired_rows(predictions: list[dict[str, Any]], primary_method: str, reference_method: str) -> list[tuple[float, float]]:
    lookup = {(row["dataset"], row["sample_id"], row["method_name"]): row for row in predictions}
    paired: list[tuple[float, float]] = []
    for dataset, sample_id in sorted({(row["dataset"], row["sample_id"]) for row in predictions}):
        primary = lookup.get((dataset, sample_id, primary_method))
        reference = lookup.get((dataset, sample_id, reference_method))
        if primary and reference:
            paired.append((float(primary["score"]), float(reference["score"])))
    return paired


def _quantile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    weight = position - lower
    return round(ordered[lower] * (1 - weight) + ordered[upper] * weight, 6)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "free-mad-lite")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]
