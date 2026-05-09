"""`comm_necessary` 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from comm_necessary.logic import METHOD_ORDER
from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.analysis_reports import render_split_context_report, write_report
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


def render_report(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("comm_necessary")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "metrics.json")
    diagnostics = _load_json(root / "diagnostics.json")
    predictions = _load_jsonl(root / "final_predictions.jsonl")

    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics, diagnostics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, predictions, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report = root / "report.md"
    local_report.write_text(markdown, encoding="utf-8")

    write_report(
        root / "split_context_report.md",
        render_split_context_report(
            metrics.get("summary", []),
            title="Split-Context 联合指标附录",
        ),
    )

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )

    summary_path = Path(publish_dir) / "HotpotQA通信必要性最近结果汇总.md"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=summary_path),
        encoding="utf-8",
    )

    return {
        "run_dir": str(root),
        "local_report": str(local_report),
        "published_report": str(publish_path),
        "summary_report": str(summary_path),
        "split_context_report": str(root / "split_context_report.md"),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    rows = metrics.get("summary", [])
    overall_rows = [row for row in rows if row.get("dataset") == "overall"]
    return [
        build_frontier_figure_spec(
            rows,
            title="通信必要性成本-性能前沿",
            caption="总体 Joint F1 相对于平均总 token 的位置关系。",
            score_field="joint_f1_mean",
            primary_metric="Joint F1",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            rows,
            title="通信必要性效率排序",
            caption="基于每千 token 得分的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 得分",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            rows,
            title="通信必要性跨数据集表现",
            caption="各 split-context 方法在不同数据集上的 Joint F1 分布。",
            score_field="joint_f1_mean",
            primary_metric="Joint F1",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="joint_metric_panel",
            title="联合指标剖面",
            caption="总体层面 Answer F1、Supporting F1 和 Joint F1 的并列对比。",
            primary_metric="F1",
            data=[
                {
                    "label": str(row.get("method_name") or "unknown"),
                    "short_label": str(row.get("method_name") or "unknown"),
                    "answer_f1_mean": float(row.get("answer_f1_mean") or 0.0),
                    "supporting_f1_mean": float(row.get("supporting_f1_mean") or 0.0),
                    "joint_f1_mean": float(row.get("joint_f1_mean") or 0.0),
                }
                for row in overall_rows
            ],
            series=[
                ("answer_f1_mean", "Answer F1"),
                ("supporting_f1_mean", "Supporting F1"),
                ("joint_f1_mean", "Joint F1"),
            ],
            x_label="F1",
            source_kind="metrics.summary",
            dataset_scope="overall",
            note="把答案质量、证据质量和联合指标并列展示，便于判断通信收益来自哪一部分。",
        ),
        build_grouped_bar_figure_spec(
            figure_id="delta_vs_controls",
            title="关键对照差值",
            caption="诊断文件中记录的关键控制组差值，重点关注 Joint F1 变化。",
            primary_metric="Joint F1 差值",
            data=[
                {
                    "label": str(row.get("comparison") or "unknown"),
                    "short_label": str(row.get("comparison") or "unknown")[:24],
                    "joint_f1_delta": float(row.get("joint_f1_delta") or 0.0),
                }
                for row in diagnostics.get("key_deltas", [])
            ],
            series=[("joint_f1_delta", "Joint F1 差值")],
            x_label="差值",
            source_kind="diagnostics",
            dataset_scope="overall",
            note="正值表示相对于对照方法有提升，负值表示通信设计带来退化。",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    predictions: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    phase = str(manifest.get("phase") or "unknown_phase")
    backbone_name = resolve_manifest_model_name(manifest)
    overall_rows = _ordered_rows([row for row in metrics.get("summary", []) if row.get("dataset") == "overall"])
    best_joint_row = max(overall_rows, key=lambda item: float(item.get("joint_f1_mean") or 0.0), default=None)
    best_token_row = max(overall_rows, key=lambda item: float(item.get("acc_per_1k_tokens") or 0.0), default=None)

    abstract: list[str] = []
    if best_joint_row is not None:
        abstract.append(f"总体 Joint F1 最优的方法是 `{best_joint_row['method_name']}`，Joint F1 为 {format_float(best_joint_row.get('joint_f1_mean'))}。")
    if best_token_row is not None:
        abstract.append(f"单位成本效率最优的方法是 `{best_token_row['method_name']}`，每千 token 得分为 {format_float(best_token_row.get('acc_per_1k_tokens'), 6)}。")
    if diagnostics.get("key_deltas"):
        strongest_delta = max(diagnostics["key_deltas"], key=lambda item: float(item.get("joint_f1_delta") or 0.0))
        abstract.append(
            f"关键对照中提升最大的比较是 `{strongest_delta['comparison']}`，Joint F1 差值为 {format_float(strongest_delta.get('joint_f1_delta'))}。"
        )

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验聚焦 split-context 场景，核心问题是：通信是否真正改善了答案质量与证据质量，而不是只改善其中一部分。",
                "主指标包括 Answer EM / F1、Supporting Facts F1 和 Joint F1；成本侧同时记录平均通信 token / 题和平均总 token / 题。",
                "由于 split-context 通信存在强设计约束，本报告特别关注关键对照差值，以判断收益来自答案交换、证据交换还是完整消息包交换。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "Answer EM", "Answer F1", "Supporting F1", "Joint F1", "平均通信 token / 题", "平均总 token / 题", "每题调用数"],
                "rows": [
                    [
                        f"`{row['method_name']}`",
                        format_float(row.get("answer_em_mean")),
                        format_float(row.get("answer_f1_mean")),
                        format_float(row.get("supporting_f1_mean")),
                        format_float(row.get("joint_f1_mean")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                    ]
                    for row in overall_rows
                ],
            },
        },
        {
            "title": "关键对照差值",
            "table": {
                "headers": ["比较", "Answer EM 差值", "Supporting F1 差值", "Joint F1 差值", "通信 token 差值"],
                "rows": [
                    [
                        f"`{row['comparison']}`",
                        format_float(row.get("answer_em_delta")),
                        format_float(row.get("supporting_f1_delta")),
                        format_float(row.get("joint_f1_delta")),
                        format_float(row.get("communication_tokens_delta"), 2),
                    ]
                    for row in diagnostics.get("key_deltas", [])
                ],
            },
            "bullets": [
                f"split 视图数：`{diagnostics.get('split_view_count', 0)}`；full-context 参考视图数：`{diagnostics.get('full_context_view_count', 0)}`。",
                "若 Joint F1 改善主要来自 Supporting F1，而 Answer EM / F1 提升有限，说明通信更像是在修复证据而非修复最终答案。",
            ],
        },
        {
            "title": "分数据集表现",
            "tables": [
                {
                    "title": dataset,
                    "headers": ["方法", "Answer EM", "Supporting F1", "Joint F1", "平均通信 token / 题", "平均总 token / 题"],
                    "rows": [
                        [
                            f"`{row['method_name']}`",
                            format_float(row.get("answer_em_mean")),
                            format_float(row.get("supporting_f1_mean")),
                            format_float(row.get("joint_f1_mean")),
                            format_float(row.get("communication_tokens_mean"), 2),
                            format_float(row.get("total_tokens_mean"), 2),
                        ]
                        for row in _ordered_rows(
                            [item for item in metrics.get("summary", []) if item.get("dataset") == dataset]
                        )
                    ],
                }
                for dataset in sorted({row["dataset"] for row in metrics.get("summary", []) if row.get("dataset") != "overall"})
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "正式比较 split-context 通信方法时，应优先以 Joint F1 为主结论，并同步报告 Supporting F1，避免把单纯的答案修正误判为真正的信息整合收益。",
                "如果某种交换方式带来更高 Joint F1，但通信成本也明显上升，应结合成本-性能前沿图判断它是否值得成为默认方案。",
                "进入更大样本 phase 前，应优先复核关键对照差值是否稳定，而不是只依据单轮 smoke 结论推进。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前报告主要面向当前 phase 的机制验证，不直接等同于更大样本上的最终论文结论。",
                "split-context 实验高度依赖任务视图切分方式，因此结论应和具体 view 设计一起解读。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`metrics.json`、`diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`paper_summary.csv`、`hotpot_predictions/`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核与后续引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="通信必要性科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", phase),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
            ("样本数", str(len({(row.get('dataset'), row.get('sample_id')) for row in predictions}))),
        ],
        sections=sections,
    )


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row.get("method_name") in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = (
            datetime.fromisoformat(created_at).date().isoformat()
            if created_at
            else "unknown-date"
        )
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "comm-necessary")).replace("/", "-")
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
