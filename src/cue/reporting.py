"""CUE 实验的科研报告与图资产生成。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
from typing import Any

from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.analysis_reports import render_frontier_report, write_report
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
from experiment_core.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


METHOD_ORDER = [
    "mv_3",
    "always_communicate",
    "disagreement_triggered",
    "consensus_freeze",
    "cue_v1",
    "mv_6",
    "sc_6",
]


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    metrics = _load_json(Path(run_dir) / "policy_metrics.json")
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


def render_cue_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> dict[str, Any]:
    publish_dir = publish_dir or default_reports_root("cue")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = _load_json(root / "policy_metrics.json")
    diagnostics = _load_json(root / "policy_diagnostics.json")
    oracle = _load_json(root / "oracle_trigger_eval.json")
    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics, diagnostics))
    base_markdown = _render_markdown(manifest, metrics, diagnostics, oracle, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")
    write_report(root / "frontier_report.md", render_frontier_report(metrics.get("summary", []), title="CUE 前沿附录"))
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
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any], diagnostics: dict[str, Any]) -> list[dict[str, Any]]:
    summary_rows = metrics.get("summary", [])
    overall_policy_rows = [
        row for row in diagnostics.get("policy_rows", [])
        if row.get("dataset") == "overall"
    ]
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="CUE 成本-性能前沿",
            caption="总体结果上，各策略的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="CUE 效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="acc_per_1k_tokens",
            primary_metric="每千 token 准确率",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="CUE 跨数据集表现",
            caption="各策略在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
        ),
        build_scatter_figure_spec(
            figure_id="policy_tradeoff",
            title="策略触发率权衡",
            caption="总体触发率相对于准确率的变化。",
            primary_metric="准确率",
            data=[
                {
                    "label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "short_label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "x": float(row.get("trigger_rate") or 0.0),
                    "y": float(row.get("accuracy_mean") or 0.0),
                    "value": float(row.get("accuracy_mean") or 0.0),
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
            figure_id="oracle_alignment",
            title="Oracle 对齐情况",
            caption="总体 Oracle 精确率与召回率对比。",
            primary_metric="比率",
            data=[
                {
                    "label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "short_label": str(row.get("display_name") or row.get("policy_name") or "unknown"),
                    "precision": float(row.get("precision") or 0.0),
                    "recall": float(row.get("recall") or 0.0),
                }
                for row in overall_policy_rows
            ],
            series=[("precision", "精确率"), ("recall", "召回率")],
            x_label="比率",
            source_kind="policy_diagnostics",
            dataset_scope="overall",
            note="精确率和召回率共同衡量策略是否把通信机会分配给真正有收益的样本。",
        ),
    ]


def _render_markdown(
    manifest: dict[str, Any],
    metrics: dict[str, Any],
    diagnostics: dict[str, Any],
    oracle: dict[str, Any],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    metric_rows = metrics.get("summary", [])
    policy_rows = diagnostics.get("policy_rows", [])
    overall_main_rows = _ordered_rows([row for row in metric_rows if row.get("dataset") == "overall"])
    per_dataset_rows = {
        dataset: _ordered_rows([row for row in metric_rows if row.get("dataset") == dataset])
        for dataset in sorted({row["dataset"] for row in metric_rows if row.get("dataset") != "overall"})
    }
    recommendation = diagnostics.get("recommended_next_default_policy", {})
    best_row = max(overall_main_rows, key=lambda item: float(item.get("accuracy_mean") or 0.0), default=None)
    best_policy_row = max(
        [row for row in overall_main_rows if str(row.get("method_name")) in {"cue_v1", "consensus_freeze", "disagreement_triggered", "always_communicate"}],
        key=lambda item: float(item.get("accuracy_mean") or 0.0),
        default=None,
    )

    abstract: list[str] = []
    if best_row is not None:
        abstract.append(f"总体准确率最高的方法是 `{best_row['display_name']}`，准确率为 {format_float(best_row.get('accuracy_mean'))}。")
    if best_policy_row is not None:
        abstract.append(f"在通信策略中，`{best_policy_row['display_name']}` 的总体表现最佳。")
    abstract.append(
        f"当前推荐的下一轮默认策略为 `{recommendation.get('selected_policy', 'cue_v1')}`。"
    )

    sections = [
        {
            "title": "研究问题与方法结构",
            "bullets": [
                "CUE 的核心问题是：在黑盒条件下，能否先估计通信效用，再把通信机会定向分配给真正需要协同的样本。",
                "主指标为准确率；成本指标采用平均总 token / 题与平均通信 token / 题；策略诊断侧重触发率、误触发率与漏掉有益通信率。",
                "报告中特别关注 `cue_v1` 相对于 `always_communicate` 是否能够在显著降低成本时保持可接受的性能损失。",
            ],
        },
        {
            "title": "总体结果",
            "table": {
                "headers": ["方法", "准确率", "平均通信 token / 题", "平均总 token / 题", "每题调用数", "每千 token 准确率"],
                "rows": [
                    [
                        f"`{row['display_name']}`",
                        format_float(row.get("accuracy_mean")),
                        format_float(row.get("communication_tokens_mean"), 2),
                        format_float(row.get("total_tokens_mean"), 2),
                        format_float(row.get("calls_per_question_mean"), 2),
                        format_float(row.get("acc_per_1k_tokens"), 6),
                    ]
                    for row in overall_main_rows
                ],
            },
        },
        {
            "title": "触发与 Oracle 诊断",
            "table": {
                "headers": ["策略", "触发率", "早退率", "Oracle 精确率", "Oracle 召回率", "误触发率", "漏掉有益通信率"],
                "rows": [
                    [
                        f"`{row['display_name']}`",
                        format_float(row.get("trigger_rate")),
                        format_float(row.get("early_exit_rate")),
                        format_float(row.get("precision")),
                        format_float(row.get("recall")),
                        format_float(row.get("false_trigger_rate")),
                        format_float(row.get("missed_beneficial_comm_rate")),
                    ]
                    for row in _ordered_policy_rows([row for row in policy_rows if row.get("dataset") == "overall"])
                ],
            },
            "bullets": [
                f"推荐策略：`{recommendation.get('selected_policy', 'cue_v1')}`。",
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
                            f"`{row['display_name']}`",
                            format_float(row.get("accuracy_mean")),
                            format_float(row.get("communication_tokens_mean"), 2),
                            format_float(row.get("total_tokens_mean"), 2),
                            format_float(row.get("acc_per_1k_tokens"), 6),
                        ]
                        for row in rows
                    ],
                }
                for dataset, rows in per_dataset_rows.items()
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "若 CUE 的 Oracle 精确率和召回率同时偏低，则应优先改进 utility 估计，而不是直接增加通信预算。",
                "若准确率接近 `always_communicate`，但总 token 明显下降，则说明 CUE 已具备作为默认策略的工程价值。",
                "进入更大样本 phase 前，建议优先复核误触发率与漏掉有益通信率是否同时可控，避免只依赖总体准确率决策。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "当前 Oracle 是基于现有控制组构造的近似标签，因此更适合做策略筛选，而非最终机制证明。",
                "本报告反映的是当前 phase 的黑盒实现效果，不能直接等同于可访问内部不确定性信号时的理想上界。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`policy_metrics.json`、`policy_diagnostics.json`、`report.md`、`figure_manifest.json`、`figures/`、`oracle_trigger_eval.json`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于后续引用与审稿期复核。",
            ],
        ),
    ]
    return render_scientific_report(
        title="CUE 科研报告",
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


def _ordered_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["method_name"]) if row["method_name"] in METHOD_ORDER else 999)


def _ordered_policy_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: METHOD_ORDER.index(row["policy_name"]) if row["policy_name"] in METHOD_ORDER else 999)


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "cue")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-cue-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))
