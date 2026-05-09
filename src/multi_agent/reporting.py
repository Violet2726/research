"""多智能体实验的科研报告与配对分析。"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

from experiment_core.foundation.workspace import default_reports_root
from experiment_core.reporting.reporting_utils import resolve_manifest_model_name
from experiment_core.reporting.run_figures import (
    append_figure_gallery_markdown,
    build_efficiency_rank_figure_spec,
    build_frontier_figure_spec,
    build_grouped_bar_figure_spec,
    build_interval_figure_spec,
    build_score_by_dataset_figure_spec,
    write_figure_bundle,
)
from experiment_core.reporting.scientific_report import (
    format_float,
    render_run_reproducibility_section,
    render_scientific_report,
)


def load_metrics(run_dir: str | Path) -> dict[str, Any]:
    """读取多智能体运行目录中的 `metrics.json`。"""
    return json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    """按数据集汇总多智能体实验摘要。"""
    metrics = load_metrics(run_dir)
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


def report_debate_vs_vote(
    run_dir: str | Path,
    publish_dir: str | Path | None = None,
) -> dict[str, Any]:
    """生成 Debate vs Vote 配对分析结果与中文科研报告。"""
    publish_dir = publish_dir or default_reports_root("multi_agent")
    root = Path(run_dir)
    manifest = _load_json(root / "manifest.json")
    metrics = load_metrics(root)
    prediction_rows = _load_jsonl(root / "final_predictions.jsonl")
    mad_rows = [row for row in prediction_rows if row.get("method_type") == "mad"]
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in mad_rows:
        enriched = _enrich_prediction_row(row)
        grouped[(enriched["dataset"], enriched["method_name"])].append(enriched)

    dataset_rows: list[dict[str, Any]] = []
    for (dataset, method_name), rows in sorted(grouped.items()):
        dataset_rows.append(_paired_summary_for_group(rows, dataset, method_name, manifest.get("phase")))

    paired_payload = {
        "run_dir": str(root),
        "experiment": manifest.get("experiment"),
        "phase": manifest.get("phase"),
        "prompt_version": manifest.get("prompt_version"),
        "backbone": manifest.get("backbone"),
        "rows": dataset_rows,
    }

    paired_json_path = root / "paired_debate_vs_vote.json"
    paired_json_path.write_text(json.dumps(paired_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    figure_bundle = write_figure_bundle(root, _build_figure_specs(metrics, dataset_rows))
    base_markdown = _render_debate_vs_vote_report(manifest, dataset_rows, root)
    markdown = append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root)
    local_report_path = root / "report.md"
    local_report_path.write_text(markdown, encoding="utf-8")

    publish_path = Path(publish_dir) / _published_report_name(manifest)
    publish_path.parent.mkdir(parents=True, exist_ok=True)
    publish_path.write_text(
        append_figure_gallery_markdown(base_markdown, figure_bundle["figures"], run_dir=root, published_path=publish_path),
        encoding="utf-8",
    )

    return {
        "run_dir": str(root),
        "paired_json": str(paired_json_path),
        "local_report": str(local_report_path),
        "published_report": str(publish_path),
        "dataset_count": len(dataset_rows),
        "figure_manifest": str(root / "figure_manifest.json"),
    }


def _build_figure_specs(metrics: dict[str, Any], dataset_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows = metrics.get("summary", [])
    interval_rows = []
    for row in dataset_rows:
        stats = row.get("statistics") or {}
        ci = stats.get("bootstrap_ci_95")
        if not stats.get("computed") or not isinstance(ci, list) or len(ci) != 2:
            continue
        interval_rows.append(
            {
                "label": f"{row['dataset']}:{row['method_name']}",
                "short_label": str(row["dataset"]),
                "value": float(row.get("accuracy_delta") or 0.0),
                "low": float(ci[0]),
                "high": float(ci[1]),
            }
        )
    return [
        build_frontier_figure_spec(
            summary_rows,
            title="多智能体成本-性能前沿",
            caption="总体结果上，多智能体方法及其对照的准确率相对于平均总 token 的位置关系。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="多智能体效率排序",
            caption="基于每千 token 准确率的总体效率排序。",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="每千 token 准确率",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="多智能体跨数据集表现",
            caption="各多智能体方法及其对照在不同数据集上的准确率分布。",
            score_field="accuracy_mean",
            primary_metric="准确率",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="debate_delta_breakdown",
            title="Debate 增益分解",
            caption="按分组展示纠正率、伤害率和净准确率差值。",
            primary_metric="比率或差值",
            data=[
                {
                    "label": f"{row['dataset']}:{row['method_name']}",
                    "short_label": str(row["dataset"]),
                    "corrected_rate": (float(row.get("corrected_count") or 0.0) / float(row.get("question_count") or 1.0)),
                    "harmed_rate": (float(row.get("harmed_count") or 0.0) / float(row.get("question_count") or 1.0)),
                    "accuracy_delta": float(row.get("accuracy_delta") or 0.0),
                }
                for row in dataset_rows
            ],
            series=[
                ("corrected_rate", "纠正率"),
                ("harmed_rate", "伤害率"),
                ("accuracy_delta", "准确率差值"),
            ],
            x_label="比率或差值",
            source_kind="paired_debate_vs_vote",
            dataset_scope="per_dataset",
            note="纠正率和伤害率都标准化为题级比例，便于和净准确率差值并列解释。",
        ),
        build_interval_figure_spec(
            figure_id="paired_effect_ci",
            title="配对效应置信区间",
            caption="debate 相对 initial vote 的配对 bootstrap 置信区间。",
            primary_metric="准确率差值",
            data=interval_rows,
            x_label="准确率差值",
            source_kind="paired_debate_vs_vote",
            dataset_scope="per_dataset",
            note="若区间整体高于 0，说明 debate 增益在该分组上较稳定。",
        ),
    ]


def _paired_summary_for_group(
    rows: list[dict[str, Any]],
    dataset: str,
    method_name: str,
    phase_name: str | None,
) -> dict[str, Any]:
    question_count = len(rows)
    initial_correct_count = sum(1 for row in rows if row["initial_vote_score"] == 1.0)
    final_correct_count = sum(1 for row in rows if row["final_vote_score"] == 1.0)
    corrected_count = sum(1 for row in rows if row["corrected_by_debate"])
    harmed_count = sum(1 for row in rows if row["harmed_by_debate"])
    unchanged_correct_count = sum(1 for row in rows if row["unchanged_correct"])
    unchanged_wrong_count = sum(1 for row in rows if row["unchanged_wrong"])

    initial_vote_accuracy = _ratio(initial_correct_count, question_count)
    debate_vote_accuracy = _ratio(final_correct_count, question_count)
    accuracy_delta = round(debate_vote_accuracy - initial_vote_accuracy, 6)
    debate_incremental_tokens_mean = _mean(float(row["debate_total_tokens_per_question"]) for row in rows)
    debate_incremental_latency_mean = _mean(float(row["debate_latency_ms_per_question"]) for row in rows)
    accuracy_gain_per_1k_debate_tokens = (
        round(accuracy_delta / debate_incremental_tokens_mean * 1000, 6)
        if debate_incremental_tokens_mean
        else 0.0
    )

    payload = {
        "dataset": dataset,
        "method_name": method_name,
        "question_count": question_count,
        "initial_vote_accuracy": initial_vote_accuracy,
        "debate_vote_accuracy": debate_vote_accuracy,
        "accuracy_delta": accuracy_delta,
        "corrected_count": corrected_count,
        "harmed_count": harmed_count,
        "unchanged_correct_count": unchanged_correct_count,
        "unchanged_wrong_count": unchanged_wrong_count,
        "debate_flip_rate": _ratio(sum(1 for row in rows if row["vote_flipped"]), question_count),
        "initial_consensus_rate": _ratio(sum(1 for row in rows if row["initial_consensus"]), question_count),
        "final_consensus_rate": _ratio(sum(1 for row in rows if row["final_consensus"]), question_count),
        "debate_incremental_prompt_tokens_mean": _mean(float(row["debate_prompt_tokens_per_question"]) for row in rows),
        "debate_incremental_completion_tokens_mean": _mean(float(row["debate_completion_tokens_per_question"]) for row in rows),
        "debate_incremental_tokens_mean": debate_incremental_tokens_mean,
        "debate_incremental_latency_mean": debate_incremental_latency_mean,
        "accuracy_gain_per_1k_debate_tokens": accuracy_gain_per_1k_debate_tokens,
    }

    if phase_name == "pilot100":
        payload["statistics"] = _paired_statistics(rows)
    else:
        payload["statistics"] = {
            "computed": False,
            "reason": "statistics are only reported for pilot100",
            "mcnemar_exact_p": None,
            "bootstrap_ci_95": None,
            "bootstrap_iterations": 0,
        }
    return payload


def _paired_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    harmed = sum(1 for row in rows if row["harmed_by_debate"])
    corrected = sum(1 for row in rows if row["corrected_by_debate"])
    deltas = _bootstrap_accuracy_delta(rows, iterations=10000, seed=42)
    return {
        "computed": True,
        "mcnemar_exact_p": _mcnemar_exact_p(harmed, corrected),
        "bootstrap_ci_95": [_quantile(deltas, 0.025), _quantile(deltas, 0.975)],
        "bootstrap_iterations": len(deltas),
        "discordant_pairs": harmed + corrected,
    }


def _bootstrap_accuracy_delta(
    rows: list[dict[str, Any]],
    iterations: int,
    seed: int,
) -> list[float]:
    rng = random.Random(seed)
    indexed = list(rows)
    samples: list[float] = []
    if not indexed:
        return [0.0]
    for _ in range(iterations):
        picked = [indexed[rng.randrange(len(indexed))] for _ in range(len(indexed))]
        initial_accuracy = sum(row["initial_vote_score"] for row in picked) / len(picked)
        final_accuracy = sum(row["final_vote_score"] for row in picked) / len(picked)
        samples.append(round(final_accuracy - initial_accuracy, 6))
    return samples


def _mcnemar_exact_p(harmed_count: int, corrected_count: int) -> float:
    discordant = harmed_count + corrected_count
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, k) for k in range(0, min(harmed_count, corrected_count) + 1)) / (2**discordant)
    return round(min(1.0, 2 * tail), 6)


def _render_debate_vs_vote_report(
    manifest: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    backbone_name = resolve_manifest_model_name(manifest)
    best_row = max(dataset_rows, key=lambda item: float(item.get("accuracy_delta") or 0.0), default=None)
    abstract: list[str] = [
        "该实验严格共享同一批初始候选答案，因此所有差异都可解释为 debate 环节本身带来的净收益或净伤害。"
    ]
    if best_row is not None:
        abstract.insert(
            0,
            f"配对增益最大的分组是 `{best_row['dataset']}` / `{best_row['method_name']}`，debate 相对 vote 的准确率差值为 {best_row['accuracy_delta']:+.4f}。",
        )

    sections = [
        {
            "title": "研究问题与实验设计",
            "bullets": [
                "本实验比较同一批初始候选答案上的两种聚合方式：直接投票的 `initial vote`，以及单轮 debate 后再投票的 `debate vote`。",
                "主指标为配对准确率差值；机制指标包括 corrected / harmed、consensus 变化、flip rate，以及 debate 增量 token / 时延。",
                "因为是严格配对设计，所以准确率差值、McNemar 检验和 bootstrap 区间都可以直接解释 debate 的净贡献。",
            ],
        },
        {
            "title": "分组结果总表",
            "table": {
                "headers": ["数据集", "方法", "题量", "initial vote 准确率", "debate vote 准确率", "准确率差值", "纠正题数", "伤害题数", "每千 debate token 增益"],
                "rows": [
                    [
                        f"`{row['dataset']}`",
                        f"`{row['method_name']}`",
                        str(row["question_count"]),
                        format_float(row.get("initial_vote_accuracy")),
                        format_float(row.get("debate_vote_accuracy")),
                        f"{float(row.get('accuracy_delta') or 0.0):+.4f}",
                        str(int(row.get("corrected_count") or 0)),
                        str(int(row.get("harmed_count") or 0)),
                        f"{float(row.get('accuracy_gain_per_1k_debate_tokens') or 0.0):+.6f}",
                    ]
                    for row in dataset_rows
                ],
            },
        },
        {
            "title": "配对统计与机制诊断",
            "tables": [
                {
                    "title": f"{row['dataset']} / {row['method_name']}",
                    "headers": ["指标", "数值"],
                    "rows": [
                        ["初始一致率", format_float(row.get("initial_consensus_rate"))],
                        ["debate 后一致率", format_float(row.get("final_consensus_rate"))],
                        ["翻票率", format_float(row.get("debate_flip_rate"))],
                        ["增量 token / 题", format_float(row.get("debate_incremental_tokens_mean"), 2)],
                        ["增量时延 / 题 (ms)", format_float(row.get("debate_incremental_latency_mean"), 2)],
                        ["McNemar exact p", str((row.get("statistics") or {}).get("mcnemar_exact_p", "未计算"))],
                        [
                            "Bootstrap 95% CI",
                            f"[{(row.get('statistics') or {}).get('bootstrap_ci_95', ['未计算', '未计算'])[0]}, {(row.get('statistics') or {}).get('bootstrap_ci_95', ['未计算', '未计算'])[1]}]"
                            if (row.get("statistics") or {}).get("computed")
                            else str((row.get("statistics") or {}).get("reason", "未计算")),
                        ],
                    ],
                }
                for row in dataset_rows
            ],
        },
        {
            "title": "结论与建议",
            "bullets": [
                "如果某个分组的 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。",
                "如果 corrected 明显多于 harmed，且 bootstrap 区间稳定高于 0，则说明 debate 在该分组具备真实增益。",
                "进入更大样本 phase 前，应优先挑选配对增益稳定为正的设置继续扩展，而不是默认所有 debate 都值得保留。",
            ],
        },
        {
            "title": "局限性",
            "bullets": [
                "`smoke20` 更适合联调与方向观察，不作为显著性结论来源；正式结论仍应依赖 `pilot100` 及以上 phase。",
                "若某个数据集存在明显噪声或较强题型异质性，应单独解释其 debate 收益，而不是简单并入总体叙述。",
            ],
        },
        render_run_reproducibility_section(
            run_dir=run_dir,
            artifact_items=[
                "关键产物：`paired_debate_vs_vote.json`、`metrics.json`、`report.md`、`figure_manifest.json`、`figures/`。",
                "本地报告与发布报告共享同一套 run 内图资产，便于复核与引用。",
            ],
        ),
    ]
    return render_scientific_report(
        title="多智能体 Debate vs Vote 科研报告",
        abstract=abstract,
        overview_items=[
            ("实验名", str(manifest.get("experiment"))),
            ("Phase", str(manifest.get("phase"))),
            ("Prompt Version", str(manifest.get("prompt_version"))),
            ("Backbone", backbone_name),
            ("运行目录", run_dir.as_posix()),
        ],
        sections=sections,
    )


def _enrich_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(row)
    gold = str(enriched["gold"])
    final_score = float(enriched.get("final_vote_score", enriched.get("score", 0.0)))
    initial_vote_prediction = enriched.get("initial_vote_prediction") or enriched.get("prediction") or ""
    initial_vote_score = enriched.get("initial_vote_score")
    if initial_vote_score is None:
        initial_vote_score = _score_like_dataset(enriched["dataset"], initial_vote_prediction, gold)
    enriched["initial_vote_prediction"] = initial_vote_prediction
    enriched["initial_vote_score"] = float(initial_vote_score)
    enriched["initial_consensus"] = bool(enriched.get("initial_consensus", not enriched.get("initial_disagreement", False)))
    enriched["final_vote_prediction"] = str(enriched.get("final_vote_prediction", enriched.get("prediction", "")))
    enriched["final_vote_score"] = final_score
    enriched["debate_total_tokens_per_question"] = float(enriched.get("debate_total_tokens_per_question", 0.0))
    enriched["debate_prompt_tokens_per_question"] = float(enriched.get("debate_prompt_tokens_per_question", 0.0))
    enriched["debate_completion_tokens_per_question"] = float(enriched.get("debate_completion_tokens_per_question", 0.0))
    enriched["debate_latency_ms_per_question"] = float(enriched.get("debate_latency_ms_per_question", 0.0))
    enriched["corrected_by_debate"] = bool(enriched.get("corrected_by_debate", enriched["initial_vote_score"] < 1.0 and final_score == 1.0))
    enriched["harmed_by_debate"] = bool(enriched.get("harmed_by_debate", enriched["initial_vote_score"] == 1.0 and final_score < 1.0))
    enriched["unchanged_correct"] = bool(enriched.get("unchanged_correct", enriched["initial_vote_score"] == 1.0 and final_score == 1.0))
    enriched["unchanged_wrong"] = bool(enriched.get("unchanged_wrong", enriched["initial_vote_score"] < 1.0 and final_score < 1.0))
    return enriched


def _score_like_dataset(dataset: str, predicted: str, gold: str) -> float:
    from experiment_core.foundation.evaluation import score_prediction

    return float(score_prediction(dataset, predicted, gold))


def _published_report_name(manifest: dict[str, Any]) -> str:
    created_at = manifest.get("created_at")
    try:
        created_date = datetime.fromisoformat(created_at).date().isoformat() if created_at else "unknown-date"
    except ValueError:
        created_date = "unknown-date"
    experiment = str(manifest.get("experiment", "multi-agent")).replace("/", "-")
    phase = str(manifest.get("phase", "phase")).replace("/", "-")
    backbone_name = str(manifest.get("backbone", {}).get("name", "backbone")).replace("/", "-")
    return f"{created_date}-{experiment}-{phase}-{backbone_name}-report.md"


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def _ratio(numerator: int, denominator: int) -> float:
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)
