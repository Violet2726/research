"""多智能体实验报告与配对分析。

该模块重点服务于 `Debate vs Vote` 分析：
在同一批初始候选答案上，对比“直接投票”和“先 debate 再投票”两种聚合方式，
并输出配对统计、bootstrap 区间和中文报告。
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime
from pathlib import Path
import json
import math
import random
from typing import Any

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
from experiment_core.foundation.workspace import default_reports_root

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
    """生成 `Debate vs Vote` 的配对分析结果与 Markdown 报告。"""
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
            title="Multi-agent frontier",
            caption="Overall accuracy versus average total tokens across multi-agent methods and matched controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
            method_label_field="method_name",
        ),
        build_efficiency_rank_figure_spec(
            summary_rows,
            title="Multi-agent efficiency ranking",
            caption="Overall efficiency ranking measured by accuracy per 1K tokens.",
            efficiency_field="accuracy_per_1k_tokens",
            primary_metric="Accuracy per 1K tokens",
            method_label_field="method_name",
        ),
        build_score_by_dataset_figure_spec(
            summary_rows,
            title="Multi-agent score by dataset",
            caption="Per-dataset accuracy map across multi-agent methods and matched controls.",
            score_field="accuracy_mean",
            primary_metric="Accuracy",
            method_label_field="method_name",
        ),
        build_grouped_bar_figure_spec(
            figure_id="debate_delta_breakdown",
            title="Debate delta breakdown",
            caption="Per-dataset correction and harm rates alongside the net accuracy delta.",
            primary_metric="Rate or delta",
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
                ("corrected_rate", "Corrected rate"),
                ("harmed_rate", "Harmed rate"),
                ("accuracy_delta", "Accuracy delta"),
            ],
            x_label="Rate or delta",
            source_kind="paired_debate_vs_vote",
            dataset_scope="per_dataset",
            note="Correction and harm are normalized by question count; the delta series preserves signed paired gain.",
        ),
        build_interval_figure_spec(
            figure_id="paired_effect_ci",
            title="Paired effect confidence intervals",
            caption="Paired bootstrap intervals for the debate-minus-vote accuracy delta.",
            primary_metric="Accuracy delta",
            data=interval_rows,
            x_label="Accuracy delta",
            source_kind="paired_debate_vs_vote",
            dataset_scope="per_dataset",
            note="Intervals that stay above zero indicate a stable paired debate gain for the reported dataset.",
        ),
    ]


def _paired_summary_for_group(
    rows: list[dict[str, Any]],
    dataset: str,
    method_name: str,
    phase_name: str | None,
) -> dict[str, Any]:
    """对同一数据集与方法的题级结果做配对分析。"""
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
    """对配对准确率差异做 McNemar 与 bootstrap 估计。"""
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
    """对 debate 相对 initial vote 的准确率增益做配对 bootstrap。"""
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
    """计算双侧 McNemar exact p-value。"""
    discordant = harmed_count + corrected_count
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, k) for k in range(0, min(harmed_count, corrected_count) + 1)
    ) / (2**discordant)
    return round(min(1.0, 2 * tail), 6)


def _render_debate_vs_vote_report(
    manifest: dict[str, Any],
    dataset_rows: list[dict[str, Any]],
    run_dir: Path,
) -> str:
    """渲染中文 Markdown 正式报告。"""
    backbone = {"name": resolve_manifest_model_name(manifest)}
    lines = [
        "# Debate vs Vote 对照实验报告",
        "",
        "## 1. 实验概览",
        "",
        f"- 实验名：`{manifest.get('experiment')}`",
        f"- Phase：`{manifest.get('phase')}`",
        f"- Prompt Version：`{manifest.get('prompt_version')}`",
        f"- Backbone：`{backbone.get('name')}`",
        f"- 运行目录：`{run_dir.as_posix()}`",
        "",
        "本实验比较的是同一批初始候选答案上的两种聚合方式：",
        "",
        "- `initial vote`：3 个 agent 的初始独立答案直接多数投票。",
        "- `debate vote`：同样 3 个 agent 在 1 轮 debate 后，对修订答案再次投票。",
        "",
        "因此，这是一组严格配对的 Debate vs Vote 实验，而不是两套独立采样的横向比较。",
        "",
        "## 2. 结果摘要",
        "",
    ]

    for row in dataset_rows:
        lines.extend(
            [
                f"### {row['dataset']}",
                "",
                f"- 题量：`{row['question_count']}`",
                f"- initial vote 准确率：`{row['initial_vote_accuracy']:.4f}`",
                f"- debate vote 准确率：`{row['debate_vote_accuracy']:.4f}`",
                f"- 准确率变化：`{row['accuracy_delta']:+.4f}`",
                f"- debate 修正题数：`{row['corrected_count']}`",
                f"- debate 改坏题数：`{row['harmed_count']}`",
                f"- 保持正确题数：`{row['unchanged_correct_count']}`",
                f"- 保持错误题数：`{row['unchanged_wrong_count']}`",
                f"- 初始一致率：`{row['initial_consensus_rate']:.4f}`",
                f"- debate 后一致率：`{row['final_consensus_rate']:.4f}`",
                f"- debate 翻票率：`{row['debate_flip_rate']:.4f}`",
                f"- debate 增量 token / 题：`{row['debate_incremental_tokens_mean']:.2f}`",
                f"- debate 增量时延 / 题：`{row['debate_incremental_latency_mean']:.2f} ms`",
                f"- 每 1k debate token 的准确率增益：`{row['accuracy_gain_per_1k_debate_tokens']:+.6f}`",
            ]
        )
        stats = row["statistics"]
        if stats["computed"]:
            lines.extend(
                [
                    f"- McNemar exact p：`{stats['mcnemar_exact_p']}`",
                    f"- Bootstrap 95% CI：`[{stats['bootstrap_ci_95'][0]}, {stats['bootstrap_ci_95'][1]}]`",
                ]
            )
        else:
            lines.append(f"- 统计检验：未计算（{stats['reason']}）")
        lines.append("")

    lines.extend(
        [
            "## 3. 解读注意事项",
            "",
            "- `smoke20` 只用于协议联调与方向观察，不做统计显著性结论。",
            "- `pilot100` 才作为主结论来源；若 HotpotQA 上仍有跨度表述噪音，应在结论里单独说明。",
            "- 若 `accuracy_delta` 接近 0，但 debate 增量 token 很高，说明 debate 没有提供足够的边际收益。",
            "",
        ]
    )
    return "\n".join(lines) + "\n"


def _enrich_prediction_row(row: dict[str, Any]) -> dict[str, Any]:
    """为旧版或新版题级记录补齐配对分析所需字段。"""
    enriched = dict(row)
    gold = str(enriched["gold"])
    final_score = float(enriched.get("final_vote_score", enriched.get("score", 0.0)))
    initial_vote_prediction = enriched.get("initial_vote_prediction") or enriched.get("prediction") or ""
    initial_vote_score = enriched.get("initial_vote_score")
    if initial_vote_score is None:
        initial_vote_score = _score_like_dataset(enriched["dataset"], initial_vote_prediction, gold)
    enriched["initial_vote_prediction"] = initial_vote_prediction
    enriched["initial_vote_score"] = float(initial_vote_score)
    enriched["initial_consensus"] = bool(
        enriched.get("initial_consensus", not enriched.get("initial_disagreement", False))
    )
    enriched["final_vote_prediction"] = str(enriched.get("final_vote_prediction", enriched.get("prediction", "")))
    enriched["final_vote_score"] = final_score
    enriched["debate_total_tokens_per_question"] = float(enriched.get("debate_total_tokens_per_question", 0.0))
    enriched["debate_prompt_tokens_per_question"] = float(enriched.get("debate_prompt_tokens_per_question", 0.0))
    enriched["debate_completion_tokens_per_question"] = float(enriched.get("debate_completion_tokens_per_question", 0.0))
    enriched["debate_latency_ms_per_question"] = float(enriched.get("debate_latency_ms_per_question", 0.0))
    enriched["corrected_by_debate"] = bool(
        enriched.get("corrected_by_debate", enriched["initial_vote_score"] < 1.0 and final_score == 1.0)
    )
    enriched["harmed_by_debate"] = bool(
        enriched.get("harmed_by_debate", enriched["initial_vote_score"] == 1.0 and final_score < 1.0)
    )
    enriched["unchanged_correct"] = bool(
        enriched.get("unchanged_correct", enriched["initial_vote_score"] == 1.0 and final_score == 1.0)
    )
    enriched["unchanged_wrong"] = bool(
        enriched.get("unchanged_wrong", enriched["initial_vote_score"] < 1.0 and final_score < 1.0)
    )
    return enriched


def _score_like_dataset(dataset: str, predicted: str, gold: str) -> float:
    """轻量复用任务级打分逻辑，避免 reporting 层循环依赖 runner。"""
    from experiment_core.foundation.evaluation import score_prediction

    return float(score_prediction(dataset, predicted, gold))


def _published_report_name(manifest: dict[str, Any]) -> str:
    """构造写入 ``reports/`` 的报告文件名。"""
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
    """读取 UTF-8 JSON 文件。"""
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL 文件。"""
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _mean(values) -> float:
    """安全计算均值。"""
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)


def _quantile(values: list[float], q: float) -> float:
    """计算简单分位数。"""
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
    """安全计算比例。"""
    if denominator == 0:
        return 0.0
    return round(numerator / denominator, 6)

