"""为 faithful matrix 运行生成论文级统计分析。

本模块刻意保持离线分析属性：它只读取已经完成的运行产物，
绝不回写方法决策图。职责是把样本级预测配对、Bootstrap 区间估计、
以及 McNemar 类检验整理成可审计的确认性证据。
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import random
from statistics import mean
from typing import Any


PREDICTION_FILE_CANDIDATES = (
    "policy_predictions.jsonl",
    "final_predictions.jsonl",
    "predictions.jsonl",
)


@dataclass(frozen=True)
class MethodComparison:
    """表示论文包里一组固定的同 run 方法对比。"""

    comparison_id: str
    experiment_name: str
    method_a: str
    method_b: str
    use_mcnemar: bool = True


FIXED_COMPARISONS: tuple[MethodComparison, ...] = (
    MethodComparison("hybrid_trigger_vs_always_communicate", "trigger_early_exit_main", "hybrid_trigger", "always_communicate"),
    MethodComparison("hybrid_trigger_vs_sc_6", "trigger_early_exit_main", "hybrid_trigger", "sc_6"),
    MethodComparison("voc_trigger_v2_vs_mv_6", "voc_trigger_main", "voc_trigger_v2", "mv_6"),
    MethodComparison("dala_same_vs_all_to_all_full", "dala_lite_same_context_main", "dala_lite", "all_to_all_full"),
    MethodComparison("hotpot_split_comm_necessity_vs_split_no_comm_mv3", "hotpotqa_split_context_communication_necessity", "full_packet_exchange", "split_no_comm_mv3"),
    MethodComparison("dala_split_vs_all_to_all_full", "dala_lite_split_context_main", "dala_lite", "all_to_all_full"),
)


def render_paper_statistics(
    state_path_or_root: str | Path,
    *,
    output_root: str | Path | None = None,
    seed: int = 42,
    bootstrap_samples: int = 2000,
) -> dict[str, str]:
    """为一个 matrix 运行写出统计产物，并返回产物路径。"""
    state_path = _resolve_state_path(state_path_or_root)
    state_payload = _load_json(state_path)
    stats = build_paper_statistics(
        state_payload,
        seed=seed,
        bootstrap_samples=bootstrap_samples,
    )

    root = Path(output_root) if output_root is not None else state_path.parent
    root.mkdir(parents=True, exist_ok=True)
    paths = {
        "paper_statistics": root / "paper_statistics.json",
        "bootstrap_ci": root / "bootstrap_ci.json",
        "paired_win_loss": root / "paired_win_loss.json",
        "mcnemar_tests": root / "mcnemar_tests.json",
        "dataset_breakdown": root / "dataset_breakdown.json",
    }
    paths["paper_statistics"].write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["bootstrap_ci"].write_text(json.dumps(stats["bootstrap_ci"], ensure_ascii=False, indent=2), encoding="utf-8")
    paths["paired_win_loss"].write_text(json.dumps(stats["paired_win_loss"], ensure_ascii=False, indent=2), encoding="utf-8")
    paths["mcnemar_tests"].write_text(json.dumps(stats["mcnemar_tests"], ensure_ascii=False, indent=2), encoding="utf-8")
    paths["dataset_breakdown"].write_text(json.dumps(stats["dataset_breakdown"], ensure_ascii=False, indent=2), encoding="utf-8")
    return {name: path.as_posix() for name, path in paths.items()}


def build_paper_statistics(
    state_payload: dict[str, Any],
    *,
    seed: int = 42,
    bootstrap_samples: int = 2000,
) -> dict[str, Any]:
    """从已完成的 matrix 条目中构建全部固定统计对比。"""
    entries = {
        str(entry.get("experiment_name")): entry
        for entry in state_payload.get("semantic_entries", [])
        if entry.get("status") == "completed" and entry.get("run_dir")
    }

    bootstrap_ci: dict[str, Any] = {}
    paired_win_loss: dict[str, Any] = {}
    mcnemar_tests: dict[str, Any] = {}
    dataset_breakdown: dict[str, Any] = {}
    comparisons: list[dict[str, Any]] = []

    for comparison in FIXED_COMPARISONS:
        entry = entries.get(comparison.experiment_name)
        if entry is None:
            skipped = {
                "comparison_id": comparison.comparison_id,
                "status": "skipped",
                "reason": "missing_completed_experiment",
            }
            comparisons.append(skipped)
            continue

        prediction_rows = _load_prediction_rows(Path(entry["run_dir"]))
        paired = _paired_scores(
            prediction_rows,
            method_a=comparison.method_a,
            method_b=comparison.method_b,
        )
        if not paired:
            skipped = {
                "comparison_id": comparison.comparison_id,
                "status": "skipped",
                "reason": "missing_paired_predictions",
                "experiment_name": comparison.experiment_name,
                "method_a": comparison.method_a,
                "method_b": comparison.method_b,
            }
            comparisons.append(skipped)
            continue

        bootstrap_ci[comparison.comparison_id] = _bootstrap_ci(
            paired,
            seed=seed,
            bootstrap_samples=bootstrap_samples,
        )
        paired_win_loss[comparison.comparison_id] = _paired_win_loss(paired)
        dataset_breakdown[comparison.comparison_id] = _dataset_breakdown(paired)
        mcnemar_tests[comparison.comparison_id] = (
            _mcnemar_test(paired) if comparison.use_mcnemar and _is_binary_paired(paired) else {
                "status": "not_applicable",
                "reason": "non_binary_or_disabled",
            }
        )
        comparisons.append(
            {
                "comparison_id": comparison.comparison_id,
                "status": "completed",
                "experiment_name": comparison.experiment_name,
                "method_a": comparison.method_a,
                "method_b": comparison.method_b,
                "paired_n": len(paired),
                "mean_a": round(mean(item["score_a"] for item in paired), 6),
                "mean_b": round(mean(item["score_b"] for item in paired), 6),
                "mean_delta": round(mean(item["score_a"] - item["score_b"] for item in paired), 6),
            }
        )

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "bootstrap_samples": bootstrap_samples,
        "comparisons": comparisons,
        "bootstrap_ci": bootstrap_ci,
        "paired_win_loss": paired_win_loss,
        "mcnemar_tests": mcnemar_tests,
        "dataset_breakdown": dataset_breakdown,
    }


def _paired_scores(
    prediction_rows: list[dict[str, Any]],
    *,
    method_a: str,
    method_b: str,
) -> list[dict[str, Any]]:
    rows_by_method: dict[str, dict[tuple[str, str], dict[str, Any]]] = {method_a: {}, method_b: {}}
    for row in prediction_rows:
        method = str(row.get("method_name") or "")
        if method not in rows_by_method:
            continue
        dataset = str(row.get("dataset") or "")
        sample_id = str(row.get("sample_id") or "")
        if not dataset or not sample_id:
            continue
        score = _as_optional_float(_score_value(row))
        if score is None:
            continue
        rows_by_method[method][(dataset, sample_id)] = row

    shared_keys = sorted(set(rows_by_method[method_a]) & set(rows_by_method[method_b]))
    paired: list[dict[str, Any]] = []
    for dataset, sample_id in shared_keys:
        row_a = rows_by_method[method_a][(dataset, sample_id)]
        row_b = rows_by_method[method_b][(dataset, sample_id)]
        paired.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "method_a": method_a,
                "method_b": method_b,
                "score_a": float(_score_value(row_a)),
                "score_b": float(_score_value(row_b)),
            }
        )
    return paired


def _bootstrap_ci(
    paired: list[dict[str, Any]],
    *,
    seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    n = len(paired)
    deltas = [item["score_a"] - item["score_b"] for item in paired]
    score_a = [item["score_a"] for item in paired]
    score_b = [item["score_b"] for item in paired]
    sampled_delta_means: list[float] = []
    sampled_a_means: list[float] = []
    sampled_b_means: list[float] = []
    for _ in range(max(1, bootstrap_samples)):
        indices = [rng.randrange(n) for _ in range(n)]
        sampled_delta_means.append(mean(deltas[index] for index in indices))
        sampled_a_means.append(mean(score_a[index] for index in indices))
        sampled_b_means.append(mean(score_b[index] for index in indices))

    return {
        "paired_n": n,
        "mean_a": round(mean(score_a), 6),
        "mean_b": round(mean(score_b), 6),
        "mean_delta": round(mean(deltas), 6),
        "score_a_ci95": _percentile_interval(sampled_a_means),
        "score_b_ci95": _percentile_interval(sampled_b_means),
        "delta_ci95": _percentile_interval(sampled_delta_means),
    }


def _paired_win_loss(paired: list[dict[str, Any]]) -> dict[str, Any]:
    wins = sum(1 for item in paired if item["score_a"] > item["score_b"])
    losses = sum(1 for item in paired if item["score_a"] < item["score_b"])
    ties = len(paired) - wins - losses
    return {
        "paired_n": len(paired),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "win_rate_excluding_ties": None if wins + losses == 0 else round(wins / (wins + losses), 6),
        "mean_delta": round(mean(item["score_a"] - item["score_b"] for item in paired), 6),
    }


def _mcnemar_test(paired: list[dict[str, Any]]) -> dict[str, Any]:
    b = sum(1 for item in paired if item["score_a"] == 1.0 and item["score_b"] == 0.0)
    c = sum(1 for item in paired if item["score_a"] == 0.0 and item["score_b"] == 1.0)
    discordant = b + c
    if discordant == 0:
        return {
            "status": "completed",
            "b": b,
            "c": c,
            "discordant": discordant,
            "statistic": 0.0,
            "p_value_chi_square_cc": 1.0,
        }
    statistic = (abs(b - c) - 1) ** 2 / discordant
    return {
        "status": "completed",
        "b": b,
        "c": c,
        "discordant": discordant,
        "statistic": round(statistic, 6),
        "p_value_chi_square_cc": round(math.erfc(math.sqrt(statistic / 2)), 6),
    }


def _dataset_breakdown(paired: list[dict[str, Any]]) -> list[dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for item in paired:
        buckets.setdefault(str(item["dataset"]), []).append(item)
    rows: list[dict[str, Any]] = []
    for dataset, items in sorted(buckets.items()):
        rows.append(
            {
                "dataset": dataset,
                "paired_n": len(items),
                "mean_a": round(mean(item["score_a"] for item in items), 6),
                "mean_b": round(mean(item["score_b"] for item in items), 6),
                "mean_delta": round(mean(item["score_a"] - item["score_b"] for item in items), 6),
            }
        )
    return rows


def _percentile_interval(values: list[float], lower: float = 0.025, upper: float = 0.975) -> dict[str, float]:
    ordered = sorted(values)
    if not ordered:
        return {"low": 0.0, "high": 0.0}
    return {
        "low": round(_percentile(ordered, lower), 6),
        "high": round(_percentile(ordered, upper), 6),
    }


def _percentile(ordered: list[float], fraction: float) -> float:
    if len(ordered) == 1:
        return ordered[0]
    position = fraction * (len(ordered) - 1)
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def _is_binary_paired(paired: list[dict[str, Any]]) -> bool:
    allowed = {0.0, 1.0}
    return all(item["score_a"] in allowed and item["score_b"] in allowed for item in paired)


def _score_value(row: dict[str, Any]) -> Any:
    for key in ("score", "answer_em", "accuracy", "correct"):
        if key in row:
            return row.get(key)
    return None


def _load_prediction_rows(run_dir: Path) -> list[dict[str, Any]]:
    for filename in PREDICTION_FILE_CANDIDATES:
        path = run_dir / filename
        if path.exists():
            return _load_jsonl(path)
    return []


def _resolve_state_path(state_path_or_root: str | Path) -> Path:
    path = Path(state_path_or_root)
    if path.is_dir():
        candidate = path / "state.json"
        if candidate.exists():
            return candidate
    return path


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(json.loads(stripped))
    return rows


def _as_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
