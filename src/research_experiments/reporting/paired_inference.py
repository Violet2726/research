"""共享的逐题配对 bootstrap、McNemar 与 Holm 统计推断。"""

from __future__ import annotations

import hashlib
import math
import random
from collections import defaultdict
from typing import Any


def paired_statistics(
    rows: list[dict[str, Any]],
    *,
    reference: str,
    competitors: list[str],
    seed: int,
    bootstrap_samples: int = 10_000,
    bbeh_adjusted_harmonic: bool = False,
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        for competitor in competitors:
            pairs = _pairs(rows, dataset, reference, competitor)
            if not pairs:
                continue
            metric = _primary_metric(dataset)
            point = _paired_point(pairs, metric=metric)
            low, high = _bootstrap(
                pairs,
                samples=bootstrap_samples,
                seed=f"{seed}:{dataset}:{competitor}",
                metric=metric,
            )
            reference_only = sum(left > right for left, right, _ in pairs)
            competitor_only = sum(right > left for left, right, _ in pairs)
            result = {
                    "dataset": dataset,
                    "reference_method": reference,
                    "comparison_method": competitor,
                    "paired_question_count": len(pairs),
                    "mean_accuracy_delta": point,
                    "accuracy_metric": metric,
                    "bootstrap_ci_95": [low, high],
                    "mcnemar_b_reference_only_correct": reference_only,
                    "mcnemar_c_comparator_only_correct": competitor_only,
                    "mcnemar_exact_p": _mcnemar_p(reference_only, competitor_only),
                }
            if bbeh_adjusted_harmonic and dataset == "bbeh":
                compat_low, compat_high = _bootstrap(
                    pairs,
                    samples=bootstrap_samples,
                    seed=f"{seed}:{dataset}:{competitor}:adjusted_harmonic",
                    metric="task_adjusted_harmonic",
                )
                result.update(
                    {
                        "bbeh_adjusted_harmonic_delta": _paired_point(
                            pairs,
                            metric="task_adjusted_harmonic",
                        ),
                        "bbeh_adjusted_harmonic_bootstrap_ci_95": [compat_low, compat_high],
                        "bbeh_adjusted_harmonic_interpretation": "secondary_full_compatibility_metric",
                    }
                )
            tests.append(result)
    present_datasets = {str(item["dataset"]) for item in tests}
    science_dataset = (
        "supergpqa_science"
        if "supergpqa_science" in present_datasets
        else "supergpqa"
        if "supergpqa" in present_datasets
        else "gpqa_diamond"
    )
    holm_scope = ["bbeh", "musr", science_dataset]
    holm_datasets = set(holm_scope)
    primary_tests = [item for item in tests if item["dataset"] in holm_datasets]
    _holm(primary_tests)
    return {
        "reference_method": reference,
        "bootstrap_samples": bootstrap_samples,
        "tests": tests,
        "bbeh_resampling": "within_task_stratified_micro_with_secondary_adjusted_harmonic"
        if bbeh_adjusted_harmonic
        else "within_task_stratified_micro",
        "musr_resampling": "within_task_stratified_macro",
        "science_resampling": "within_domain_stratified_macro",
        "holm_scope": holm_scope,
        "holm_scope_datasets_present": sorted({item["dataset"] for item in primary_tests}),
    }


def _pairs(rows, dataset, reference, competitor):
    index = {}
    for row in rows:
        if row.get("dataset") != dataset:
            continue
        key = (str(row["sample_id"]), str(row["method_name"]))
        if key in index:
            raise ValueError(f"Duplicate paired-inference row for {dataset}:{key[0]}:{key[1]}.")
        index[key] = row
    sample_ids = sorted(
        {sample for sample, method in index if method == reference}
        & {sample for sample, method in index if method == competitor}
    )
    return [
        (
            float(index[(sample, reference)].get("score") or 0),
            float(index[(sample, competitor)].get("score") or 0),
            _stratum_for(dataset, index[(sample, reference)]),
        )
        for sample in sample_ids
    ]


def _bootstrap(pairs, *, samples, seed, metric):
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    values = []
    if metric in {
        "task_stratified_micro_accuracy",
        "task_macro_accuracy",
        "domain_macro_accuracy",
        "task_adjusted_harmonic",
    }:
        groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for pair in pairs:
            groups[pair[2]].append(pair)
        tasks = sorted(groups)
        for _ in range(samples):
            left_group_scores = []
            right_group_scores = []
            left_total = 0.0
            right_total = 0.0
            item_total = 0
            for task in tasks:
                group = groups[task]
                draw = [group[rng.randrange(len(group))] for _ in group]
                left_group_scores.append(sum(left for left, _, _ in draw) / len(draw))
                right_group_scores.append(sum(right for _, right, _ in draw) / len(draw))
                left_total += sum(left for left, _, _ in draw)
                right_total += sum(right for _, right, _ in draw)
                item_total += len(draw)
            if metric == "task_adjusted_harmonic":
                values.append(_adjusted_harmonic(left_group_scores) - _adjusted_harmonic(right_group_scores))
            elif metric in {"task_macro_accuracy", "domain_macro_accuracy"}:
                values.append(
                    sum(left_group_scores) / len(left_group_scores)
                    - sum(right_group_scores) / len(right_group_scores)
                )
            else:
                values.append((left_total - right_total) / item_total)
    else:
        for _ in range(samples):
            draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            values.append(sum(left - right for left, right, _ in draw) / len(draw))
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _paired_point(pairs, *, metric: str) -> float:
    if metric in {"micro_accuracy", "task_stratified_micro_accuracy"}:
        return sum(left - right for left, right, _ in pairs) / len(pairs)
    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for pair in pairs:
        groups[pair[2]].append(pair)
    left_scores = [sum(left for left, _, _ in group) / len(group) for group in groups.values()]
    right_scores = [sum(right for _, right, _ in group) / len(group) for group in groups.values()]
    if metric == "task_adjusted_harmonic":
        return _adjusted_harmonic(left_scores) - _adjusted_harmonic(right_scores)
    return sum(left_scores) / len(left_scores) - sum(right_scores) / len(right_scores)


def _primary_metric(dataset: str) -> str:
    if dataset == "bbeh":
        return "task_stratified_micro_accuracy"
    if dataset == "musr":
        return "task_macro_accuracy"
    if dataset in {"gpqa_diamond", "supergpqa", "supergpqa_science"}:
        return "domain_macro_accuracy"
    return "micro_accuracy"


def _stratum_for(dataset: str, row: dict[str, Any]) -> str:
    if dataset in {"bbeh", "musr"}:
        return str(row.get("task") or "unknown")
    if dataset in {"gpqa_diamond", "supergpqa", "supergpqa_science"}:
        return str(row.get("high_level_domain") or row.get("subdomain") or "unknown").casefold()
    return "all"


def _adjusted_harmonic(values: list[float]) -> float:
    """BBEH Full adjusted harmonic: add 1pp, average harmonically, subtract 1pp."""

    if not values:
        return 0.0
    adjusted = [float(value) + 0.01 for value in values]
    return max(0.0, len(adjusted) / sum(1.0 / value for value in adjusted) - 0.01)


def _mcnemar_p(reference_only: int, competitor_only: int) -> float:
    total = reference_only + competitor_only
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, index) for index in range(min(reference_only, competitor_only) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _holm(tests) -> None:
    ordered = sorted(tests, key=lambda row: float(row["mcnemar_exact_p"]))
    running = 0.0
    total = len(ordered)
    for index, row in enumerate(ordered):
        adjusted = min(1.0, (total - index) * float(row["mcnemar_exact_p"]))
        running = max(running, adjusted)
        row["holm_adjusted_p"] = running
