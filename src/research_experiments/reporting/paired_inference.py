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
    bbeh_harmonic: bool = False,
) -> dict[str, Any]:
    tests: list[dict[str, Any]] = []
    datasets = sorted({str(row["dataset"]) for row in rows})
    for dataset in datasets:
        for competitor in competitors:
            pairs = _pairs(rows, dataset, reference, competitor)
            if not pairs:
                continue
            use_task_harmonic = bbeh_harmonic and dataset == "bbeh"
            point = _paired_point(pairs, task_harmonic=use_task_harmonic)
            low, high = _bootstrap(
                pairs,
                samples=bootstrap_samples,
                seed=f"{seed}:{dataset}:{competitor}",
                stratified=use_task_harmonic,
            )
            reference_only = sum(left > right for left, right, _ in pairs)
            competitor_only = sum(right > left for left, right, _ in pairs)
            tests.append(
                {
                    "dataset": dataset,
                    "reference_method": reference,
                    "comparison_method": competitor,
                    "paired_question_count": len(pairs),
                    "mean_accuracy_delta": point,
                    "accuracy_metric": "task_harmonic" if use_task_harmonic else "micro_accuracy",
                    "bootstrap_ci_95": [low, high],
                    "mcnemar_b_reference_only_correct": reference_only,
                    "mcnemar_c_comparator_only_correct": competitor_only,
                    "mcnemar_exact_p": _mcnemar_p(reference_only, competitor_only),
                }
            )
    primary_tests = [item for item in tests if item["dataset"] in {"bbeh", "gpqa_diamond"}]
    _holm(primary_tests)
    return {
        "reference_method": reference,
        "bootstrap_samples": bootstrap_samples,
        "tests": tests,
        "bbeh_resampling": "within_task_stratified_harmonic" if bbeh_harmonic else "item_micro",
        "holm_scope": ["bbeh", "gpqa_diamond"],
    }


def _pairs(rows, dataset, reference, competitor):
    index = {
        (str(row["sample_id"]), str(row["method_name"])): row
        for row in rows
        if row.get("dataset") == dataset
    }
    sample_ids = sorted(
        {sample for sample, method in index if method == reference}
        & {sample for sample, method in index if method == competitor}
    )
    return [
        (
            float(index[(sample, reference)].get("score") or 0),
            float(index[(sample, competitor)].get("score") or 0),
            str(index[(sample, reference)].get("task") or "unknown"),
        )
        for sample in sample_ids
    ]


def _bootstrap(pairs, *, samples, seed, stratified):
    rng = random.Random(int(hashlib.sha256(seed.encode()).hexdigest()[:16], 16))
    values = []
    if stratified:
        groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
        for pair in pairs:
            groups[pair[2]].append(pair)
        tasks = sorted(groups)
        for _ in range(samples):
            left_task_scores = []
            right_task_scores = []
            for task in tasks:
                group = groups[task]
                draw = [group[rng.randrange(len(group))] for _ in group]
                left_task_scores.append(sum(left for left, _, _ in draw) / len(draw))
                right_task_scores.append(sum(right for _, right, _ in draw) / len(draw))
            values.append(_harmonic(left_task_scores) - _harmonic(right_task_scores))
    else:
        for _ in range(samples):
            draw = [pairs[rng.randrange(len(pairs))] for _ in pairs]
            values.append(sum(left - right for left, right, _ in draw) / len(draw))
    values.sort()
    return values[int(0.025 * (len(values) - 1))], values[int(0.975 * (len(values) - 1))]


def _paired_point(pairs, *, task_harmonic: bool) -> float:
    if not task_harmonic:
        return sum(left - right for left, right, _ in pairs) / len(pairs)
    groups: dict[str, list[tuple[float, float, str]]] = defaultdict(list)
    for pair in pairs:
        groups[pair[2]].append(pair)
    left_scores = [sum(left for left, _, _ in group) / len(group) for group in groups.values()]
    right_scores = [sum(right for _, right, _ in group) / len(group) for group in groups.values()]
    return _harmonic(left_scores) - _harmonic(right_scores)


def _harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


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

