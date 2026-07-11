"""BRD-MAD 报告的预注册配对统计推断。"""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any


def paired_method_statistics(
    prediction_rows: list[dict[str, Any]],
    *,
    reference_method: str = "brd_quorum_3",
    bootstrap_samples: int = 10_000,
    seed: int = 20260711,
    bbeh_harmonic: bool = True,
) -> dict[str, Any]:
    """Paired bootstrap, McNemar, and per-dataset Holm correction.

    BBEH resamples within task so that its official task-balanced metric is not
    accidentally converted to an example-weighted result.  Other datasets use
    ordinary paired item bootstrap.
    """

    keyed: dict[tuple[str, str, str], dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in prediction_rows:
        keyed[(str(row.get("dataset")), str(row.get("model_name")), str(row.get("sample_id")))][str(row.get("method_name"))] = row
    by_dataset_model: dict[tuple[str, str], list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for (dataset, model, _), methods in keyed.items():
        by_dataset_model[(dataset, model)].append(methods)

    tests: list[dict[str, Any]] = []
    for (dataset, model), pairs in by_dataset_model.items():
        competitors = sorted({name for pair in pairs for name in pair if name != reference_method})
        for competitor in competitors:
            usable = [pair for pair in pairs if reference_method in pair and competitor in pair]
            if not usable:
                continue
            deltas = (
                [_bbeh_harmonic_delta(usable, reference_method, competitor)]
                if dataset == "bbeh" and bbeh_harmonic
                else [
                    float(pair[reference_method].get("score") or 0.0)
                    - float(pair[competitor].get("score") or 0.0)
                    for pair in usable
                ]
            )
            bootstrap_dataset = dataset if bbeh_harmonic else ("bbeh_micro" if dataset == "bbeh" else dataset)
            low, high = _bootstrap_ci(usable, reference_method, competitor, dataset=bootstrap_dataset, samples=bootstrap_samples, seed=f"{seed}:{dataset}:{model}:{competitor}")
            b, c = _mcnemar_counts(usable, reference_method, competitor)
            tests.append(
                {
                    "dataset": dataset,
                    "model_name": model,
                    "reference_method": reference_method,
                    "comparison_method": competitor,
                    "paired_question_count": len(usable),
                    "absolute_accuracy_delta": sum(deltas) / len(deltas),
                    "bootstrap_ci_95": [low, high],
                    "mcnemar_b_reference_only_correct": b,
                    "mcnemar_c_comparator_only_correct": c,
                    "mcnemar_exact_p": _mcnemar_exact_p(b, c),
                }
            )
    _holm_by_dataset_model(tests)
    return {
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": seed,
        "reference_method": reference_method,
        "bbeh_resampling": "within_task_stratified" if bbeh_harmonic else "item_micro",
        "tests": tests,
    }


def _bootstrap_ci(
    pairs: list[dict[str, dict[str, Any]]],
    reference: str,
    competitor: str,
    *,
    dataset: str,
    samples: int,
    seed: str,
) -> tuple[float, float]:
    rng = random.Random(seed)
    if not pairs:
        return 0.0, 0.0
    if dataset == "bbeh":
        groups: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
        for pair in pairs:
            task = str(pair[reference].get("task") or "unknown")
            groups[task].append(pair)
        draws = [_bbeh_draw(groups, reference, competitor, rng) for _ in range(samples)]
    else:
        draws = []
        count = len(pairs)
        for _ in range(samples):
            draws.append(sum(_delta(pairs[rng.randrange(count)], reference, competitor) for _ in range(count)) / count)
    draws.sort()
    return _quantile(draws, 0.025), _quantile(draws, 0.975)


def _bbeh_draw(groups, reference: str, competitor: str, rng: random.Random) -> float:
    reference_scores: list[float] = []
    comparator_scores: list[float] = []
    for pairs in groups.values():
        count = len(pairs)
        drawn = [pairs[rng.randrange(count)] for _ in range(count)]
        reference_scores.append(sum(float(pair[reference].get("score") or 0.0) for pair in drawn) / count)
        comparator_scores.append(sum(float(pair[competitor].get("score") or 0.0) for pair in drawn) / count)
    return _harmonic(reference_scores) - _harmonic(comparator_scores)


def _bbeh_harmonic_delta(pairs, reference: str, competitor: str) -> float:
    by_task: dict[str, list[dict[str, dict[str, Any]]]] = defaultdict(list)
    for pair in pairs:
        by_task[str(pair[reference].get("task") or "unknown")].append(pair)
    reference_scores = [
        sum(float(pair[reference].get("score") or 0.0) for pair in values) / len(values)
        for values in by_task.values()
    ]
    comparator_scores = [
        sum(float(pair[competitor].get("score") or 0.0) for pair in values) / len(values)
        for values in by_task.values()
    ]
    return _harmonic(reference_scores) - _harmonic(comparator_scores)


def _harmonic(values: list[float]) -> float:
    if not values or any(value <= 0.0 for value in values):
        return 0.0
    return len(values) / sum(1.0 / value for value in values)


def _delta(pair: dict[str, dict[str, Any]], reference: str, competitor: str) -> float:
    return float(pair[reference].get("score") or 0.0) - float(pair[competitor].get("score") or 0.0)


def _quantile(values: list[float], probability: float) -> float:
    if not values:
        return 0.0
    position = (len(values) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return values[lower]
    return values[lower] + (values[upper] - values[lower]) * (position - lower)


def _mcnemar_counts(pairs, reference: str, competitor: str) -> tuple[int, int]:
    b = c = 0
    for pair in pairs:
        ref = float(pair[reference].get("score") or 0.0) == 1.0
        comp = float(pair[competitor].get("score") or 0.0) == 1.0
        if ref and not comp:
            b += 1
        elif comp and not ref:
            c += 1
    return b, c


def _mcnemar_exact_p(b: int, c: int) -> float:
    total = b + c
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, value) for value in range(0, min(b, c) + 1)) / (2**total)
    return min(1.0, 2.0 * tail)


def _holm_by_dataset_model(tests: list[dict[str, Any]]) -> None:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for test in tests:
        groups[(str(test["dataset"]), str(test["model_name"]))].append(test)
    for group in groups.values():
        ordered = sorted(group, key=lambda item: float(item["mcnemar_exact_p"]))
        adjusted_floor = 0.0
        total = len(ordered)
        for index, test in enumerate(ordered):
            adjusted = min(1.0, (total - index) * float(test["mcnemar_exact_p"]))
            adjusted_floor = max(adjusted_floor, adjusted)
            test["holm_adjusted_p_within_dataset"] = adjusted_floor
