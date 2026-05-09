"""预测归一化与任务级打分逻辑。

本模块刻意保持轻量，只实现当前仓库需要的答案归一化和精确匹配打分，
以便所有实验线共享同一套“答案如何比较”的基础规则。
"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable


def normalize_prediction(dataset: str, final_answer: str) -> str:
    """按数据集类型把模型答案归一化为可比较的形式。"""
    if dataset in {"gsm8k", "gsm_symbolic"}:
        return normalize_number(final_answer)
    if dataset == "math500":
        return normalize_math_expression(final_answer)
    if dataset == "strategyqa":
        return normalize_yes_no(final_answer)
    if dataset == "hotpotqa":
        return normalize_text(final_answer)
    if dataset in {"mmlu_pro", "gpqa_diamond"}:
        return normalize_multiple_choice(final_answer)
    raise ValueError(f"Unsupported dataset {dataset}")


def normalize_gold(dataset: str, answer: str) -> str:
    """对金标答案沿用与预测值一致的归一化规则。"""
    return normalize_prediction(dataset, answer)


def score_prediction(dataset: str, predicted: str, gold: str) -> float:
    """计算单题得分。

    当前仓库统一采用精确匹配：归一化后完全一致记为 `1.0`，否则记为 `0.0`。
    """
    if dataset in {"mmlu_pro", "gpqa_diamond"}:
        return score_multiple_choice(predicted, gold)
    return 1.0 if normalize_prediction(dataset, predicted) == normalize_gold(dataset, gold) else 0.0


def aggregate_majority(candidates: Iterable[str]) -> tuple[str, dict[str, int]]:
    """聚合同一题的多次回答，并在平票时保持“先出现者优先”。"""
    ordered = [candidate for candidate in candidates if candidate]
    counts = Counter(ordered)
    if not counts:
        return "", {}
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    return winner, dict(counts)


def normalize_number(value: str) -> str:
    """把数值答案清洗成稳定字符串，避免 `1` 与 `1.0` 被视为不同。"""
    match = re.search(r"[-+]?\d[\d,]*(?:\.\d+)?", value.replace(",", ""))
    if not match:
        return value.strip().lower()
    numeric = match.group(0).replace(",", "")
    try:
        as_float = float(numeric)
    except ValueError:
        return numeric
    if math.isclose(as_float, round(as_float)):
        return str(int(round(as_float)))
    return str(as_float).rstrip("0").rstrip(".")


def normalize_yes_no(value: str) -> str:
    """把多种 `yes / no` 变体收敛成标准二元标签。"""
    lowered = value.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return lowered


def normalize_text(value: str) -> str:
    """对文本答案做轻量归一化，近似常见 QA 任务的 EM 预处理。"""
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = " ".join(lowered.split())
    return lowered


def normalize_multiple_choice(value: str) -> str:
    """Normalize MCQ answers to either an option letter or normalized option text."""
    normalized = value.strip()
    if not normalized:
        return ""
    match = re.search(r"\b([A-J])\b", normalized.upper())
    if match:
        return match.group(1)
    return normalize_text(normalized)


def normalize_math_expression(value: str) -> str:
    """Lightweight normalization for short mathematical expressions."""
    normalized = value.strip().lower()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.rstrip(".")
    return normalized


def score_multiple_choice(predicted: str, gold: str) -> float:
    """Accept either the option letter or the exact option text for MCQ benchmarks."""
    predicted_norm = normalize_multiple_choice(predicted)
    gold_letter, gold_text = _decode_multiple_choice_gold(gold)
    accepted = {gold_letter}
    if gold_text:
        accepted.add(normalize_text(gold_text))
    return 1.0 if predicted_norm in accepted else 0.0


def _decode_multiple_choice_gold(gold: str) -> tuple[str, str]:
    if "|||" in gold:
        letter, text = gold.split("|||", 1)
        return normalize_multiple_choice(letter), text.strip()
    normalized = normalize_multiple_choice(gold)
    return normalized, ""
