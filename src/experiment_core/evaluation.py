"""输出归一化与任务级打分逻辑。"""

from __future__ import annotations

import math
import re
import string
from collections import Counter
from typing import Iterable


def normalize_prediction(dataset: str, final_answer: str) -> str:
    """按数据集类型把模型答案归一化到可比较的形式。"""
    if dataset == "gsm8k":
        return normalize_number(final_answer)
    if dataset == "strategyqa":
        return normalize_yes_no(final_answer)
    if dataset == "hotpotqa":
        return normalize_text(final_answer)
    raise ValueError(f"Unsupported dataset {dataset}")


def normalize_gold(dataset: str, answer: str) -> str:
    """金标答案沿用与预测值一致的归一化规则。"""
    return normalize_prediction(dataset, answer)


def score_prediction(dataset: str, predicted: str, gold: str) -> float:
    """当前项目统一采用精确匹配，命中记 1.0，否则记 0.0。"""
    return 1.0 if normalize_prediction(dataset, predicted) == normalize_gold(dataset, gold) else 0.0


def aggregate_majority(candidates: Iterable[str]) -> tuple[str, dict[str, int]]:
    """聚合同一题的多次回答，并在平票时保持先出现者优先。"""
    ordered = [candidate for candidate in candidates if candidate]
    counts = Counter(ordered)
    if not counts:
        return "", {}
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    return winner, dict(counts)


def normalize_number(value: str) -> str:
    """把数值答案清洗成稳定字符串，避免 ``1`` 与 ``1.0`` 被视作不同。"""
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
    """把多种 yes / no 变体收敛成标准二元标签。"""
    lowered = value.strip().lower()
    if lowered.startswith("yes"):
        return "yes"
    if lowered.startswith("no"):
        return "no"
    return lowered


def normalize_text(value: str) -> str:
    """对文本答案做轻量归一化，近似常见 QA 的 EM 预处理。"""
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = " ".join(lowered.split())
    return lowered
