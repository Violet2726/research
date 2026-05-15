"""预测归一化与任务级打分逻辑。

本模块刻意保持轻量，只实现当前仓库需要的答案归一化和精确匹配打分，
以便所有实验线共享同一套“答案如何比较”的基础规则。
"""

from __future__ import annotations

import json
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
    if dataset == "tabfact":
        return normalize_tabfact_label(final_answer)
    if dataset in {
        "hotpotqa",
        "webquestions",
        "grailqa",
        "wikitq",
        "dog_webquestions",
        "dog_grailqa",
        "dog_webqsp",
        "dog_cwq",
        "dog_metaqa_1hop",
        "dog_metaqa_2hop",
        "dog_metaqa_3hop",
    }:
        return normalize_text(final_answer)
    if dataset in {"mmlu_pro", "gpqa_diamond"}:
        return normalize_multiple_choice(final_answer)
    raise ValueError(f"Unsupported dataset {dataset}")


def normalize_gold(dataset: str, answer: str) -> str:
    """对金标答案沿用与预测值一致的归一化规则。"""
    if dataset in {
        "webquestions",
        "grailqa",
        "wikitq",
        "dog_webquestions",
        "dog_grailqa",
        "dog_webqsp",
        "dog_cwq",
        "dog_metaqa_1hop",
        "dog_metaqa_2hop",
        "dog_metaqa_3hop",
    }:
        answers = _decode_text_answer_set_gold(answer)
        return normalize_text(answers[0]) if answers else ""
    return normalize_prediction(dataset, answer)


def score_prediction(dataset: str, predicted: str, gold: str) -> float:
    """计算单题得分。

    当前仓库统一采用精确匹配：归一化后完全一致记为 `1.0`，否则记为 `0.0`。
    """
    if dataset in {"mmlu_pro", "gpqa_diamond"}:
        return score_multiple_choice(predicted, gold)
    if dataset in {"webquestions", "grailqa"}:
        return score_text_answer_set(predicted, gold)
    if dataset == "wikitq":
        return score_wikitq_answer_set(predicted, gold)
    if dataset == "tabfact":
        return 1.0 if normalize_tabfact_label(predicted) == normalize_tabfact_label(gold) else 0.0
    if dataset in {
        "dog_webquestions",
        "dog_grailqa",
        "dog_webqsp",
        "dog_cwq",
        "dog_metaqa_1hop",
        "dog_metaqa_2hop",
        "dog_metaqa_3hop",
    }:
        return score_text_answer_alias_exact(predicted, gold)
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


def normalize_tabfact_label(value: str) -> str:
    """把 TabFact 的真假标签归一成 `entailed / refuted`。"""

    lowered = normalize_text(value)
    if lowered in {"entailed", "entail", "true", "yes", "supported", "correct"}:
        return "entailed"
    if lowered in {"refuted", "refute", "false", "no", "unsupported", "incorrect"}:
        return "refuted"
    return lowered


def normalize_text(value: str) -> str:
    """对文本答案做轻量归一化，近似常见 QA 任务的 EM 预处理。"""
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    lowered = " ".join(lowered.split())
    return lowered


def normalize_multiple_choice(value: str) -> str:
    """把选择题答案归一成选项字母或标准化后的选项文本。"""
    normalized = value.strip()
    if not normalized:
        return ""
    match = re.search(r"\b([A-J])\b", normalized.upper())
    if match:
        return match.group(1)
    return normalize_text(normalized)


def normalize_math_expression(value: str) -> str:
    """对短数学表达式做轻量归一化，尽量保留判题所需语义。"""
    normalized = value.strip().lower()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("\\left", "").replace("\\right", "")
    normalized = normalized.replace("\\!", "")
    normalized = normalized.replace(" ", "")
    normalized = normalized.replace("{", "").replace("}", "")
    normalized = normalized.rstrip(".")
    return normalized


def score_multiple_choice(predicted: str, gold: str) -> float:
    """选择题允许命中字母选项，或命中对应选项文本。"""
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


def score_text_answer_set(predicted: str, gold: str) -> float:
    """在多个文本别名上取最大 token-F1，更贴近图问答常见评测口径。"""
    predicted_norm = normalize_text(predicted)
    if not predicted_norm:
        return 0.0
    gold_aliases = [normalize_text(answer) for answer in _decode_text_answer_set_gold(gold) if normalize_text(answer)]
    if not gold_aliases:
        return 0.0
    return round(max(_token_f1(predicted_norm, gold_alias) for gold_alias in gold_aliases), 6)


def score_text_answer_alias_exact(predicted: str, gold: str) -> float:
    """按 DoG 官方脚本的近似方式比较文本答案集合。

    官方实现会把生成答案整体转成小写并去空格，只要任一金标别名是其子串就算命中。
    这里保留这一判定口径，用于高保真论文主线。
    """

    normalized_prediction = re.sub(r"\s+", "", str(predicted or "").lower())
    if not normalized_prediction:
        return 0.0
    gold_aliases = [
        re.sub(r"\s+", "", normalize_text(answer))
        for answer in _decode_text_answer_set_gold(gold)
        if normalize_text(answer)
    ]
    if any(alias and alias in normalized_prediction for alias in gold_aliases):
        return 1.0
    return 0.0


def score_wikitq_answer_set(predicted: str, gold: str) -> float:
    """按 WikiTQ 的答案集合口径比较最终答案。"""

    gold_answers = [normalize_text(item) for item in _decode_text_answer_set_gold(gold) if normalize_text(item)]
    predicted_answers = [normalize_text(item) for item in _decode_predicted_answer_set(predicted, gold_answers) if normalize_text(item)]
    if not gold_answers or not predicted_answers:
        return 0.0
    return 1.0 if set(predicted_answers) == set(gold_answers) else 0.0


def _decode_text_answer_set_gold(gold: str) -> list[str]:
    stripped = str(gold or "").strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(payload, list):
            return [str(item).strip() for item in payload if str(item).strip()]
    return [stripped]


def _decode_predicted_answer_set(predicted: str, gold_answers: list[str]) -> list[str]:
    stripped = str(predicted or "").strip()
    if not stripped:
        return []

    cleaned = stripped.replace("\r", "\n")
    if "|" in cleaned:
        return [item.strip() for item in cleaned.split("|") if item.strip()]
    if ";" in cleaned:
        return [item.strip() for item in cleaned.split(";") if item.strip()]
    if "\n" in cleaned:
        return [item.strip() for item in cleaned.splitlines() if item.strip()]
    if len(gold_answers) > 1 and "," in cleaned:
        return [item.strip() for item in cleaned.split(",") if item.strip()]
    return [cleaned]


def _token_f1(predicted: str, gold: str) -> float:
    predicted_tokens = predicted.split()
    gold_tokens = gold.split()
    if not predicted_tokens or not gold_tokens:
        return 0.0
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)
