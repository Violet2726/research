"""预测归一化与任务级打分逻辑。

本模块刻意保持轻量，只实现当前仓库需要的答案归一化和精确匹配打分，
以便所有实验线共享同一套“答案如何比较”的基础规则。
"""

from __future__ import annotations

import json
import math
import subprocess
import re
import string
import sys
import tempfile
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
    if dataset == "commongen_hard":
        return normalize_commongen_sentence(final_answer)
    if dataset in {
        "realmistake_math_problem_generation",
        "realmistake_fine_grained_fact_verification",
        "realmistake_answerability_classification",
    }:
        return normalize_error_detection_verdict(final_answer)
    if dataset in {
        "hotpotqa",
        "webquestions",
        "grailqa",
        "wikitq",
        "webquestions_paper_test",
        "grailqa_test",
        "webqsp",
        "cwq",
        "metaqa_1hop",
        "metaqa_2hop",
        "metaqa_3hop",
    }:
        return normalize_text(final_answer)
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        return normalize_multiple_choice(final_answer)
    if dataset == "mmlu":
        return normalize_multiple_choice(final_answer)
    if dataset == "humaneval":
        return normalize_code_completion(final_answer)
    raise ValueError(f"Unsupported dataset {dataset}")


def normalize_gold(dataset: str, answer: str) -> str:
    """对金标答案沿用与预测值一致的归一化规则。"""
    if dataset in {
        "webquestions",
        "grailqa",
        "wikitq",
        "webquestions_paper_test",
        "grailqa_test",
        "webqsp",
        "cwq",
        "metaqa_1hop",
        "metaqa_2hop",
        "metaqa_3hop",
    }:
        answers = _decode_text_answer_set_gold(answer)
        return normalize_text(answers[0]) if answers else ""
    return normalize_prediction(dataset, answer)


def score_prediction(dataset: str, predicted: str, gold: str) -> float:
    """计算单题得分。

    当前仓库统一采用精确匹配：归一化后完全一致记为 `1.0`，否则记为 `0.0`。
    """
    if dataset in {"mmlu_pro", "gpqa_diamond", "mmlu_abstract_algebra"}:
        return score_multiple_choice(predicted, gold)
    if dataset in {"webquestions", "grailqa", "grailqa_test"}:
        return score_text_answer_set(predicted, gold)
    if dataset == "wikitq":
        return score_wikitq_answer_set(predicted, gold)
    if dataset == "tabfact":
        return 1.0 if normalize_tabfact_label(predicted) == normalize_tabfact_label(gold) else 0.0
    if dataset == "commongen_hard":
        return score_commongen_hard(predicted, gold)
    if dataset in {
        "realmistake_math_problem_generation",
        "realmistake_fine_grained_fact_verification",
        "realmistake_answerability_classification",
    }:
        return 1.0 if normalize_error_detection_verdict(predicted) == normalize_error_detection_verdict(gold) else 0.0
    if dataset == "humaneval":
        return score_humaneval(predicted, gold)
    if dataset in {
        "webquestions_paper_test",
        "webqsp",
        "cwq",
        "metaqa_1hop",
        "metaqa_2hop",
        "metaqa_3hop",
    }:
        return score_text_answer_alias_exact(predicted, gold)
    if dataset == "mmlu":
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


def normalize_tabfact_label(value: str) -> str:
    """把 TabFact 的真假标签归一成 `entailed / refuted`。"""

    lowered = normalize_text(value)
    if lowered in {"entailed", "entail", "true", "yes", "supported", "correct"}:
        return "entailed"
    if lowered in {"refuted", "refute", "false", "no", "unsupported", "incorrect"}:
        return "refuted"
    return lowered


def normalize_error_detection_verdict(value: str) -> str:
    """把错误检测标签归一成 `contains_error / contains_no_error`。"""

    lowered = normalize_text(value)
    if lowered in {
        "error",
        "contains error",
        "containserror",
        "contains an error",
        "the model response contains an error",
        "therefore the model response contains an error",
        "incorrect",
        "has error",
    }:
        return "contains_error"
    if lowered in {
        "no error",
        "noerror",
        "contains no error",
        "containsnoerror",
        "contains no errors",
        "contains no mistakes",
        "the model response contains no error",
        "therefore the model response contains no error",
        "correct",
    }:
        return "contains_no_error"
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


def normalize_commongen_sentence(value: str) -> str:
    """把 CommonGen 类生成题的答案清洗成稳定句子。"""

    cleaned = re.sub(r"\s+", " ", _strip_code_fences(value).strip())
    return cleaned.strip().strip("\"'")


def normalize_code_completion(value: str) -> str:
    """把 HumanEval 代码补全整理成可执行的补全文本。"""

    cleaned = _strip_code_fences(str(value or ""))
    return cleaned.replace("\r\n", "\n").rstrip() + ("\n" if cleaned.strip() else "")


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


def score_commongen_hard(predicted: str, gold: str) -> float:
    """用概念覆盖率作为 CommonGen-Hard 的稳定主指标。"""

    normalized_prediction = normalize_commongen_sentence(predicted).lower()
    if not normalized_prediction:
        return 0.0
    try:
        payload = json.loads(gold)
    except json.JSONDecodeError:
        return 0.0
    concept_set = payload.get("concept_set") or []
    normalized_concepts = [_normalize_commongen_concept(item) for item in concept_set]
    normalized_concepts = [item for item in normalized_concepts if item]
    if not normalized_concepts:
        return 0.0
    hits = sum(1 for concept in normalized_concepts if concept in normalized_prediction)
    return round(hits / len(normalized_concepts), 6)


def score_humaneval(predicted: str, gold: str) -> float:
    """在本地 Python 子进程里执行 HumanEval 用例，返回 pass@1。"""

    try:
        payload = json.loads(gold)
    except json.JSONDecodeError:
        return 0.0

    prompt = str(payload.get("prompt") or "")
    test_code = str(payload.get("test") or "")
    entry_point = str(payload.get("entry_point") or "").strip()
    completion = normalize_code_completion(predicted)
    if not prompt or not test_code or not entry_point or not completion.strip():
        return 0.0

    program = "\n".join(
        [
            prompt.rstrip("\n"),
            completion.rstrip("\n"),
            "",
            test_code.strip("\n"),
            "",
            f"check({entry_point})",
            "",
        ]
    )
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".py", delete=False) as handle:
        handle.write(program)
        temp_path = handle.name
    try:
        completed = subprocess.run(
            [sys.executable, temp_path],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except Exception:
        return 0.0
    finally:
        try:
            import os

            os.unlink(temp_path)
        except OSError:
            pass
    return 1.0 if completed.returncode == 0 else 0.0


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


def _normalize_commongen_concept(value: str) -> str:
    concept = str(value or "").strip().lower()
    concept = re.sub(r"_[nv]$", "", concept)
    concept = concept.replace("_", " ")
    return normalize_text(concept)


def _strip_code_fences(value: str) -> str:
    text = str(value or "")
    fence_match = re.search(r"```(?:python)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
    if fence_match:
        return fence_match.group(1).strip("\r\n")
    return text.strip("\r\n")
