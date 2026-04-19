"""数据集特定的解析兜底逻辑。"""

from __future__ import annotations

import re
from typing import Any


def extract_fallback_answer(dataset: str, raw_text: str) -> tuple[dict[str, Any], str] | None:
    """当 JSON 解析失败时，尝试按数据集规则直接提取最终答案。"""
    text = raw_text.strip()
    if not text:
        return None
    if dataset == "gsm8k":
        answer = _extract_gsm8k_answer(text)
        if answer is None:
            return None
        return {"reasoning": text, "final_answer": answer}, "dataset_fallback_gsm8k"
    if dataset == "strategyqa":
        answer = _extract_strategyqa_answer(text)
        if answer is None:
            return None
        return {"reasoning": text, "final_answer": answer}, "dataset_fallback_strategyqa"
    if dataset == "hotpotqa":
        answer = _extract_hotpotqa_answer(text)
        if answer is None:
            return None
        return {"reasoning": text, "final_answer": answer}, "dataset_fallback_hotpotqa"
    return None


def _extract_gsm8k_answer(text: str) -> str | None:
    """从 GSM8K 风格自然语言解题过程里抽取最后数字。"""
    boxed_matches = re.findall(r"boxed\{([^}]*)\}", text, flags=re.IGNORECASE)
    if boxed_matches:
        candidate = _extract_last_number(boxed_matches[-1])
        if candidate is not None:
            return candidate

    final_patterns = [
        r"(?:final answer|answer is|so[, ]+the answer is)\D*([-+]?\d[\d,]*(?:\.\d+)?)",
        r"=\s*([-+]?\d[\d,]*(?:\.\d+)?)\s*(?:dollars?|eggs?|hours?|miles?)?\s*$",
    ]
    for pattern in final_patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).replace(",", "")

    return _extract_last_number(text)


def _extract_strategyqa_answer(text: str) -> str | None:
    """从尾部结论中抽取 yes / no。"""
    lowered = text.lower()
    if re.search(r"\bthe answer is yes\b", lowered) or re.search(r"\byes\b", lowered[-40:]):
        return "yes"
    if re.search(r"\bthe answer is no\b", lowered) or re.search(r"\bno\b", lowered[-40:]):
        return "no"
    return None


def _extract_hotpotqa_answer(text: str) -> str | None:
    """从 HotpotQA 风格自由文本中截取最终短答案。"""
    patterns = [
        r"(?:final answer|answer is|therefore[, ]+the answer is)\s*[:\-]?\s*(.+)$",
        r"^(.+)$",
    ]
    for pattern in patterns:
        match = re.search(pattern, text.strip(), flags=re.IGNORECASE | re.MULTILINE)
        if match:
            candidate = match.group(1).strip()
            candidate = candidate.strip(" .")
            if candidate:
                return candidate
    return None


def _extract_last_number(text: str) -> str | None:
    """返回文本中最后一个数值片段。"""
    matches = re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", text)
    if not matches:
        return None
    return matches[-1].replace(",", "")
