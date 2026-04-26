"""`comm_necessary` 的机制级纯逻辑函数。

这里集中实现 HotpotQA 通信必要性实验的核心逻辑：
supporting facts 归一化、不同消息包强度压缩、答案与证据聚合，
以及联合 answer / support 指标的计算。
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import math
import re
import string
from typing import Any, Iterable

from experiment_core.evaluation import normalize_prediction


METHOD_ORDER = [
    "full_context_single",
    "split_no_comm_mv3",
    "answer_only_exchange",
    "evidence_exchange",
    "full_packet_exchange",
]

PACKET_MODES = ["answer_only", "evidence", "full_packet"]


@dataclass(frozen=True)
class HotpotScores:
    """HotpotQA answer/support/joint 指标。"""

    answer_em: float
    answer_f1: float
    supporting_em: float
    supporting_f1: float
    joint_em: float
    joint_f1: float
    support_title_recall: float
    support_fact_recall: float


def approximate_token_count(text: str) -> int:
    """沿用仓库里的轻量 token 估算，便于稳定统计通信量。"""
    cleaned = text.strip()
    if not cleaned:
        return 0
    return max(1, len(cleaned) // 4)


def trim_text(text: str | None, token_cap: int) -> str:
    """按近似 token cap 截断模型生成的证据文本。"""
    normalized = (text or "").strip()
    if not normalized:
        return ""
    char_cap = max(1, token_cap * 4)
    if len(normalized) <= char_cap:
        return normalized
    return normalized[: max(1, char_cap - 3)].rstrip() + "..."


def majority_vote_with_counts(candidates: Iterable[str]) -> tuple[str, dict[str, int], bool]:
    """多数投票；平票时保留先出现答案，保证可复现。"""
    ordered = [candidate for candidate in candidates if candidate]
    counts = Counter(ordered)
    if not counts:
        return "", {}, False
    winner = max(counts.items(), key=lambda item: (item[1], -ordered.index(item[0])))[0]
    consensus = len(counts) == 1 and bool(ordered)
    return winner, dict(counts), consensus


def normalize_supporting_facts(value: object) -> list[tuple[str, int]]:
    """把模型输出的 supporting facts 归一成 ``(title, sent_id)`` 列表。"""
    facts: list[tuple[str, int]] = []
    if value is None:
        return facts
    raw_items: list[object]
    if isinstance(value, dict):
        titles = value.get("title", [])
        sent_ids = value.get("sent_id", [])
        raw_items = [[title, sent_id] for title, sent_id in zip(_as_list(titles), _as_list(sent_ids), strict=False)]
    elif isinstance(value, list):
        raw_items = value
    else:
        return facts

    for item in raw_items:
        title: object | None = None
        sent_id: object | None = None
        if isinstance(item, dict):
            title = item.get("title")
            sent_id = item.get("sent_id", item.get("sentence_id"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            title, sent_id = item[0], item[1]
        normalized_title = str(title or "").strip()
        if not normalized_title:
            continue
        try:
            normalized_sent_id = int(str(sent_id).strip())
        except (TypeError, ValueError):
            continue
        fact = (normalized_title, normalized_sent_id)
        if fact not in facts:
            facts.append(fact)
    return facts


def gold_supporting_facts(sample_metadata: dict[str, Any]) -> list[tuple[str, int]]:
    """读取 HotpotQA gold supporting facts。"""
    return normalize_supporting_facts(sample_metadata.get("supporting_facts"))


def support_facts_to_jsonable(facts: Iterable[tuple[str, int]]) -> list[list[object]]:
    """导出官方 HotpotQA prediction JSON 需要的列表形状。"""
    return [[title, sent_id] for title, sent_id in facts]


def aggregate_supporting_facts(rows: list[dict[str, Any]], winner_answer: str | None = None) -> list[tuple[str, int]]:
    """聚合候选 supporting facts；优先使用最终答案一致的 agent。"""
    selected_rows = rows
    if winner_answer:
        filtered = [row for row in rows if str(row.get("normalized_answer") or "") == winner_answer]
        if filtered:
            selected_rows = filtered
    counts: Counter[tuple[str, int]] = Counter()
    first_seen: dict[tuple[str, int], int] = {}
    for row_index, row in enumerate(selected_rows):
        for fact in normalize_supporting_facts(row.get("supporting_facts")):
            counts[fact] += 1
            first_seen.setdefault(fact, row_index)
    ordered = sorted(counts, key=lambda fact: (-counts[fact], first_seen[fact], fact[0], fact[1]))
    return ordered[:4]


def build_packet(
    row: dict[str, Any],
    *,
    packet_mode: str,
    token_cap: int,
) -> dict[str, Any]:
    """把 agent 输出压缩成不同通信强度的消息包。"""
    if packet_mode not in PACKET_MODES:
        raise ValueError(f"Unsupported packet_mode: {packet_mode}")
    fields: dict[str, Any] = {
        "final_answer": str(row.get("normalized_answer") or ""),
        "confidence_raw": row.get("confidence_raw"),
    }
    if packet_mode in {"evidence", "full_packet"}:
        fields["evidence_summary"] = trim_text(str(row.get("evidence_summary") or ""), 80)
        fields["supporting_facts"] = support_facts_to_jsonable(normalize_supporting_facts(row.get("supporting_facts")))
    if packet_mode == "full_packet":
        fields["reasoning_trace"] = trim_text(str(row.get("reasoning_trace") or ""), 120)

    fitted = _fit_fields_to_cap(fields, token_cap)
    packet_text = json.dumps(fitted, ensure_ascii=False, sort_keys=True)
    return {
        "agent_id": int(row.get("agent_id") or 0),
        "packet_mode": packet_mode,
        "packet_fields": fitted,
        "packet_text": packet_text,
        "approx_packet_tokens": approximate_token_count(packet_text),
        "token_cap": token_cap,
    }


def score_hotpot_prediction(
    *,
    predicted_answer: str,
    gold_answer: str,
    predicted_supporting_facts: Iterable[tuple[str, int]],
    gold_supporting_facts: Iterable[tuple[str, int]],
) -> HotpotScores:
    """计算 answer、supporting facts 与 joint 指标。"""
    answer_em = 1.0 if normalize_prediction("hotpotqa", predicted_answer) == normalize_prediction("hotpotqa", gold_answer) else 0.0
    answer_precision, answer_recall, answer_f1 = _answer_prf(predicted_answer, gold_answer)
    predicted_set = set(_normalize_fact(fact) for fact in predicted_supporting_facts)
    gold_set = set(_normalize_fact(fact) for fact in gold_supporting_facts)
    supporting_em = 1.0 if predicted_set == gold_set else 0.0
    support_precision, support_recall, support_f1 = _set_prf(predicted_set, gold_set)
    joint_precision = answer_precision * support_precision
    joint_recall = answer_recall * support_recall
    joint_f1 = _safe_f1(joint_precision, joint_recall)
    support_title_recall = _title_recall(predicted_set, gold_set)
    return HotpotScores(
        answer_em=answer_em,
        answer_f1=answer_f1,
        supporting_em=supporting_em,
        supporting_f1=support_f1,
        joint_em=answer_em * supporting_em,
        joint_f1=joint_f1,
        support_title_recall=support_title_recall,
        support_fact_recall=support_recall,
    )


def official_prediction_payload(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """构造 HotpotQA 官方评测脚本兼容的 prediction JSON。"""
    return {
        "answer": {str(row["sample_id"]): str(row.get("prediction") or "") for row in rows},
        "sp": {
            str(row["sample_id"]): support_facts_to_jsonable(normalize_supporting_facts(row.get("supporting_facts")))
            for row in rows
        },
    }


def _fit_fields_to_cap(fields: dict[str, Any], token_cap: int) -> dict[str, Any]:
    """优先删减长文本字段，保留答案与证据坐标。"""
    fitted = {key: value for key, value in fields.items() if _has_content(value)}
    text_keys = ["reasoning_trace", "evidence_summary"]
    while approximate_token_count(json.dumps(fitted, ensure_ascii=False, sort_keys=True)) > token_cap:
        trimmed = False
        for key in text_keys:
            value = fitted.get(key)
            if isinstance(value, str) and value:
                next_value = value[:-12].rstrip()
                fitted[key] = next_value + "..." if next_value else ""
                trimmed = True
                if approximate_token_count(json.dumps(fitted, ensure_ascii=False, sort_keys=True)) <= token_cap:
                    break
        fitted = {key: value for key, value in fitted.items() if _has_content(value)}
        if not trimmed:
            break
    return fitted


def _has_content(value: object) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(value.strip())
    if isinstance(value, list):
        return bool(value)
    return True


def _as_list(value: object) -> list[object]:
    if isinstance(value, list):
        return value
    return [value]


def _normalize_answer_for_f1(value: str) -> list[str]:
    lowered = value.lower()
    lowered = re.sub(r"\b(a|an|the)\b", " ", lowered)
    lowered = lowered.translate(str.maketrans("", "", string.punctuation))
    return " ".join(lowered.split()).split()


def _answer_prf(predicted: str, gold: str) -> tuple[float, float, float]:
    predicted_tokens = _normalize_answer_for_f1(predicted)
    gold_tokens = _normalize_answer_for_f1(gold)
    if not predicted_tokens and not gold_tokens:
        return 1.0, 1.0, 1.0
    if not predicted_tokens or not gold_tokens:
        return 0.0, 0.0, 0.0
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0, 0.0, 0.0
    precision = overlap / len(predicted_tokens)
    recall = overlap / len(gold_tokens)
    return precision, recall, _safe_f1(precision, recall)


def _set_prf(predicted: set[tuple[str, int]], gold: set[tuple[str, int]]) -> tuple[float, float, float]:
    if not predicted and not gold:
        return 1.0, 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0, 0.0
    overlap = len(predicted & gold)
    precision = overlap / len(predicted)
    recall = overlap / len(gold)
    return precision, recall, _safe_f1(precision, recall)


def _safe_f1(precision: float, recall: float) -> float:
    if math.isclose(precision + recall, 0.0):
        return 0.0
    return 2 * precision * recall / (precision + recall)


def _normalize_fact(fact: tuple[str, int]) -> tuple[str, int]:
    return (fact[0].strip(), int(fact[1]))


def _title_recall(predicted: set[tuple[str, int]], gold: set[tuple[str, int]]) -> float:
    gold_titles = {title for title, _ in gold}
    if not gold_titles:
        return 0.0
    predicted_titles = {title for title, _ in predicted}
    return len(predicted_titles & gold_titles) / len(gold_titles)

