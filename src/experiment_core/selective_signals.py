"""共享的选择性通信信号与 trigger 规则。

本模块把“置信度如何标准化”“何时应该触发通信”这两件事收敛成统一实现，
避免不同实验线各自解释置信度字段，导致结果不可横向比较。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


UNCERTAINTY_TYPE_CHOICES = (
    "none",
    "calculation",
    "evidence_selection",
    "entity_linking",
    "multi_hop",
    "commonsense_gap",
    "format_extraction",
    "other",
)


@dataclass(frozen=True)
class ConfidenceSummary:
    """一组 agent 置信度的聚合统计。"""

    valid_confidences: list[float]
    invalid_agent_ids: list[int]
    any_invalid_confidence: bool
    mean_confidence: float | None
    confidence_spread: float | None


@dataclass(frozen=True)
class TriggerDecision:
    """一次 trigger 规则判断结果。"""

    triggered: bool
    decision_reason: str
    fail_open_applied: bool


@dataclass(frozen=True)
class DivergenceSummary:
    """一组 agent 在答案、claim 与 uncertainty 上的分歧统计。"""

    answer_unique_count: int
    answer_divergence_score: float
    claim_similarity_mean: float
    claim_divergence_score: float
    uncertainty_type_diversity_score: float


def confidence_display(value: object) -> object:
    """保留 `confidence_raw` 的原始展示形式。"""
    if value is None:
        return ""
    return value


def normalize_confidence(raw_value: object) -> tuple[float | None, bool, str]:
    """把 `confidence_raw` 归一化到可比较的 `[0, 1]` 区间。"""
    if raw_value is None:
        return None, False, "missing"
    if isinstance(raw_value, bool):
        return None, False, "invalid_bool"
    if isinstance(raw_value, (int, float)):
        numeric_value = float(raw_value)
    else:
        text = str(raw_value).strip().rstrip("%")
        try:
            numeric_value = float(text)
        except ValueError:
            return None, False, "non_numeric"
    if 0.0 <= numeric_value <= 1.0:
        return round(numeric_value, 6), True, "unit_interval"
    if 1.0 < numeric_value <= 100.0:
        return round(numeric_value / 100.0, 6), True, "percent_scaled"
    return None, False, "out_of_range"


def summarize_confidence_rows(
    rows: list[dict[str, Any]],
    *,
    confidence_value_key: str = "confidence_value",
    confidence_valid_key: str = "confidence_valid",
    confidence_source_key: str = "confidence_source",
    agent_id_key: str = "agent_id",
) -> ConfidenceSummary:
    """从多条 turn 记录中汇总均值、离散度和非法置信度来源。"""
    valid_confidences = [
        float(row[confidence_value_key])
        for row in rows
        if row.get(confidence_valid_key) and row.get(confidence_value_key) is not None
    ]
    invalid_agent_ids = [
        int(row[agent_id_key])
        for row in rows
        if (not row.get(confidence_valid_key)) and row.get(confidence_source_key) != "missing"
    ]
    mean_confidence = round(sum(valid_confidences) / len(valid_confidences), 6) if valid_confidences else None
    if len(valid_confidences) >= 2:
        confidence_spread = round(max(valid_confidences) - min(valid_confidences), 6)
    elif len(valid_confidences) == 1:
        confidence_spread = 0.0
    else:
        confidence_spread = None
    return ConfidenceSummary(
        valid_confidences=valid_confidences,
        invalid_agent_ids=invalid_agent_ids,
        any_invalid_confidence=bool(invalid_agent_ids),
        mean_confidence=mean_confidence,
        confidence_spread=confidence_spread,
    )


def summarize_divergence_rows(
    rows: list[dict[str, Any]],
    *,
    answer_key: str = "normalized_answer",
    claim_span_key: str = "claim_span",
    uncertainty_type_key: str = "uncertainty_type",
) -> DivergenceSummary:
    """从多条 Stage A 记录中汇总答案与代理推理分歧。"""
    sample_count = len(rows)
    denominator = max(1, sample_count - 1)

    answers = [_normalize_answer_label(row.get(answer_key)) for row in rows]
    answer_unique_count = len({value for value in answers if value})
    answer_divergence_score = round(max(0, answer_unique_count - 1) / denominator, 6)

    claim_spans = [_normalize_claim_span(row.get(claim_span_key)) for row in rows]
    claim_similarity_mean = pairwise_token_jaccard_mean(claim_spans)
    claim_divergence_score = round(1.0 - claim_similarity_mean, 6)

    uncertainty_types = [
        normalize_uncertainty_type(row.get(uncertainty_type_key)) or "missing"
        for row in rows
    ]
    uncertainty_unique_count = len(set(uncertainty_types))
    uncertainty_type_diversity_score = round(max(0, uncertainty_unique_count - 1) / denominator, 6)

    return DivergenceSummary(
        answer_unique_count=answer_unique_count,
        answer_divergence_score=answer_divergence_score,
        claim_similarity_mean=claim_similarity_mean,
        claim_divergence_score=claim_divergence_score,
        uncertainty_type_diversity_score=uncertainty_type_diversity_score,
    )


def normalize_uncertainty_type(raw_value: object) -> str | None:
    """把 uncertainty type 归一化到固定枚举。"""
    if raw_value is None:
        return None
    normalized = str(raw_value).strip().lower()
    if not normalized:
        return None
    return normalized if normalized in UNCERTAINTY_TYPE_CHOICES else "other"


def pairwise_token_jaccard_mean(values: list[str]) -> float:
    """计算多条短文本的两两 token-Jaccard 均值。"""
    if len(values) <= 1:
        return 1.0
    token_sets = [_tokenize_for_overlap(value) for value in values]
    pair_scores: list[float] = []
    for left_index in range(len(token_sets)):
        for right_index in range(left_index + 1, len(token_sets)):
            pair_scores.append(_token_jaccard(token_sets[left_index], token_sets[right_index]))
    if not pair_scores:
        return 1.0
    return round(sum(pair_scores) / len(pair_scores), 6)


def decide_trigger(
    *,
    trigger_type: str,
    initial_disagreement: bool,
    answer_divergence_score: float | None = None,
    claim_divergence_score: float | None = None,
    uncertainty_type_diversity_score: float | None = None,
    mean_confidence: float | None = None,
    confidence_spread: float | None = None,
    any_invalid_confidence: bool = False,
    mean_conf_threshold: float | None = None,
    conf_spread_threshold: float | None = None,
    claim_divergence_threshold: float | None = None,
    uncertainty_type_diversity_threshold: float | None = None,
    fail_open_to_always: bool = False,
) -> TriggerDecision:
    """按统一规则判断是否触发通信。"""
    if trigger_type == "always_communicate":
        return TriggerDecision(True, "always_on", False)
    if trigger_type == "disagreement_triggered":
        return TriggerDecision(
            initial_disagreement,
            "answer_disagreement" if initial_disagreement else "early_exit_by_agreement",
            False,
        )
    if trigger_type == "confidence_triggered":
        threshold = mean_conf_threshold if mean_conf_threshold is not None else 0.75
        if mean_confidence is None:
            return TriggerDecision(False, "confidence_unavailable", False)
        triggered = mean_confidence < threshold
        return TriggerDecision(
            triggered,
            "low_mean_confidence" if triggered else "early_exit_by_high_confidence",
            False,
        )
    if trigger_type == "hybrid_trigger":
        if initial_disagreement:
            return TriggerDecision(True, "answer_disagreement", False)
        threshold = mean_conf_threshold if mean_conf_threshold is not None else 0.75
        spread_threshold = conf_spread_threshold if conf_spread_threshold is not None else 0.20
        if mean_confidence is None:
            return TriggerDecision(False, "confidence_unavailable", False)
        if mean_confidence < threshold:
            return TriggerDecision(True, "low_mean_confidence", False)
        if confidence_spread is not None and confidence_spread > spread_threshold:
            return TriggerDecision(True, "high_confidence_spread", False)
        return TriggerDecision(False, "early_exit_by_hybrid_rule", False)
    if trigger_type == "voc_trigger_v2":
        if (answer_divergence_score or 0.0) >= 0.5:
            return TriggerDecision(True, "answer_divergence", False)
        claim_threshold = claim_divergence_threshold if claim_divergence_threshold is not None else 0.55
        uncertainty_threshold = (
            uncertainty_type_diversity_threshold
            if uncertainty_type_diversity_threshold is not None
            else 0.5
        )
        if (
            (claim_divergence_score or 0.0) >= claim_threshold
            and (uncertainty_type_diversity_score or 0.0) >= uncertainty_threshold
        ):
            return TriggerDecision(True, "claim_uncertainty_divergence", False)
        if mean_confidence is not None and mean_confidence < 0.95 and (confidence_spread or 0.0) >= 0.10:
            return TriggerDecision(True, "low_mean_confidence_with_spread", False)
        return TriggerDecision(False, "early_exit_by_voc_rule", False)
    raise ValueError(f"Unsupported trigger type: {trigger_type}")


def decide_trigger_from_policy(
    policy: Any,
    *,
    initial_disagreement: bool,
    answer_divergence_score: float | None,
    claim_divergence_score: float | None,
    uncertainty_type_diversity_score: float | None,
    mean_confidence: float | None,
    confidence_spread: float | None,
    any_invalid_confidence: bool,
) -> TriggerDecision:
    """从策略对象读取参数后复用统一 trigger 规则。"""
    return decide_trigger(
        trigger_type=str(getattr(policy, "trigger_type")),
        initial_disagreement=initial_disagreement,
        answer_divergence_score=answer_divergence_score,
        claim_divergence_score=claim_divergence_score,
        uncertainty_type_diversity_score=uncertainty_type_diversity_score,
        mean_confidence=mean_confidence,
        confidence_spread=confidence_spread,
        any_invalid_confidence=any_invalid_confidence,
        mean_conf_threshold=getattr(policy, "mean_conf_threshold", None),
        conf_spread_threshold=getattr(policy, "conf_spread_threshold", None),
        claim_divergence_threshold=getattr(policy, "claim_divergence_threshold", None),
        uncertainty_type_diversity_threshold=getattr(policy, "uncertainty_type_diversity_threshold", None),
        fail_open_to_always=bool(getattr(policy, "fail_open_to_always", True)),
    )


def _normalize_answer_label(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _normalize_claim_span(value: object) -> str:
    if value is None:
        return ""
    return str(value).strip().lower()


def _tokenize_for_overlap(value: str) -> set[str]:
    cleaned = "".join(character.lower() if character.isalnum() else " " for character in value)
    return {token for token in cleaned.split() if token}


def _token_jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    if not union:
        return 1.0
    return round(len(left & right) / len(union), 6)
