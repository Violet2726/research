"""HotpotQA 上下文绑定 span 证书校验。"""

from __future__ import annotations

import re
import time
from typing import Any

from research_experiments.families.cred_v.certificates.types import CertificateValidation

_NON_ANSWERS = {"unknown", "not stated", "not specified", "not in context", "n/a", "none"}


def verify_hotpot_certificate(
    *,
    context: str,
    raw_context: dict[str, Any] | None = None,
    leader_answer: str,
    payload: dict[str, Any],
) -> CertificateValidation:
    started = time.perf_counter()
    certificate_type = str(payload.get("certificate_type") or "").strip()
    answer = str(payload.get("answer") or payload.get("final_answer") or "").strip()
    evidence_span = str(payload.get("evidence_span") or "").strip()
    missing_tokens = [str(item).strip() for item in payload.get("missing_tokens", []) if str(item).strip()]
    source_title = str(payload.get("source_title") or "").strip()
    try:
        source_sentence_index = int(payload.get("source_sentence_index"))
    except (TypeError, ValueError):
        source_sentence_index = -1

    def result(valid: bool, *, failure_reason: str = "") -> CertificateValidation:
        return CertificateValidation(
            valid=valid,
            certificate_type=certificate_type,
            normalized_answer=answer,
            challenger_pass=valid,
            leader_pass=False if valid else _normalize(leader_answer) == _normalize(answer),
            failure_reason=failure_reason,
            checker_runtime_ms=round((time.perf_counter() - started) * 1000.0, 6),
        )

    if certificate_type != "context_span_completion":
        return result(False, failure_reason="unsupported_certificate_type")
    if not answer or not evidence_span or not missing_tokens or not source_title or source_sentence_index < 0:
        return result(False, failure_reason="incomplete_certificate")
    if _normalize(answer) in _NON_ANSWERS or _normalize(evidence_span) != _normalize(answer):
        return result(False, failure_reason="non_answer_or_span_mismatch")
    if len(re.findall(re.escape(answer), context, flags=re.IGNORECASE)) != 1:
        return result(False, failure_reason="span_not_unique")
    source_sentence = _declared_source_sentence(
        context=context,
        raw_context=raw_context,
        source_title=source_title,
        sentence_index=source_sentence_index,
    )
    if source_sentence is None or len(re.findall(re.escape(answer), source_sentence, flags=re.IGNORECASE)) != 1:
        return result(False, failure_reason="declared_source_mismatch")
    leader_tokens = _tokens(leader_answer)
    answer_tokens = _tokens(answer)
    missing = _tokens(" ".join(missing_tokens))
    if not leader_tokens or not _is_contiguous_subsequence(leader_tokens, answer_tokens):
        return result(False, failure_reason="not_strict_completion")
    actual_extra = list(answer_tokens)
    start = _subsequence_start(leader_tokens, actual_extra)
    del actual_extra[start : start + len(leader_tokens)]
    if sorted(actual_extra) != sorted(missing):
        return result(False, failure_reason="missing_token_mismatch")
    return result(True)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", _normalize(value))


def _subsequence_start(needle: list[str], haystack: list[str]) -> int:
    for index in range(0, len(haystack) - len(needle) + 1):
        if haystack[index : index + len(needle)] == needle:
            return index
    return -1


def _is_contiguous_subsequence(needle: list[str], haystack: list[str]) -> bool:
    return _subsequence_start(needle, haystack) >= 0 and len(needle) < len(haystack)


def _declared_source_sentence(
    *,
    context: str,
    raw_context: dict[str, Any] | None,
    source_title: str,
    sentence_index: int,
) -> str | None:
    if isinstance(raw_context, dict):
        titles = list(raw_context.get("title") or [])
        paragraphs = list(raw_context.get("sentences") or [])
        for title, sentences in zip(titles, paragraphs, strict=False):
            if _normalize(title) != _normalize(source_title):
                continue
            sentence_list = list(sentences or [])
            if 0 <= sentence_index < len(sentence_list):
                return str(sentence_list[sentence_index])
            return None
        return None

    title_prefix = f"[{source_title}]"
    for line in str(context or "").splitlines():
        if not line.lower().startswith(title_prefix.lower()):
            continue
        paragraph = line[len(title_prefix) :].strip()
        sentences = [item.strip() for item in re.split(r"(?<=[.!?])\s+", paragraph) if item.strip()]
        return sentences[sentence_index] if 0 <= sentence_index < len(sentences) else None
    return None
