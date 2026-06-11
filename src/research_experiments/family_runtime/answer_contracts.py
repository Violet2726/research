"""共享答案合约、paper transcript 抽取与 repair 执行辅助。"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.core.execution.cache import RequestCache
from research_experiments.core.execution.providers import OpenAICompatibleProvider
from research_experiments.core.execution.rate_limits import RequestThrottle
from research_experiments.core.execution.runner_common import (
    CachedRequestResult,
    CachedTurnResult,
    TurnRequestExecutor,
    execute_cached_request,
    execute_cached_turn,
)
from research_experiments.core.prompts.dataset_contracts import (
    build_json_system_prompt,
    dataset_instruction_for_sample,
)
from research_experiments.core.structured_outputs import SCHEMA_ANSWER_CORE, validate_structured_output

JSON_ANSWER_CORE_CONTRACT = "json_answer_core"
PAPER_TRANSCRIPT_HARDENED_CONTRACT = "paper_transcript_hardened"

AnswerContract = Literal["json_answer_core", "paper_transcript_hardened"]
AnswerExtractionStatus = Literal["ok", "failed", "not_attempted", "not_applicable"]

_MATH_DATASETS = {"gsm8k", "math500", "competition_math"}
_MULTIPLE_CHOICE_DATASETS = {"gpqa_diamond", "mmlu", "mmlu_abstract_algebra", "mmlu_pro"}


@dataclass(frozen=True)
class TranscriptAnswerExtraction:
    status: AnswerExtractionStatus
    validated_output: dict[str, Any]
    source: str | None
    error: str | None
    raw_output_incomplete_suspected: bool


@dataclass(frozen=True)
class AnswerContractTurnResult:
    payload: dict[str, Any]
    prompt_hash: str
    cache_key: str
    cache_hit: bool
    response_payload: dict[str, Any]
    request_error: str | None
    request_status: str
    output_status: str
    validated_output: dict[str, Any]
    usage: dict[str, Any]
    raw_usage: dict[str, Any]
    repair_usage: dict[str, Any]
    repair_latency_ms: float
    answer_extraction_status: AnswerExtractionStatus
    answer_extraction_source: str | None
    answer_extraction_error: str | None
    raw_output_incomplete_suspected: bool
    repair_call_used: bool
    repair_output_status: str | None
    repair_request_error: str | None
    request_count: int
    cache_request_count: int
    network_request_count: int
    raw_finish_reason: str | None
    repair_cache_hit: bool
    repair_request_started_at: str | None


def answer_contract_for_prompt_version(prompt_version: str) -> AnswerContract:
    if prompt_version == "multi_agent_paper_text":
        return PAPER_TRANSCRIPT_HARDENED_CONTRACT
    if prompt_version in {"multi_agent_controlled_json", "multi_agent_debate_json"}:
        return JSON_ANSWER_CORE_CONTRACT
    raise ValueError(f"Unsupported vanilla MAD prompt_version: {prompt_version}")


def validate_prompt_answer_contract(prompt_version: str, answer_contract: str) -> AnswerContract:
    normalized = str(answer_contract or "").strip()
    expected = answer_contract_for_prompt_version(prompt_version)
    if normalized not in {JSON_ANSWER_CORE_CONTRACT, PAPER_TRANSCRIPT_HARDENED_CONTRACT}:
        raise ValueError(
            f"Unsupported answer_contract {answer_contract!r}. "
            f"Expected one of {[JSON_ANSWER_CORE_CONTRACT, PAPER_TRANSCRIPT_HARDENED_CONTRACT]}."
        )
    if normalized != expected:
        raise ValueError(
            f"prompt_version={prompt_version!r} requires answer_contract={expected!r}, got {normalized!r}."
        )
    return expected


def execute_answer_contract_turn(
    *,
    backbone,
    provider: OpenAICompatibleProvider,
    cache: RequestCache,
    throttle: RequestThrottle | None,
    sample: DatasetSample | None,
    messages: list[dict[str, str]],
    temperature: float,
    top_p: float,
    seed: int | None,
    dataset: str,
    answer_contract: AnswerContract,
    use_response_format: bool,
    allow_network_repair: bool,
    request_executor: TurnRequestExecutor | None = None,
    repair_request_executor: TurnRequestExecutor | None = None,
) -> AnswerContractTurnResult:
    if answer_contract == JSON_ANSWER_CORE_CONTRACT:
        turn = execute_cached_turn(
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
            seed=seed,
            schema_id=SCHEMA_ANSWER_CORE,
            dataset=dataset,
            use_response_format=use_response_format,
            request_executor=request_executor,
        )
        request_status = "request_fail" if turn.request_error else "ok"
        answer_status: AnswerExtractionStatus
        if turn.request_error:
            answer_status = "not_attempted"
        else:
            answer_status = "ok" if turn.output_status == "ok" else "failed"
        return AnswerContractTurnResult(
            payload=turn.payload,
            prompt_hash=turn.prompt_hash,
            cache_key=turn.cache_key,
            cache_hit=turn.cache_hit,
            response_payload=turn.response_payload,
            request_error=turn.request_error,
            request_status=request_status,
            output_status=turn.output_status,
            validated_output=dict(turn.validated_output),
            usage=dict(turn.usage),
            raw_usage=dict(turn.usage),
            repair_usage={},
            repair_latency_ms=0.0,
            answer_extraction_status=answer_status,
            answer_extraction_source="json_answer_core",
            answer_extraction_error=None if turn.output_status == "ok" else "json_answer_core_validation_failed",
            raw_output_incomplete_suspected=False,
            repair_call_used=False,
            repair_output_status=None,
            repair_request_error=None,
            request_count=1,
            cache_request_count=1 if turn.cache_hit else 0,
            network_request_count=0 if turn.cache_hit else 1,
            raw_finish_reason=_raw_finish_reason(turn.response_payload),
            repair_cache_hit=False,
            repair_request_started_at=None,
        )
    if answer_contract != PAPER_TRANSCRIPT_HARDENED_CONTRACT:
        raise ValueError(f"Unsupported answer contract: {answer_contract}")

    raw_request = execute_cached_request(
        backbone=backbone,
        provider=provider,
        cache=cache,
        throttle=throttle,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        seed=seed,
        use_response_format=use_response_format,
        request_executor=request_executor,
    )
    request_status = "request_fail" if raw_request.request_error else "ok"
    if raw_request.request_error:
        return AnswerContractTurnResult(
            payload=raw_request.payload,
            prompt_hash=raw_request.prompt_hash,
            cache_key=raw_request.cache_key,
            cache_hit=raw_request.cache_hit,
            response_payload=raw_request.response_payload,
            request_error=raw_request.request_error,
            request_status=request_status,
            output_status="request_fail",
            validated_output={},
            usage=dict(raw_request.usage),
            raw_usage=dict(raw_request.usage),
            repair_usage={},
            repair_latency_ms=0.0,
            answer_extraction_status="not_attempted",
            answer_extraction_source=None,
            answer_extraction_error=None,
            raw_output_incomplete_suspected=False,
            repair_call_used=False,
            repair_output_status=None,
            repair_request_error=None,
            request_count=1,
            cache_request_count=1 if raw_request.cache_hit else 0,
            network_request_count=0 if raw_request.cache_hit else 1,
            raw_finish_reason=_raw_finish_reason(raw_request.response_payload),
            repair_cache_hit=False,
            repair_request_started_at=None,
        )

    extraction = extract_paper_transcript_answer(
        str(raw_request.response_payload.get("assistant_text") or ""),
        dataset=dataset,
    )
    should_repair = allow_network_repair and sample is not None and (
        extraction.status != "ok" or extraction.raw_output_incomplete_suspected
    )
    repair_result: CachedTurnResult | None = None
    if should_repair:
        repair_result = execute_cached_turn(
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            messages=build_paper_answer_repair_messages(
                sample=sample,
                transcript=str(raw_request.response_payload.get("assistant_text") or ""),
            ),
            temperature=0.0,
            top_p=1.0,
            seed=_repair_seed(seed),
            schema_id=SCHEMA_ANSWER_CORE,
            dataset=dataset,
            use_response_format=True,
            request_executor=repair_request_executor,
        )

    validated_output: dict[str, Any] = {}
    output_status = "schema_fail"
    answer_status: AnswerExtractionStatus = extraction.status
    answer_source = extraction.source
    answer_error = extraction.error
    if extraction.status == "ok" and not extraction.raw_output_incomplete_suspected:
        validated_output = dict(extraction.validated_output)
        output_status = "ok"
        answer_status = "ok"
    elif repair_result is not None and repair_result.output_status == "ok":
        validated_output = dict(repair_result.validated_output)
        output_status = "ok"
        answer_status = "ok"
        answer_source = "repair_json_answer_core"
        answer_error = None
    elif extraction.status == "ok":
        validated_output = dict(extraction.validated_output)
        output_status = "ok"
        answer_status = "ok"

    repair_usage = dict(repair_result.usage) if repair_result is not None else {}
    combined_usage = _merge_usage(dict(raw_request.usage), repair_usage)
    cache_request_count = (1 if raw_request.cache_hit else 0) + (
        1 if repair_result is not None and repair_result.cache_hit else 0
    )
    request_count = 1 + (1 if repair_result is not None else 0)
    return AnswerContractTurnResult(
        payload=raw_request.payload,
        prompt_hash=raw_request.prompt_hash,
        cache_key=raw_request.cache_key,
        cache_hit=raw_request.cache_hit and (repair_result is None or repair_result.cache_hit),
        response_payload=raw_request.response_payload,
        request_error=raw_request.request_error,
        request_status=request_status,
        output_status=output_status,
        validated_output=validated_output,
        usage=combined_usage,
        raw_usage=dict(raw_request.usage),
        repair_usage=repair_usage,
        repair_latency_ms=float(repair_result.response_payload.get("latency_ms") or 0.0) if repair_result is not None else 0.0,
        answer_extraction_status=answer_status,
        answer_extraction_source=answer_source,
        answer_extraction_error=answer_error if output_status != "ok" else None,
        raw_output_incomplete_suspected=extraction.raw_output_incomplete_suspected,
        repair_call_used=repair_result is not None,
        repair_output_status=repair_result.output_status if repair_result is not None else None,
        repair_request_error=repair_result.request_error if repair_result is not None else None,
        request_count=request_count,
        cache_request_count=cache_request_count,
        network_request_count=request_count - cache_request_count,
        raw_finish_reason=_raw_finish_reason(raw_request.response_payload),
        repair_cache_hit=repair_result.cache_hit if repair_result is not None else False,
        repair_request_started_at=(
            str(repair_result.response_payload.get("request_started_at") or "")
            if repair_result is not None and repair_result.response_payload.get("request_started_at")
            else None
        ),
    )


def refresh_answer_contract_turn(
    *,
    row: dict[str, Any],
    sample: DatasetSample | None,
    backbone,
    provider: OpenAICompatibleProvider | None,
    cache: RequestCache | None,
    throttle: RequestThrottle | None,
    answer_contract: AnswerContract,
    allow_network_repair: bool,
    repair_request_executor: TurnRequestExecutor | None = None,
) -> AnswerContractTurnResult:
    """Re-resolve one persisted turn row under the current answer-contract logic."""

    if answer_contract == JSON_ANSWER_CORE_CONTRACT:
        usage = _usage_from_row(row, "")
        response_payload = {
            "assistant_text": str(row.get("assistant_text") or ""),
            "provider_reasoning_text": str(row.get("provider_reasoning_text") or ""),
            "finish_reason": row.get("raw_finish_reason"),
            "latency_ms": float(row.get("raw_latency_ms") or row.get("latency_ms") or 0.0),
            "request_started_at": row.get("request_started_at"),
        }
        request_error = str(row.get("request_error") or "") or None
        request_status = str(row.get("request_status") or ("request_fail" if request_error else "ok"))
        output_status = str(row.get("output_status") or ("request_fail" if request_error else "schema_fail"))
        answer_status: AnswerExtractionStatus
        if request_error:
            answer_status = "not_attempted"
        else:
            answer_status = "ok" if output_status == "ok" else "failed"
        return AnswerContractTurnResult(
            payload=dict(row.get("payload") or {}),
            prompt_hash=str(row.get("prompt_hash") or ""),
            cache_key="",
            cache_hit=bool(row.get("cache_hit")),
            response_payload=response_payload,
            request_error=request_error,
            request_status=request_status,
            output_status=output_status,
            validated_output=dict(row.get("validated_output") or {}),
            usage=usage,
            raw_usage=usage,
            repair_usage={},
            repair_latency_ms=0.0,
            answer_extraction_status=answer_status,
            answer_extraction_source=str(row.get("answer_extraction_source") or "json_answer_core"),
            answer_extraction_error=str(row.get("answer_extraction_error") or "") or None,
            raw_output_incomplete_suspected=bool(row.get("raw_output_incomplete_suspected")),
            repair_call_used=False,
            repair_output_status=None,
            repair_request_error=None,
            request_count=max(1, int(row.get("request_count") or 1)),
            cache_request_count=max(0, int(row.get("cache_request_count") or (1 if row.get("cache_hit") else 0))),
            network_request_count=max(0, int(row.get("network_request_count") or 0)),
            raw_finish_reason=str(row.get("raw_finish_reason")) if row.get("raw_finish_reason") is not None else None,
            repair_cache_hit=False,
            repair_request_started_at=None,
        )
    if answer_contract != PAPER_TRANSCRIPT_HARDENED_CONTRACT:
        raise ValueError(f"Unsupported answer contract: {answer_contract}")

    request_error = str(row.get("request_error") or "") or None
    raw_usage = _usage_from_row(row, "raw_")
    if not raw_usage:
        raw_usage = _usage_from_row(row, "")
    response_payload = {
        "assistant_text": str(row.get("assistant_text") or ""),
        "provider_reasoning_text": str(row.get("provider_reasoning_text") or ""),
        "finish_reason": row.get("raw_finish_reason"),
        "latency_ms": float(row.get("raw_latency_ms") or row.get("latency_ms") or 0.0),
        "request_started_at": row.get("request_started_at"),
    }
    if request_error:
        return AnswerContractTurnResult(
            payload=dict(row.get("payload") or {}),
            prompt_hash=str(row.get("prompt_hash") or ""),
            cache_key="",
            cache_hit=bool(row.get("cache_hit")),
            response_payload=response_payload,
            request_error=request_error,
            request_status="request_fail",
            output_status="request_fail",
            validated_output={},
            usage=raw_usage,
            raw_usage=raw_usage,
            repair_usage={},
            repair_latency_ms=0.0,
            answer_extraction_status="not_attempted",
            answer_extraction_source=None,
            answer_extraction_error=None,
            raw_output_incomplete_suspected=False,
            repair_call_used=False,
            repair_output_status=None,
            repair_request_error=None,
            request_count=1,
            cache_request_count=1 if row.get("cache_hit") else 0,
            network_request_count=0 if row.get("cache_hit") else 1,
            raw_finish_reason=str(row.get("raw_finish_reason")) if row.get("raw_finish_reason") is not None else None,
            repair_cache_hit=False,
            repair_request_started_at=None,
        )

    extraction = extract_paper_transcript_answer(
        str(row.get("assistant_text") or ""),
        dataset=str(row.get("dataset") or ""),
    )
    repair_result: CachedTurnResult | None = None
    should_repair = (
        allow_network_repair
        and sample is not None
        and provider is not None
        and cache is not None
        and (extraction.status != "ok" or extraction.raw_output_incomplete_suspected)
    )
    if should_repair:
        payload = dict(row.get("payload") or {})
        seed = payload.get("seed")
        repair_result = execute_cached_turn(
            backbone=backbone,
            provider=provider,
            cache=cache,
            throttle=throttle,
            messages=build_paper_answer_repair_messages(
                sample=sample,
                transcript=str(row.get("assistant_text") or ""),
            ),
            temperature=0.0,
            top_p=1.0,
            seed=_repair_seed(int(seed) if isinstance(seed, int) else None),
            schema_id=SCHEMA_ANSWER_CORE,
            dataset=str(row.get("dataset") or ""),
            use_response_format=True,
            request_executor=repair_request_executor,
        )

    validated_output: dict[str, Any] = {}
    output_status = "schema_fail"
    answer_status: AnswerExtractionStatus = extraction.status
    answer_source = extraction.source
    answer_error = extraction.error
    if extraction.status == "ok" and not extraction.raw_output_incomplete_suspected:
        validated_output = dict(extraction.validated_output)
        output_status = "ok"
        answer_status = "ok"
    elif repair_result is not None and repair_result.output_status == "ok":
        validated_output = dict(repair_result.validated_output)
        output_status = "ok"
        answer_status = "ok"
        answer_source = "repair_json_answer_core"
        answer_error = None
    elif extraction.status == "ok":
        validated_output = dict(extraction.validated_output)
        output_status = "ok"
        answer_status = "ok"

    repair_usage = dict(repair_result.usage) if repair_result is not None else {}
    combined_usage = _merge_usage(raw_usage, repair_usage)
    cache_request_count = (1 if row.get("cache_hit") else 0) + (
        1 if repair_result is not None and repair_result.cache_hit else 0
    )
    request_count = 1 + (1 if repair_result is not None else 0)
    return AnswerContractTurnResult(
        payload=dict(row.get("payload") or {}),
        prompt_hash=str(row.get("prompt_hash") or ""),
        cache_key="",
        cache_hit=bool(row.get("cache_hit")) and (repair_result is None or repair_result.cache_hit),
        response_payload=response_payload,
        request_error=None,
        request_status="ok",
        output_status=output_status,
        validated_output=validated_output,
        usage=combined_usage,
        raw_usage=raw_usage,
        repair_usage=repair_usage,
        repair_latency_ms=float(repair_result.response_payload.get("latency_ms") or 0.0) if repair_result is not None else 0.0,
        answer_extraction_status=answer_status,
        answer_extraction_source=answer_source,
        answer_extraction_error=answer_error if output_status != "ok" else None,
        raw_output_incomplete_suspected=extraction.raw_output_incomplete_suspected,
        repair_call_used=repair_result is not None,
        repair_output_status=repair_result.output_status if repair_result is not None else None,
        repair_request_error=repair_result.request_error if repair_result is not None else None,
        request_count=request_count,
        cache_request_count=cache_request_count,
        network_request_count=request_count - cache_request_count,
        raw_finish_reason=str(row.get("raw_finish_reason")) if row.get("raw_finish_reason") is not None else None,
        repair_cache_hit=repair_result.cache_hit if repair_result is not None else False,
        repair_request_started_at=(
            str(repair_result.response_payload.get("request_started_at") or "")
            if repair_result is not None and repair_result.response_payload.get("request_started_at")
            else None
        ),
    )


def extract_paper_transcript_answer(transcript: str, *, dataset: str) -> TranscriptAnswerExtraction:
    text = str(transcript or "").strip()
    if not text:
        return TranscriptAnswerExtraction(
            status="failed",
            validated_output={},
            source=None,
            error="assistant transcript is empty",
            raw_output_incomplete_suspected=True,
        )

    candidate: str | None = None
    source: str | None = None
    if dataset in _MATH_DATASETS:
        candidate, source = _extract_math_candidate(text)
    elif dataset in _MULTIPLE_CHOICE_DATASETS:
        candidate, source = _extract_multiple_choice_candidate(text)
    elif dataset == "hotpotqa":
        candidate, source = _extract_span_candidate(text)
    elif dataset == "strategyqa":
        candidate, source = _extract_yes_no_candidate(text)
    else:
        candidate, source = _extract_terminal_answer_line(text)

    incomplete = _suspect_incomplete_paper_output(text, dataset=dataset, extracted_candidate=candidate)
    if candidate is None:
        return TranscriptAnswerExtraction(
            status="failed",
            validated_output={},
            source=None,
            error="no explicit answer signal found in paper transcript",
            raw_output_incomplete_suspected=incomplete,
        )
    if dataset in _MATH_DATASETS:
        candidate = _normalize_explicit_math_candidate(candidate)
    try:
        validated = validate_structured_output(
            json.dumps({"final_answer": candidate}, ensure_ascii=False),
            SCHEMA_ANSWER_CORE,
            dataset=dataset,
        )
    except Exception as exc:
        return TranscriptAnswerExtraction(
            status="failed",
            validated_output={},
            source=source,
            error=str(exc),
            raw_output_incomplete_suspected=incomplete,
        )
    return TranscriptAnswerExtraction(
        status="ok",
        validated_output=validated,
        source=source,
        error=None,
        raw_output_incomplete_suspected=incomplete,
    )


def build_paper_answer_repair_messages(
    *,
    sample: DatasetSample,
    transcript: str,
) -> list[dict[str, str]]:
    user_parts = [
        "You are repairing a paper-style debate transcript into one judgeable final answer.",
        dataset_instruction_for_sample(sample, hotpot_style="short_span"),
        f"Question:\n{sample.question.strip()}",
    ]
    if sample.prompt_context:
        user_parts.append(f"Context:\n{sample.prompt_context}")
    user_parts.append(f"Transcript:\n{transcript.strip()}")
    user_parts.append(
        'Return exactly one JSON object with key "final_answer". '
        'You may optionally include "reasoning". '
        "Do not quote the transcript. "
        "Do not invent a new solution path. "
        "Extract or conservatively infer the final answer that should be scored."
    )
    return [
        {
            "role": "system",
            "content": build_json_system_prompt(
                "You normalize free-form reasoning transcripts into exact final answers for evaluation.",
                extra_rules=[
                    "Return JSON only.",
                    "Do not use markdown fences.",
                    "If the transcript names an intermediate entity instead of the requested answer type, map it to the requested final answer.",
                    "For multiple-choice datasets, final_answer must be only the option letter.",
                ],
            ),
        },
        {"role": "user", "content": "\n\n".join(user_parts)},
    ]


def _extract_math_candidate(text: str) -> tuple[str | None, str | None]:
    boxed = _extract_last_boxed_answer(text)
    if boxed:
        return boxed, "explicit_boxed_answer"
    candidate, source = _extract_terminal_answer_line(text)
    if candidate is None:
        return None, None
    if re.search(r"\d", candidate) or any(token in candidate for token in (r"\frac", r"\sqrt", r"\pi")):
        return candidate, source
    return None, None


def _extract_multiple_choice_candidate(text: str) -> tuple[str | None, str | None]:
    for line in reversed(_tail_lines(text)):
        direct = re.fullmatch(r"\(?([A-J])\)?[.)]?", line.strip().upper())
        if direct:
            return direct.group(1), "explicit_terminal_option"
        labeled = re.search(
            r"(?i)(?:final answer|answer|option|choice)\s*(?:is|:)?\s*\(?([A-J])\)?",
            line,
        )
        if labeled:
            return labeled.group(1).upper(), "explicit_multiple_choice_line"
    return None, None


def _extract_span_candidate(text: str) -> tuple[str | None, str | None]:
    return _extract_terminal_answer_line(text)


def _extract_yes_no_candidate(text: str) -> tuple[str | None, str | None]:
    for line in reversed(_tail_lines(text)):
        stripped = line.strip().lower().rstrip(".")
        if stripped in {"yes", "no"}:
            return stripped, "explicit_yes_no_terminal"
        labeled = re.search(r"(?i)(?:final answer|answer)\s*(?:is|:)\s*(yes|no)\b", line)
        if labeled:
            return labeled.group(1).lower(), "explicit_yes_no_line"
    return None, None


def _extract_terminal_answer_line(text: str) -> tuple[str | None, str | None]:
    for line in reversed(_tail_lines(text)):
        match = re.search(r"(?i)(?:final answer|answer)\s*(?:is|:)\s*(.+)$", line)
        if not match:
            continue
        candidate = _clean_terminal_answer(match.group(1))
        if candidate:
            return candidate, "explicit_answer_line"
    return None, None


def _tail_lines(text: str, *, limit: int = 6) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-limit:]


def _extract_last_boxed_answer(text: str) -> str | None:
    matches: list[str] = []
    token = r"\boxed{"
    start = 0
    while True:
        index = text.find(token, start)
        if index < 0:
            break
        cursor = index + len(token)
        depth = 1
        collected: list[str] = []
        while cursor < len(text):
            char = text[cursor]
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    matches.append("".join(collected).strip())
                    start = cursor + 1
                    break
            collected.append(char)
            cursor += 1
        else:
            break
    return matches[-1] if matches else None


def _suspect_incomplete_paper_output(
    text: str,
    *,
    dataset: str,
    extracted_candidate: str | None,
) -> bool:
    stripped = text.rstrip()
    if not stripped:
        return True
    if stripped.count("```") % 2 == 1:
        return True
    if r"\boxed{" in stripped and _extract_last_boxed_answer(stripped) is None:
        return True
    if stripped.count(r"\(") > stripped.count(r"\)"):
        return True
    if stripped.count(r"\[") > stripped.count(r"\]"):
        return True
    if dataset in _MATH_DATASETS and extracted_candidate is None and stripped.endswith(("\\", "=", ":", ",", "/", "+", "-", "*")):
        return True
    if dataset in {"hotpotqa", "strategyqa"} and extracted_candidate is None and stripped.endswith(("is", "are", "the", "a", "an", ":")):
        return True
    return False


def _clean_terminal_answer(value: str) -> str:
    cleaned = value.strip().strip("`").strip()
    cleaned = cleaned.rstrip(".")
    cleaned = cleaned.strip()
    if cleaned.startswith("(") and cleaned.endswith(")") and len(cleaned) > 2:
        return cleaned[1:-1].strip()
    return cleaned.strip("\"'")


def _normalize_explicit_math_candidate(value: str) -> str:
    candidate = str(value or "").strip()
    candidate = candidate.replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    match = re.fullmatch(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", candidate)
    if match:
        numerator = match.group(1).strip()
        denominator = match.group(2).strip()
        return f"{numerator}/{denominator}"
    return candidate


def _repair_seed(seed: int | None) -> int | None:
    if seed is None:
        return None
    return seed + 900_000


def _merge_usage(raw_usage: dict[str, Any], repair_usage: dict[str, Any]) -> dict[str, Any]:
    if not repair_usage:
        return dict(raw_usage)
    merged = dict(raw_usage)
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        merged[key] = float(raw_usage.get(key) or 0.0) + float(repair_usage.get(key) or 0.0)
    return merged


def _raw_finish_reason(response_payload: dict[str, Any]) -> str | None:
    value = response_payload.get("finish_reason")
    return str(value) if value is not None else None


def _usage_from_row(row: dict[str, Any], prefix: str) -> dict[str, Any]:
    prompt = row.get(f"{prefix}prompt_tokens")
    completion = row.get(f"{prefix}completion_tokens")
    total = row.get(f"{prefix}total_tokens")
    if prompt is None and completion is None and total is None:
        return {}
    return {
        "prompt_tokens": float(prompt or 0.0),
        "completion_tokens": float(completion or 0.0),
        "total_tokens": float(total or 0.0),
    }
