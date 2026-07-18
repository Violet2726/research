"""CATCH-v3 索引化成对分歧审计原语。

The selector in this protocol never copies text or invents a diagnostic
question.  It can only point at deterministic, candidate-scoped evidence
units.  Blinded witnesses observe the selected text but never candidate
identity, vote counts, answer labels, or the full traces.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from dataclasses import asdict, dataclass
from typing import Any

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.algorithms import (
    DecodeDecision,
    StageDecision,
)


@dataclass(frozen=True)
class EvidenceUnit:
    candidate_key: str
    unit_id: str
    start: int
    end: int
    text: str
    sha256: str
    eligible: bool
    ineligible_reason: str | None = None


@dataclass(frozen=True)
class TargetPair:
    pair_id: str
    anchor_key: str
    challenger_key: str
    left_candidate_key: str
    right_candidate_key: str


@dataclass(frozen=True)
class ContrastCoordinate:
    contrast_id: str
    pair_id: str
    anchor_key: str
    challenger_key: str
    left_candidate_key: str
    right_candidate_key: str
    left_unit_ids: tuple[str, ...]
    right_unit_ids: tuple[str, ...]
    left_text: str
    right_text: str
    left_span: tuple[int, int]
    right_span: tuple[int, int]
    sha256: str


@dataclass(frozen=True)
class ContrastValidation:
    coordinates: tuple[ContrastCoordinate, ...]
    dropped: tuple[dict[str, str], ...]
    protocol_error: str | None
    leakage_count: int
    eligible_challengers: tuple[str, ...]


@dataclass(frozen=True)
class IcvWitnessPacket:
    panel_index: int
    contrasts: tuple[dict[str, str], ...]
    public_to_internal: dict[str, str]
    public_left_to_candidate: dict[str, str]
    public_right_to_candidate: dict[str, str]


@dataclass(frozen=True)
class IcvWitnessParseResult:
    top_level_valid: bool
    observations: dict[str, str]
    expected_coordinate_count: int
    valid_coordinate_count: int
    decisive_coordinate_count: int
    erased_rows: tuple[dict[str, str], ...]


def build_target_pairs(
    stage: StageDecision,
    *,
    seed: int,
    sample_id: str,
    max_challengers: int = 2,
) -> tuple[TargetPair, ...]:
    """Target the plurality anchor and at most two strongest challengers."""

    if not stage.anchor_key:
        return ()
    challengers = [candidate.key for candidate in stage.candidates if candidate.key != stage.anchor_key]
    pairs: list[TargetPair] = []
    for index, challenger_key in enumerate(challengers[:max_challengers]):
        values = [stage.anchor_key, challenger_key]
        random.Random(_stable_hash(seed, sample_id, f"target-pair:{index}")).shuffle(values)
        pairs.append(
            TargetPair(
                pair_id=f"P{index}",
                anchor_key=stage.anchor_key,
                challenger_key=challenger_key,
                left_candidate_key=values[0],
                right_candidate_key=values[1],
            )
        )
    return tuple(pairs)


def segment_stage_evidence(
    sample: DatasetSample,
    stage: StageDecision,
) -> dict[str, tuple[EvidenceUnit, ...]]:
    return {
        candidate.key: segment_reasoning_evidence(
            sample,
            candidate_key=candidate.key,
            answer=candidate.answer,
            reasoning=candidate.representative_reasoning,
        )
        for candidate in stage.candidates
    }


def segment_reasoning_evidence(
    sample: DatasetSample,
    *,
    candidate_key: str,
    answer: str,
    reasoning: str,
) -> tuple[EvidenceUnit, ...]:
    """Create stable 24--512 character evidence units with normalized offsets."""

    source = _normalize(reasoning)
    spans = _sentence_spans_v3(source)
    spans = _split_long_spans_v3(source, spans, maximum=512)
    spans = _merge_short_spans(source, spans, minimum=24, maximum=512)
    units: list[EvidenceUnit] = []
    for index, (start, end) in enumerate(spans):
        text = source[start:end].strip()
        if not text:
            continue
        adjusted_start = source.find(text, start, end)
        adjusted_end = adjusted_start + len(text)
        reason = _evidence_ineligible_reason(sample, text=text, answer=answer)
        units.append(
            EvidenceUnit(
                candidate_key=candidate_key,
                unit_id=f"E{index}",
                start=adjusted_start,
                end=adjusted_end,
                text=text,
                sha256=_sha256(text),
                eligible=reason is None,
                ineligible_reason=reason,
            )
        )
    return tuple(units)


def selector_public_payload(
    pairs: tuple[TargetPair, ...],
    evidence: dict[str, tuple[EvidenceUnit, ...]],
) -> list[dict[str, Any]]:
    """Render pair-local IDs so an anchor has no shared public identity."""

    rendered: list[dict[str, Any]] = []
    for pair in pairs:
        rendered.append(
            {
                "pair_id": pair.pair_id,
                "left_evidence": [
                    {"id": f"L:{unit.unit_id}", "text": unit.text}
                    for unit in evidence.get(pair.left_candidate_key, ())
                    if unit.eligible
                ],
                "right_evidence": [
                    {"id": f"R:{unit.unit_id}", "text": unit.text}
                    for unit in evidence.get(pair.right_candidate_key, ())
                    if unit.eligible
                ],
            }
        )
    return rendered


def validate_contrast_selector(
    payload: dict[str, Any] | None,
    *,
    pairs: tuple[TargetPair, ...],
    evidence: dict[str, tuple[EvidenceUnit, ...]],
    max_per_pair: int = 3,
    max_total: int = 6,
) -> ContrastValidation:
    """Resolve selector IDs into exact trace spans and reject reused evidence."""

    if not isinstance(payload, dict) or set(payload) != {"contrasts"}:
        return ContrastValidation((), (), "selector_top_level_schema_failure", 0, ())
    rows = payload.get("contrasts")
    if not isinstance(rows, list):
        return ContrastValidation((), (), "selector_contrasts_must_be_a_list", 0, ())
    if len(rows) > max_total:
        return ContrastValidation((), (), "selector_too_many_contrasts", 0, ())

    pair_by_id = {pair.pair_id: pair for pair in pairs}
    unit_maps = {
        key: {unit.unit_id: unit for unit in units}
        for key, units in evidence.items()
    }
    accepted: list[ContrastCoordinate] = []
    dropped: list[dict[str, str]] = []
    pair_counts: dict[str, int] = {}
    seen_ids: set[str] = set()
    used_units: dict[str, set[str]] = {}
    leakage_count = 0

    for index, raw in enumerate(rows):
        fallback_id = f"index:{index}"
        if not isinstance(raw, dict):
            dropped.append({"contrast_id": fallback_id, "reason": "contrast_must_be_an_object"})
            continue
        contrast_id = str(raw.get("contrast_id") or fallback_id)
        if set(raw) != {"pair_id", "contrast_id", "left_unit_ids", "right_unit_ids"}:
            dropped.append({"contrast_id": contrast_id, "reason": "contrast_unknown_or_missing_fields"})
            continue
        pair_id = str(raw.get("pair_id") or "")
        pair = pair_by_id.get(pair_id)
        if pair is None:
            dropped.append({"contrast_id": contrast_id, "reason": "unknown_pair_id"})
            continue
        if not re.fullmatch(r"C[0-9]+", contrast_id) or contrast_id in seen_ids:
            dropped.append({"contrast_id": contrast_id, "reason": "invalid_or_duplicate_contrast_id"})
            continue
        if pair_counts.get(pair_id, 0) >= max_per_pair:
            dropped.append({"contrast_id": contrast_id, "reason": "too_many_contrasts_for_pair"})
            continue
        left = _resolve_group(
            raw.get("left_unit_ids"),
            side="L",
            candidate_key=pair.left_candidate_key,
            unit_maps=unit_maps,
        )
        right = _resolve_group(
            raw.get("right_unit_ids"),
            side="R",
            candidate_key=pair.right_candidate_key,
            unit_maps=unit_maps,
        )
        reason = left[0] or right[0]
        if reason is not None:
            if "leak" in reason or "ineligible" in reason:
                leakage_count += 1
            dropped.append({"contrast_id": contrast_id, "reason": reason})
            continue
        left_units = left[1]
        right_units = right[1]
        assert left_units is not None and right_units is not None
        reused = any(
            unit.unit_id in used_units.get(unit.candidate_key, set())
            for unit in (*left_units, *right_units)
        )
        if reused:
            dropped.append({"contrast_id": contrast_id, "reason": "overlapping_or_reused_evidence"})
            continue
        left_text = _group_text(left_units)
        right_text = _group_text(right_units)
        if _normalize(left_text).casefold() == _normalize(right_text).casefold():
            dropped.append({"contrast_id": contrast_id, "reason": "identical_contrast_sides"})
            continue
        for unit in (*left_units, *right_units):
            used_units.setdefault(unit.candidate_key, set()).add(unit.unit_id)
        seen_ids.add(contrast_id)
        pair_counts[pair_id] = pair_counts.get(pair_id, 0) + 1
        canonical_payload = {
            "contrast_id": contrast_id,
            "pair_id": pair_id,
            "left_unit_ids": [unit.unit_id for unit in left_units],
            "right_unit_ids": [unit.unit_id for unit in right_units],
            "left_text": left_text,
            "right_text": right_text,
        }
        accepted.append(
            ContrastCoordinate(
                contrast_id=contrast_id,
                pair_id=pair_id,
                anchor_key=pair.anchor_key,
                challenger_key=pair.challenger_key,
                left_candidate_key=pair.left_candidate_key,
                right_candidate_key=pair.right_candidate_key,
                left_unit_ids=tuple(unit.unit_id for unit in left_units),
                right_unit_ids=tuple(unit.unit_id for unit in right_units),
                left_text=left_text,
                right_text=right_text,
                left_span=(left_units[0].start, left_units[-1].end),
                right_span=(right_units[0].start, right_units[-1].end),
                sha256=_sha256(json.dumps(canonical_payload, ensure_ascii=False, sort_keys=True)),
            )
        )

    eligible = tuple(
        pair.challenger_key
        for pair in pairs
        if pair_counts.get(pair.pair_id, 0) == max_per_pair
    )
    return ContrastValidation(
        coordinates=tuple(accepted),
        dropped=tuple(dropped),
        protocol_error=None,
        leakage_count=leakage_count,
        eligible_challengers=eligible,
    )


def build_icv_witness_packet(
    coordinates: tuple[ContrastCoordinate, ...],
    *,
    seed: int,
    sample_id: str,
    panel_index: int,
) -> IcvWitnessPacket:
    rng = random.Random(_stable_hash(seed, sample_id, f"witness:{panel_index}"))
    ordered = list(coordinates)
    rng.shuffle(ordered)
    rendered: list[dict[str, str]] = []
    public_to_internal: dict[str, str] = {}
    public_left_to_candidate: dict[str, str] = {}
    public_right_to_candidate: dict[str, str] = {}
    for index, coordinate in enumerate(ordered):
        public_id = f"X{index}"
        swap = bool(rng.getrandbits(1))
        left_text = coordinate.right_text if swap else coordinate.left_text
        right_text = coordinate.left_text if swap else coordinate.right_text
        left_key = coordinate.right_candidate_key if swap else coordinate.left_candidate_key
        right_key = coordinate.left_candidate_key if swap else coordinate.right_candidate_key
        rendered.append(
            {
                "contrast_id": public_id,
                "statement_left": left_text,
                "statement_right": right_text,
            }
        )
        public_to_internal[public_id] = coordinate.contrast_id
        public_left_to_candidate[public_id] = left_key
        public_right_to_candidate[public_id] = right_key
    return IcvWitnessPacket(
        panel_index=panel_index,
        contrasts=tuple(rendered),
        public_to_internal=public_to_internal,
        public_left_to_candidate=public_left_to_candidate,
        public_right_to_candidate=public_right_to_candidate,
    )


def parse_icv_witness(
    payload: dict[str, Any] | None,
    *,
    packet: IcvWitnessPacket,
) -> IcvWitnessParseResult:
    expected = len(packet.public_to_internal)
    if not isinstance(payload, dict) or set(payload) != {"answers"} or not isinstance(payload.get("answers"), list):
        return IcvWitnessParseResult(False, {}, expected, 0, 0, ())
    by_public: dict[str, list[str]] = {}
    erased: list[dict[str, str]] = []
    for index, raw in enumerate(payload["answers"]):
        if not isinstance(raw, dict) or set(raw) != {"contrast_id", "verdict"}:
            erased.append({"row": str(index), "reason": "invalid_witness_row_schema"})
            continue
        public_id = str(raw.get("contrast_id") or "")
        verdict = str(raw.get("verdict") or "")
        if public_id not in packet.public_to_internal:
            erased.append({"row": str(index), "reason": "unknown_contrast_id"})
            continue
        by_public.setdefault(public_id, []).append(verdict)
    observations: dict[str, str] = {}
    valid_count = 0
    decisive_count = 0
    valid_verdicts = {"LEFT_ONLY", "RIGHT_ONLY", "BOTH", "NEITHER"}
    for public_id, internal_id in packet.public_to_internal.items():
        values = by_public.get(public_id, [])
        if len(values) != 1 or values[0] not in valid_verdicts:
            observations[internal_id] = "ERASURE"
            erased.append({"row": public_id, "reason": "missing_duplicate_or_invalid_verdict"})
            continue
        valid_count += 1
        verdict = values[0]
        if verdict in {"BOTH", "NEITHER"}:
            observations[internal_id] = "ERASURE"
            continue
        decisive_count += 1
        observations[internal_id] = (
            packet.public_left_to_candidate[public_id]
            if verdict == "LEFT_ONLY"
            else packet.public_right_to_candidate[public_id]
        )
    return IcvWitnessParseResult(
        True,
        observations,
        expected,
        valid_count,
        decisive_count,
        tuple(erased),
    )


def decode_icv(
    stage: StageDecision,
    coordinates: tuple[ContrastCoordinate, ...],
    panels: tuple[IcvWitnessParseResult, ...],
) -> DecodeDecision:
    """Apply the frozen 2-of-3, two-panel, unique-challenger rule."""

    if len(panels) != 2 or any(not panel.top_level_valid for panel in panels):
        return DecodeDecision(stage.anchor_answer, stage.anchor_key, False, "witness_panel_failure", (), ())
    by_challenger: dict[str, list[ContrastCoordinate]] = {}
    for coordinate in coordinates:
        by_challenger.setdefault(coordinate.challenger_key, []).append(coordinate)
    diagnostics: list[dict[str, Any]] = []
    passing: list[str] = []
    for challenger, items in sorted(by_challenger.items()):
        if len(items) != 3:
            continue
        panel_passes: list[bool] = []
        for panel_index, panel in enumerate(panels, start=1):
            values = [panel.observations.get(item.contrast_id, "ERASURE") for item in items]
            n_c = sum(value == challenger for value in values)
            n_a = sum(value == stage.anchor_key for value in values)
            erasures = len(values) - n_c - n_a
            passed = n_c >= 2 and n_c > n_a
            panel_passes.append(passed)
            diagnostics.append(
                {
                    "challenger_key": challenger,
                    "panel_index": panel_index,
                    "challenger_support": n_c,
                    "anchor_support": n_a,
                    "erasures": erasures,
                    "passed": passed,
                }
            )
        if all(panel_passes):
            passing.append(challenger)
    if len(passing) != 1:
        resolver = "multiple_challengers_passed" if len(passing) > 1 else "no_challenger_double_passed"
        return DecodeDecision(
            stage.anchor_answer,
            stage.anchor_key,
            False,
            resolver,
            tuple(passing),
            tuple(diagnostics),
        )
    winner = next(candidate for candidate in stage.candidates if candidate.key == passing[0])
    return DecodeDecision(
        winner.answer,
        winner.key,
        True,
        "unique_challenger_double_passed",
        tuple(passing),
        tuple(diagnostics),
    )


def evidence_unit_to_dict(unit: EvidenceUnit) -> dict[str, Any]:
    return asdict(unit)


def coordinate_to_dict(coordinate: ContrastCoordinate) -> dict[str, Any]:
    return asdict(coordinate)


def _resolve_group(
    raw_ids: Any,
    *,
    side: str,
    candidate_key: str,
    unit_maps: dict[str, dict[str, EvidenceUnit]],
) -> tuple[str | None, tuple[EvidenceUnit, ...] | None]:
    if not isinstance(raw_ids, list) or not 1 <= len(raw_ids) <= 3:
        return "evidence_group_size_invalid", None
    units: list[EvidenceUnit] = []
    for raw_id in raw_ids:
        match = re.fullmatch(rf"{side}:(E[0-9]+)", str(raw_id or ""))
        if match is None:
            return "evidence_id_side_or_format_invalid", None
        unit = unit_maps.get(candidate_key, {}).get(match.group(1))
        if unit is None:
            return "unknown_evidence_id", None
        if not unit.eligible:
            return f"ineligible_or_leaking_evidence:{unit.ineligible_reason}", None
        units.append(unit)
    indices = [int(unit.unit_id[1:]) for unit in units]
    if indices != list(range(indices[0], indices[0] + len(indices))):
        return "evidence_group_not_contiguous", None
    if len({unit.unit_id for unit in units}) != len(units):
        return "duplicate_evidence_id", None
    text = _group_text(tuple(units))
    if not 16 <= len(text) <= 512 or re.search(r"[\w]", text, flags=re.UNICODE) is None:
        return "evidence_group_text_contract_failed", None
    return None, tuple(units)


def _group_text(units: tuple[EvidenceUnit, ...]) -> str:
    return " ".join(unit.text for unit in units).strip()


def _sentence_spans(source: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for match in re.finditer(r"(?:[.!?。！？]+(?=\s|$)|\n+)", source):
        end = match.end()
        if source[start:end].strip():
            spans.append((start, end))
        start = end
    if source[start:].strip():
        spans.append((start, len(source)))
    return spans


def _split_long_spans(source: str, spans: list[tuple[int, int]], *, maximum: int) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    for start, end in spans:
        cursor = start
        while end - cursor > maximum:
            window = source[cursor : cursor + maximum + 1]
            breaks = [match.end() for match in re.finditer(r"[;,:，；：]\s*|\s+", window)]
            cut = max((value for value in breaks if value >= maximum // 2), default=maximum)
            output.append((cursor, cursor + cut))
            cursor += cut
        if source[cursor:end].strip():
            output.append((cursor, end))
    return output


def _merge_short_spans(
    source: str,
    spans: list[tuple[int, int]],
    *,
    minimum: int,
    maximum: int,
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    index = 0
    while index < len(spans):
        start, end = spans[index]
        if len(source[start:end].strip()) < minimum and index + 1 < len(spans):
            next_end = spans[index + 1][1]
            if next_end - start <= maximum:
                end = next_end
                index += 1
        if len(source[start:end].strip()) < minimum and output and end - output[-1][0] <= maximum:
            previous_start, _ = output.pop()
            start = previous_start
        output.append((start, end))
        index += 1
    return output


def _evidence_ineligible_reason(sample: DatasetSample, *, text: str, answer: str) -> str | None:
    normalized = _normalize(text)
    lowered = normalized.casefold()
    patterns = {
        "structured_marker": r"(?i)FINAL_ANSWER|</?think>",
        "candidate_identifier": r"(?i)\b(?:candidate|hypothesis)\s*[A-Z0-9]+\b",
        "option_identifier": r"(?i)\b(?:option|choice)\s*[A-Z]\b|\([A-Z]\)",
        "explicit_conclusion": (
            r"(?i)\b(?:final\s+answer|correct\s+(?:answer|option|choice)|"
            r"(?:therefore|thus|hence|so),?\s+(?:the\s+)?(?:answer|option|choice))\b|"
            r"(?:最终答案|正确(?:答案|选项)|因此[^。！？\n]{0,24}(?:答案|选项))"
        ),
    }
    for reason, pattern in patterns.items():
        if re.search(pattern, normalized):
            return reason
    answer_texts = {_normalize(answer).casefold()}
    options = sample.metadata.get("options")
    if isinstance(options, list):
        for option in options:
            if isinstance(option, dict) and str(option.get("label") or "").upper() == str(answer).strip("() .").upper():
                answer_texts.add(_normalize(option.get("text")).casefold())
    for answer_text in answer_texts:
        if lowered == answer_text or (
            len(answer_text) >= 4
            and re.search(r"(?i)\b(?:therefore|thus|hence|so)\b", normalized)
            and answer_text in lowered
        ):
            return "candidate_answer_leakage"
    return None


def _normalize(value: Any) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n").replace("\r", "\n").strip()


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _stable_hash(seed: int, sample_id: str, role: str) -> int:
    return int(_sha256(f"{seed}\0{sample_id}\0{role}")[:16], 16)


def _sentence_spans_v3(source: str) -> list[tuple[int, int]]:
    """Unicode-safe sentence boundaries without source-encoding literals."""

    spans: list[tuple[int, int]] = []
    start = 0
    pattern = r"(?:[.!?\u3002\uff01\uff1f]+(?=\s|$)|\n+)"
    for match in re.finditer(pattern, source):
        end = match.end()
        if source[start:end].strip():
            spans.append((start, end))
        start = end
    if source[start:].strip():
        spans.append((start, len(source)))
    return spans


def _split_long_spans_v3(
    source: str,
    spans: list[tuple[int, int]],
    *,
    maximum: int,
) -> list[tuple[int, int]]:
    output: list[tuple[int, int]] = []
    break_pattern = r"[;,:\uff0c\uff1b\uff1a]\s*|\s+"
    for start, end in spans:
        cursor = start
        while end - cursor > maximum:
            window = source[cursor : cursor + maximum + 1]
            breaks = [match.end() for match in re.finditer(break_pattern, window)]
            cut = max((value for value in breaks if value >= maximum // 2), default=maximum)
            output.append((cursor, cursor + cut))
            cursor += cut
        if source[cursor:end].strip():
            output.append((cursor, end))
    return output
