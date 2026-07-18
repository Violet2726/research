"""CATCH 的纯确定性候选编码、主动选题与解码层。"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class CandidateClass:
    key: str
    answer: str
    support_count: int
    representative_reasoning: str
    representative_trace_sha256: str


@dataclass(frozen=True)
class StageDecision:
    anchor_key: str
    anchor_answer: str
    candidates: tuple[CandidateClass, ...]
    vote_counts: dict[str, int]
    valid_count: int

    @property
    def triggered(self) -> bool:
        return len(self.candidates) > 1


@dataclass(frozen=True)
class TestOutcome:
    outcome_id: str
    text: str


@dataclass(frozen=True)
class Commitment:
    outcome_id: str
    trace_start: int
    trace_end: int
    evidence: str
    evidence_sha256: str


@dataclass(frozen=True)
class DiagnosticTest:
    test_id: str
    question: str
    outcomes: tuple[TestOutcome, ...]
    commitments: dict[str, Commitment | None]
    target_pairs: tuple[tuple[str, str], ...] = ()


@dataclass(frozen=True)
class TestBankValidation:
    tests: tuple[DiagnosticTest, ...]
    dropped: tuple[dict[str, str], ...]
    protocol_error: str | None
    evidence_quote_count: int = 0
    aligned_evidence_quote_count: int = 0
    leakage_count: int = 0


@dataclass(frozen=True)
class SelectionResult:
    tests: tuple[DiagnosticTest, ...]
    pair_distances: dict[str, int]
    objective: tuple[int, int, int, float]
    tie_break_sha256: str


@dataclass(frozen=True)
class WitnessPacket:
    panel_index: int
    tests: tuple[dict[str, Any], ...]
    public_test_to_internal: dict[str, str]
    public_outcome_to_internal: dict[str, dict[str, str]]


@dataclass(frozen=True)
class WitnessParseResult:
    """A recoverable witness vector plus coordinate-level erasure diagnostics."""

    vector: dict[str, str] | None
    top_level_valid: bool
    expected_coordinate_count: int
    valid_coordinate_count: int
    erased_rows: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class DecodeDecision:
    answer: str
    answer_key: str
    override_accepted: bool
    resolver: str
    passing_challengers: tuple[str, ...]
    panel_diagnostics: tuple[dict[str, Any], ...]


def build_stage_decision(rows: list[dict[str, Any]], *, seed: int, sample_id: str) -> StageDecision:
    """Cluster only valid canonical answers and select stable representative traces."""

    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        key = str(row.get("answer_class_key") or "").strip()
        answer = str(row.get("normalized_answer") or row.get("prediction") or "").strip()
        if key and answer:
            grouped.setdefault(key, []).append(row)
    if not grouped:
        return StageDecision("", "", (), {}, 0)
    ordered_keys = sorted(
        grouped,
        key=lambda key: (-len(grouped[key]), _stable_hash(seed, sample_id, f"anchor:{key}"), key),
    )
    candidates: list[CandidateClass] = []
    for key in ordered_keys:
        representative = min(
            grouped[key],
            key=lambda row: _sha256(f"{_reasoning(row)}\0{_answer(row)}"),
        )
        reasoning = _reasoning(representative)
        candidates.append(
            CandidateClass(
                key=key,
                answer=_answer(representative),
                support_count=len(grouped[key]),
                representative_reasoning=reasoning,
                representative_trace_sha256=_sha256(reasoning),
            )
        )
    return StageDecision(
        anchor_key=ordered_keys[0],
        anchor_answer=candidates[0].answer,
        candidates=tuple(candidates),
        vote_counts={key: len(grouped[key]) for key in ordered_keys},
        valid_count=sum(len(items) for items in grouped.values()),
    )


def build_hypothesis_labels(stage: StageDecision, *, seed: int, sample_id: str) -> dict[str, str]:
    """Anonymize classes without preserving plurality order."""

    keys = [candidate.key for candidate in stage.candidates]
    random.Random(_stable_hash(seed, sample_id, "hypotheses")).shuffle(keys)
    return {f"H{index}": key for index, key in enumerate(keys)}


def validate_test_bank(
    payload: dict[str, Any] | None,
    *,
    stage: StageDecision,
    hypothesis_to_key: dict[str, str],
    max_tests: int = 6,
) -> TestBankValidation:
    """Validate finite tests, trace-backed commitments, and leakage constraints."""

    if not isinstance(payload, dict) or not isinstance(payload.get("tests"), list):
        return TestBankValidation((), (), "tests_must_be_a_list")
    raw_tests = payload["tests"]
    if len(raw_tests) > max_tests:
        return TestBankValidation((), (), "too_many_tests")
    candidate_by_key = {candidate.key: candidate for candidate in stage.candidates}
    accepted: list[DiagnosticTest] = []
    dropped: list[dict[str, str]] = []
    seen_ids: set[str] = set()
    seen_questions: set[str] = set()
    evidence_quote_count = 0
    aligned_evidence_quote_count = 0
    leakage_count = 0
    for index, raw_test in enumerate(raw_tests):
        test_id = str(raw_test.get("test_id") or "") if isinstance(raw_test, dict) else f"index:{index}"
        attempted, aligned = _evidence_quote_alignment_counts(
            raw_test,
            candidate_by_key=candidate_by_key,
            hypothesis_to_key=hypothesis_to_key,
        )
        evidence_quote_count += attempted
        aligned_evidence_quote_count += aligned
        try:
            test = _validate_one_test(
                raw_test,
                candidate_by_key=candidate_by_key,
                hypothesis_to_key=hypothesis_to_key,
            )
            normalized_question = _normalize_text(test.question)
            if test.test_id in seen_ids:
                raise ValueError("duplicate_test_id")
            if normalized_question in seen_questions:
                raise ValueError("duplicate_question")
            seen_ids.add(test.test_id)
            seen_questions.add(normalized_question)
            accepted.append(test)
        except (TypeError, ValueError) as exc:
            reason = str(exc)
            dropped.append({"test_id": test_id, "reason": reason})
            if reason in {
                "answer_or_candidate_leakage",
                "candidate_id_in_outcome",
                "final_answer_evidence_forbidden",
            }:
                leakage_count += 1
    return TestBankValidation(
        tuple(accepted),
        tuple(dropped),
        None,
        evidence_quote_count,
        aligned_evidence_quote_count,
        leakage_count,
    )


def _validate_one_test(
    raw_test: Any,
    *,
    candidate_by_key: dict[str, CandidateClass],
    hypothesis_to_key: dict[str, str],
) -> DiagnosticTest:
    if not isinstance(raw_test, dict):
        raise TypeError("test_not_object")
    test_id = str(raw_test.get("test_id") or "").strip()
    if re.fullmatch(r"T[0-9]+", test_id) is None:
        raise ValueError("invalid_test_id")
    question = str(raw_test.get("question") or "").strip()
    if not question or len(question) > 1_000 or not any(character.isalnum() for character in question):
        raise ValueError("invalid_question")
    lowered = question.casefold()
    forbidden = ("final answer", "which option", "options:", "candidate h", "hypothesis h")
    if any(marker in lowered for marker in forbidden) or re.search(r"\([A-Z]\)", question):
        raise ValueError("answer_or_candidate_leakage")

    raw_outcomes = raw_test.get("outcomes")
    if not isinstance(raw_outcomes, list) or not 2 <= len(raw_outcomes) <= 4:
        raise ValueError("outcome_count_out_of_range")
    outcomes: list[TestOutcome] = []
    outcome_ids: set[str] = set()
    outcome_texts: set[str] = set()
    for raw_outcome in raw_outcomes:
        if not isinstance(raw_outcome, dict):
            raise ValueError("outcome_not_object")
        outcome_id = str(raw_outcome.get("id") or "").strip()
        text = str(raw_outcome.get("text") or "").strip()
        if re.fullmatch(r"O[0-9]+", outcome_id) is None or not text or len(text) > 300:
            raise ValueError("invalid_outcome")
        normalized = _normalize_text(text)
        if outcome_id in outcome_ids or normalized in outcome_texts:
            raise ValueError("duplicate_outcome")
        if re.search(r"\bH[0-9]+\b", text):
            raise ValueError("candidate_id_in_outcome")
        outcome_ids.add(outcome_id)
        outcome_texts.add(normalized)
        outcomes.append(TestOutcome(outcome_id, text))

    raw_commitments = raw_test.get("commitments")
    if not isinstance(raw_commitments, dict) or set(raw_commitments) != set(hypothesis_to_key):
        raise ValueError("commitment_hypotheses_mismatch")
    commitments: dict[str, Commitment | None] = {}
    for hypothesis, candidate_key in hypothesis_to_key.items():
        raw_commitment = raw_commitments[hypothesis]
        if raw_commitment is None:
            commitments[candidate_key] = None
            continue
        if not isinstance(raw_commitment, dict):
            raise ValueError("commitment_not_object_or_null")
        outcome_id = str(raw_commitment.get("outcome_id") or "").strip()
        if outcome_id not in outcome_ids:
            raise ValueError("unknown_commitment_outcome")
        evidence_quote = str(raw_commitment.get("evidence_quote") or "")
        if not evidence_quote.strip() or not any(character.isalnum() for character in evidence_quote):
            raise ValueError("missing_evidence_quote")
        if "final_answer" in _normalize_text(evidence_quote):
            raise ValueError("final_answer_evidence_forbidden")
        reasoning = candidate_by_key[candidate_key].representative_reasoning
        matches = _find_normalized_quote_spans(reasoning, evidence_quote)
        if not matches:
            raise ValueError("evidence_quote_not_found")
        if len(matches) != 1:
            raise ValueError("evidence_quote_ambiguous")
        start, end = matches[0]
        evidence = reasoning[start:end]
        if not evidence.strip() or not any(character.isalnum() for character in evidence):
            raise ValueError("empty_trace_evidence")
        commitments[candidate_key] = Commitment(
            outcome_id=outcome_id,
            trace_start=start,
            trace_end=end,
            evidence=evidence,
            evidence_sha256=_sha256(_normalize_match(evidence)),
        )
    non_null = [item for item in commitments.values() if item is not None]
    if len(non_null) < 2 or len({item.outcome_id for item in non_null}) < 2:
        raise ValueError("test_does_not_discriminate")
    target_pairs = _differing_candidate_pairs(commitments)
    if not target_pairs:
        raise ValueError("test_does_not_discriminate")
    return DiagnosticTest(test_id, question, tuple(outcomes), commitments, target_pairs)


def effective_pair_coordinates(
    tests: Iterable[DiagnosticTest],
    left_key: str,
    right_key: str,
    *,
    available_test_ids: set[str] | None = None,
) -> tuple[DiagnosticTest, ...]:
    """Return a maximum stable set of differing, non-overlapping coordinates."""

    differing = []
    for test in tests:
        if available_test_ids is not None and test.test_id not in available_test_ids:
            continue
        left = test.commitments.get(left_key)
        right = test.commitments.get(right_key)
        if left is not None and right is not None and left.outcome_id != right.outcome_id:
            differing.append(test)
    best: tuple[DiagnosticTest, ...] = ()
    best_hash = ""
    for size in range(1, len(differing) + 1):
        for subset in itertools.combinations(differing, size):
            if not _coordinates_are_independent(subset, (left_key, right_key)):
                continue
            digest = _sha256("\0".join(sorted(test.test_id for test in subset)))
            if len(subset) > len(best) or (len(subset) == len(best) and (not best_hash or digest < best_hash)):
                best = subset
                best_hash = digest
    return best


def select_tests(
    tests: Iterable[DiagnosticTest],
    *,
    stage: StageDecision,
    d_min: int,
    max_selected: int = 4,
) -> SelectionResult:
    candidates = tuple(tests)
    challengers = [candidate.key for candidate in stage.candidates if candidate.key != stage.anchor_key]
    if not candidates or not challengers:
        return SelectionResult((), {}, (0, 0, 0, 0.0), _sha256("empty"))
    best_subset: tuple[DiagnosticTest, ...] = ()
    best_distances: dict[str, int] = {}
    best_objective = (-1, -1, -1, -1.0)
    best_hash = ""
    for size in range(1, min(max_selected, len(candidates)) + 1):
        for subset in itertools.combinations(candidates, size):
            distances = {
                key: len(effective_pair_coordinates(subset, stage.anchor_key, key))
                for key in challengers
            }
            eligible = [distance for distance in distances.values() if distance >= d_min]
            objective = (
                len(eligible),
                min(eligible) if eligible else 0,
                sum(distances.values()),
                round(_partition_entropy(subset, stage), 12),
            )
            digest = _sha256("\0".join(sorted(test.test_id for test in subset)))
            if objective > best_objective or (objective == best_objective and (not best_hash or digest < best_hash)):
                best_subset = subset
                best_distances = distances
                best_objective = objective
                best_hash = digest
    return SelectionResult(best_subset, best_distances, best_objective, best_hash)


def select_first_tests(tests: Iterable[DiagnosticTest], *, limit: int = 4) -> tuple[DiagnosticTest, ...]:
    return tuple(sorted(tests, key=lambda item: (item.test_id, _sha256(item.question)))[:limit])


def shuffle_commitments(
    tests: Iterable[DiagnosticTest],
    *,
    stage: StageDecision,
    seed: int,
    sample_id: str,
) -> tuple[DiagnosticTest, ...]:
    """Negative control: permute complete candidate signatures without changing tests."""

    keys = [candidate.key for candidate in stage.candidates]
    shuffled = list(keys)
    random.Random(_stable_hash(seed, sample_id, "signature-shuffle")).shuffle(shuffled)
    if len(shuffled) > 1 and shuffled == keys:
        shuffled = shuffled[1:] + shuffled[:1]
    source_for_target = dict(zip(keys, shuffled, strict=True))
    return tuple(
        DiagnosticTest(
            test_id=test.test_id,
            question=test.question,
            outcomes=test.outcomes,
            commitments={target: test.commitments.get(source) for target, source in source_for_target.items()},
            target_pairs=_differing_candidate_pairs(
                {target: test.commitments.get(source) for target, source in source_for_target.items()}
            ),
        )
        for test in tests
    )


def build_witness_packet(
    tests: Iterable[DiagnosticTest],
    *,
    seed: int,
    sample_id: str,
    panel_index: int,
) -> WitnessPacket:
    """Independently permute test order and public outcome labels."""

    selected = list(tests)
    rng = random.Random(_stable_hash(seed, sample_id, f"witness:{panel_index}"))
    rng.shuffle(selected)
    public_test_to_internal: dict[str, str] = {}
    public_outcome_to_internal: dict[str, dict[str, str]] = {}
    rendered: list[dict[str, Any]] = []
    for index, test in enumerate(selected):
        public_test_id = f"Q{index}"
        public_test_to_internal[public_test_id] = test.test_id
        outcomes = list(test.outcomes)
        rng.shuffle(outcomes)
        public_mapping: dict[str, str] = {}
        rendered_outcomes = []
        for outcome_index, outcome in enumerate(outcomes):
            public_id = f"R{outcome_index}"
            public_mapping[public_id] = outcome.outcome_id
            rendered_outcomes.append({"id": public_id, "text": outcome.text})
        public_outcome_to_internal[public_test_id] = public_mapping
        rendered.append({"test_id": public_test_id, "question": test.question, "outcomes": rendered_outcomes})
    return WitnessPacket(
        panel_index=panel_index,
        tests=tuple(rendered),
        public_test_to_internal=public_test_to_internal,
        public_outcome_to_internal=public_outcome_to_internal,
    )


def parse_witness_answers(
    payload: dict[str, Any] | None,
    *,
    packet: WitnessPacket,
) -> dict[str, str] | None:
    """Map a witness response back to internal outcomes; invalid rows are erasures."""

    return parse_witness_answers_detailed(payload, packet=packet).vector


def parse_witness_answers_detailed(
    payload: dict[str, Any] | None,
    *,
    packet: WitnessPacket,
) -> WitnessParseResult:
    """Recover valid coordinates without promoting one bad row to a panel failure."""

    if not isinstance(payload, dict) or not isinstance(payload.get("answers"), list):
        return WitnessParseResult(None, False, len(packet.tests), 0, ())
    observed: dict[str, str] = {}
    seen: set[str] = set()
    erased: list[dict[str, str]] = []
    for index, row in enumerate(payload["answers"]):
        if not isinstance(row, dict):
            erased.append({"row": str(index), "reason": "answer_not_object"})
            continue
        public_test_id = str(row.get("test_id") or "")
        public_outcome_id = str(row.get("outcome_id") or "")
        if public_test_id in seen:
            erased.append({"row": str(index), "reason": "duplicate_test_id"})
            internal = packet.public_test_to_internal.get(public_test_id)
            if internal is not None:
                observed.pop(internal, None)
            continue
        if public_test_id not in packet.public_test_to_internal:
            erased.append({"row": str(index), "reason": "unknown_test_id"})
            continue
        if public_outcome_id not in packet.public_outcome_to_internal[public_test_id]:
            erased.append({"row": str(index), "reason": "unknown_outcome_id"})
            seen.add(public_test_id)
            continue
        seen.add(public_test_id)
        internal_test = packet.public_test_to_internal[public_test_id]
        observed[internal_test] = packet.public_outcome_to_internal[public_test_id][public_outcome_id]
    return WitnessParseResult(
        observed,
        True,
        len(packet.tests),
        len(observed),
        tuple(erased),
    )


def decode_witnesses(
    stage: StageDecision,
    tests: Iterable[DiagnosticTest],
    witness_vectors: list[dict[str, str] | None],
    *,
    d_min: int,
    margin: int,
    required_panels: int = 2,
) -> DecodeDecision:
    selected = tuple(tests)
    if not stage.anchor_key:
        return DecodeDecision("", "", False, "no_valid_stage_answer", (), ())
    if len(witness_vectors) < required_panels or any(vector is None for vector in witness_vectors[:required_panels]):
        return DecodeDecision(
            stage.anchor_answer,
            stage.anchor_key,
            False,
            "witness_protocol_failure",
            (),
            (),
        )
    diagnostics: list[dict[str, Any]] = []
    passes_by_panel: list[set[str]] = []
    for panel_index, vector in enumerate(witness_vectors[:required_panels], start=1):
        assert vector is not None
        panel_passes: set[str] = set()
        challenger_rows = []
        for candidate in stage.candidates:
            if candidate.key == stage.anchor_key:
                continue
            coordinates = effective_pair_coordinates(
                selected,
                stage.anchor_key,
                candidate.key,
                available_test_ids=set(vector),
            )
            anchor_distance = 0
            challenger_distance = 0
            for test in coordinates:
                observed = vector[test.test_id]
                anchor = test.commitments[stage.anchor_key]
                challenger = test.commitments[candidate.key]
                assert anchor is not None and challenger is not None
                anchor_distance += int(observed != anchor.outcome_id)
                challenger_distance += int(observed != challenger.outcome_id)
            advantage = anchor_distance - challenger_distance
            passed = len(coordinates) >= d_min and advantage >= margin
            if passed:
                panel_passes.add(candidate.key)
            challenger_rows.append(
                {
                    "challenger_key": candidate.key,
                    "effective_test_ids": [test.test_id for test in coordinates],
                    "effective_distance": len(coordinates),
                    "anchor_distance": anchor_distance,
                    "challenger_distance": challenger_distance,
                    "advantage": advantage,
                    "passed": passed,
                }
            )
        passes_by_panel.append(panel_passes)
        diagnostics.append({"panel_index": panel_index, "challengers": challenger_rows})
    passing = set.intersection(*passes_by_panel) if passes_by_panel else set()
    if len(passing) != 1:
        resolver = "no_unique_challenger" if not passing else "multiple_challengers"
        return DecodeDecision(
            stage.anchor_answer,
            stage.anchor_key,
            False,
            resolver,
            tuple(sorted(passing)),
            tuple(diagnostics),
        )
    winner_key = next(iter(passing))
    winner = next(candidate for candidate in stage.candidates if candidate.key == winner_key)
    return DecodeDecision(
        winner.answer,
        winner.key,
        True,
        "unique_double_witness_override" if required_panels == 2 else "unique_single_witness_override",
        (winner.key,),
        tuple(diagnostics),
    )


def decide_direct_judges(
    stage: StageDecision,
    selected_keys: Iterable[str | None],
) -> tuple[str, bool, str]:
    valid_keys = {candidate.key for candidate in stage.candidates}
    votes = Counter(key for key in selected_keys if key in valid_keys)
    if not votes:
        return stage.anchor_answer, False, "no_valid_judge_vote"
    ordered = votes.most_common()
    if len(ordered) > 1 and ordered[0][1] == ordered[1][1]:
        return stage.anchor_answer, False, "judge_tie"
    winner_key = ordered[0][0]
    if winner_key == stage.anchor_key:
        return stage.anchor_answer, False, "judge_selected_anchor"
    winner = next(candidate for candidate in stage.candidates if candidate.key == winner_key)
    return winner.answer, True, "direct_judge_majority_override"


def test_to_dict(test: DiagnosticTest) -> dict[str, Any]:
    return asdict(test)


def _evidence_quote_alignment_counts(
    raw_test: Any,
    *,
    candidate_by_key: dict[str, CandidateClass],
    hypothesis_to_key: dict[str, str],
) -> tuple[int, int]:
    if not isinstance(raw_test, dict) or not isinstance(raw_test.get("commitments"), dict):
        return 0, 0
    attempted = 0
    aligned = 0
    for hypothesis, raw_commitment in raw_test["commitments"].items():
        if raw_commitment is None or not isinstance(raw_commitment, dict):
            continue
        quote = str(raw_commitment.get("evidence_quote") or "")
        if not quote.strip():
            continue
        attempted += 1
        candidate_key = hypothesis_to_key.get(str(hypothesis))
        candidate = candidate_by_key.get(str(candidate_key))
        if candidate is not None and len(_find_normalized_quote_spans(candidate.representative_reasoning, quote)) == 1:
            aligned += 1
    return attempted, aligned


def _find_normalized_quote_spans(text: str, quote: str) -> list[tuple[int, int]]:
    normalized_text, starts, ends = _normalized_text_with_offsets(text)
    normalized_quote = _normalize_match(quote)
    if not normalized_quote:
        return []
    matches: list[tuple[int, int]] = []
    offset = 0
    while True:
        index = normalized_text.find(normalized_quote, offset)
        if index < 0:
            break
        final_index = index + len(normalized_quote) - 1
        if index < len(starts) and final_index < len(ends):
            raw_span = (starts[index], ends[final_index])
            if _normalize_match(text[raw_span[0] : raw_span[1]]) == normalized_quote:
                matches.append(raw_span)
        offset = index + 1
    return sorted(set(matches))


def _normalized_text_with_offsets(value: str) -> tuple[str, list[int], list[int]]:
    rendered: list[str] = []
    starts: list[int] = []
    ends: list[int] = []
    index = 0
    while index < len(value):
        if value[index : index + 2] == "\r\n":
            piece = "\n"
            raw_end = index + 2
        else:
            piece = unicodedata.normalize("NFKC", value[index])
            raw_end = index + 1
        for character in piece:
            rendered.append(character)
            starts.append(index)
            ends.append(raw_end)
        index = raw_end
    return "".join(rendered), starts, ends


def _differing_candidate_pairs(
    commitments: dict[str, Commitment | None],
) -> tuple[tuple[str, str], ...]:
    pairs = []
    for left, right in itertools.combinations(sorted(commitments), 2):
        left_commitment = commitments[left]
        right_commitment = commitments[right]
        if (
            left_commitment is not None
            and right_commitment is not None
            and left_commitment.outcome_id != right_commitment.outcome_id
        ):
            pairs.append((left, right))
    return tuple(pairs)


def _coordinates_are_independent(tests: Iterable[DiagnosticTest], keys: tuple[str, ...]) -> bool:
    seen: dict[str, list[tuple[int, int]]] = {key: [] for key in keys}
    for test in tests:
        for key in keys:
            commitment = test.commitments.get(key)
            if commitment is None:
                continue
            span = (commitment.trace_start, commitment.trace_end)
            if any(_overlap(span, prior) for prior in seen[key]):
                return False
            seen[key].append(span)
    return True


def _overlap(left: tuple[int, int], right: tuple[int, int]) -> bool:
    return max(left[0], right[0]) < min(left[1], right[1])


def _partition_entropy(tests: Iterable[DiagnosticTest], stage: StageDecision) -> float:
    total = len(stage.candidates)
    if total == 0:
        return 0.0
    entropy = 0.0
    for test in tests:
        groups = Counter(
            test.commitments[candidate.key].outcome_id
            if test.commitments.get(candidate.key) is not None
            else "__ERASURE__"
            for candidate in stage.candidates
        )
        entropy += -sum((count / total) * math.log2(count / total) for count in groups.values())
    return entropy


def _answer(row: dict[str, Any]) -> str:
    return str(row.get("normalized_answer") or row.get("prediction") or "").strip()


def _reasoning(row: dict[str, Any]) -> str:
    validated = row.get("validated_output")
    if isinstance(validated, dict) and str(validated.get("reasoning") or "").strip():
        return str(validated["reasoning"])
    return str(row.get("reasoning") or "").strip()


def _normalize_text(value: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split()).casefold()


def _normalize_match(value: str) -> str:
    return unicodedata.normalize("NFKC", str(value or "")).replace("\r\n", "\n")


def _stable_hash(seed: int, sample_id: str, purpose: str) -> str:
    return _sha256(f"catch-v1:{seed}:{sample_id}:{purpose}")


def _sha256(value: str) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()
