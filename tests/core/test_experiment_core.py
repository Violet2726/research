"""覆盖 `experiment_core` 共享基础能力的测试。"""

from __future__ import annotations

from pathlib import Path
import json
import time

import httpx
import pytest

from experiment_core.foundation.artifacts import BufferedJsonlWriter
from experiment_core.foundation.cache import (
    CachedResponse,
    RequestCache,
    RequestCacheRouter,
    build_request_cache_key,
    cache_successful_response,
    inspect_cache_shard,
    json_dump,
    resolve_cache_shard_path,
    summarize_cache_root,
)
from experiment_core.foundation.config import load_benchmark_config, load_model_catalog, parse_model_ref, resolve_model_ref
from experiment_core.foundation.datasets import generate_split_manifests, load_split_ids, select_samples
from experiment_core.foundation.providers import (
    OpenAICompatibleProvider,
    ProviderResponse,
    _extract_message_channels,
    build_payload,
    execute_completion_request,
)
from experiment_core.foundation.rate_limits import SlidingWindowRateLimiter
from experiment_core.foundation.runtime import finalize_run_outputs
from experiment_core.controls.selective_signals import decide_trigger, summarize_confidence_rows, summarize_divergence_rows
from experiment_core.foundation.structured_output import (
    OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
    OUTPUT_MODE_BUDGET_SOLVER,
    OUTPUT_MODE_COMM_NECESSARY_BELIEF,
    OUTPUT_MODE_COMM_NECESSARY_SOLVER,
    OUTPUT_MODE_CUE_SOLVER,
    OUTPUT_MODE_CORE,
    OUTPUT_MODE_SELECTIVE_COMM,
    OUTPUT_MODE_SPARC_AUDIT,
    OUTPUT_MODE_SPARC_BELIEF_UPDATE,
    OUTPUT_MODE_SPARC_MESSAGE,
    OUTPUT_MODE_SPARC_SOLVER,
    parse_selective_output,
    validate_or_recover_structured_output,
    validate_structured_output,
)
from experiment_core.foundation.workspace import (
    auto_publish_runs_enabled,
    auto_push_cache_snapshot_enabled,
    default_cache_root,
    default_cache_hf_repo,
    default_datasets_root,
    default_files_root,
    default_reports_root,
    default_runs_root,
    default_runs_hf_repo,
    workspace_defaults,
)


def test_parse_model_ref() -> None:
    assert parse_model_ref("deepseek/deepseek-v4-flash") == ("deepseek", "deepseek-v4-flash")


def test_load_model_catalog() -> None:
    catalog = load_model_catalog()
    assert catalog
    assert all("/" in key for key in catalog)


def test_resolve_local_ollama_model_ref() -> None:
    resolved = resolve_model_ref("local_ollama/qwen3:4b")
    assert resolved.provider == "local_ollama"
    assert resolved.model_id == "qwen3:4b"
    assert resolved.base_url == "http://127.0.0.1:11434/v1"
    assert resolved.api_key_env == "OLLAMA_API_KEY"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True


def test_resolve_deepseek_model_ref() -> None:
    resolved = resolve_model_ref("deepseek/deepseek-v4-flash")
    assert resolved.provider == "deepseek"
    assert resolved.model_id == "deepseek-v4-flash"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True


def test_resolve_xiaomimimo_model_ref() -> None:
    resolved = resolve_model_ref("xiaomimimo/mimo-v2.5")
    assert resolved.provider == "xiaomimimo"
    assert resolved.model_id == "mimo-v2.5"
    assert resolved.reasoning_effort == "none"
    assert resolved.supports_response_format is True


def test_workspace_defaults_follow_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_ROOT", "experiment-runs")
    monkeypatch.setenv("RESEARCH_REPORTS_ROOT", "published-reports")
    monkeypatch.setenv("RESEARCH_CACHE_ROOT", "tmp/cache")
    monkeypatch.setenv("RESEARCH_DATASETS_ROOT", "tmp/datasets")
    monkeypatch.setenv("RESEARCH_FILES_ROOT", "notes")
    monkeypatch.setenv("RESEARCH_RUNS_HF_REPO", "owner/research-runs")
    monkeypatch.setenv("RESEARCH_CACHE_HF_REPO", "owner/research-cache")
    monkeypatch.setenv("RESEARCH_AUTO_PUBLISH_RUNS", "1")
    monkeypatch.setenv("RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT", "true")

    assert default_runs_root("selective_comm") == "experiment-runs/selective_comm"
    assert default_reports_root("selective_comm") == "published-reports/selective_comm"
    assert default_cache_root() == "tmp/cache"
    assert default_datasets_root() == "tmp/datasets"
    assert default_files_root() == "notes"
    assert default_runs_hf_repo() == "owner/research-runs"
    assert default_cache_hf_repo() == "owner/research-cache"
    assert auto_publish_runs_enabled() is True
    assert auto_push_cache_snapshot_enabled() is True

    payload = workspace_defaults("selective_comm")
    assert payload["runs_root"] == "experiment-runs"
    assert payload["reports_root"] == "published-reports"
    assert payload["experiment_runs_root"] == "experiment-runs/selective_comm"
    assert payload["experiment_reports_root"] == "published-reports/selective_comm"
    assert payload["experiment_cache_root"] == "tmp/cache"
    assert payload["datasets_root"] == "tmp/datasets"
    assert payload["runs_hf_repo"] == "owner/research-runs"
    assert payload["cache_hf_repo"] == "owner/research-cache"
    assert payload["auto_publish_runs"] is True
    assert payload["auto_push_cache_snapshot"] is True


def test_workspace_defaults_use_local_roots_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in [
        "RESEARCH_RUNS_ROOT",
        "RESEARCH_REPORTS_ROOT",
        "RESEARCH_CACHE_ROOT",
        "RESEARCH_DATASETS_ROOT",
        "RESEARCH_FILES_ROOT",
        "RESEARCH_RUNS_HF_REPO",
        "RESEARCH_CACHE_HF_REPO",
        "RESEARCH_AUTO_PUBLISH_RUNS",
        "RESEARCH_AUTO_PUSH_CACHE_SNAPSHOT",
    ]:
        monkeypatch.delenv(name, raising=False)

    assert default_runs_root("single_agent") == "local/runs/single_agent"
    assert default_reports_root("single_agent") == "local/reports/single_agent"
    assert default_cache_root() == "local/cache"
    assert default_datasets_root() == "local/datasets"
    assert default_files_root() == "files"
    assert default_runs_hf_repo() is None
    assert default_cache_hf_repo() is None
    assert auto_publish_runs_enabled() is False
    assert auto_push_cache_snapshot_enabled() is False


def test_build_payload_maps_thinking_control_by_provider() -> None:
    local_model = resolve_model_ref("local_ollama/qwen3:4b")
    local_payload = build_payload(
        local_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert local_payload["reasoning_effort"] == "none"
    assert "enable_thinking" not in local_payload

    deepseek_model = resolve_model_ref("deepseek/deepseek-v4-flash")
    deepseek_payload = build_payload(
        deepseek_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert deepseek_payload["thinking"] == {"type": "disabled"}
    assert "enable_thinking" not in deepseek_payload

    xiaomimimo_model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    xiaomimimo_payload = build_payload(
        xiaomimimo_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
    )
    assert xiaomimimo_payload["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in xiaomimimo_payload
    assert "enable_thinking" not in xiaomimimo_payload
    assert xiaomimimo_payload["response_format"] == {"type": "json_object"}

    xiaomimimo_payload_no_format = build_payload(
        xiaomimimo_model,
        [{"role": "user", "content": "hi"}],
        temperature=0.0,
        top_p=1.0,
        max_output_tokens=16,
        seed=42,
        use_response_format=False,
    )
    assert xiaomimimo_payload_no_format["thinking"] == {"type": "disabled"}
    assert "response_format" not in xiaomimimo_payload_no_format


def test_validate_core_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning": "short",
            }
        ),
        OUTPUT_MODE_CORE,
    )
    assert payload["final_answer"] == "yes"
    assert payload["reasoning"] == "short"


def test_validate_or_recover_core_output_from_truncated_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 42, "reasoning": "simple arithmetic"',
        OUTPUT_MODE_CORE,
    )
    assert payload["final_answer"] == "42"
    assert payload["reasoning"] == "simple arithmetic"


def test_validate_selective_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": 0.4,
                "reasoning": "short",
                "claim_span": "supporting span",
                "uncertainty_type": "evidence_selection",
                "key_evidence": "one clue",
                "uncertain_point": None,
            }
        ),
        OUTPUT_MODE_SELECTIVE_COMM,
    )
    assert payload["confidence_raw"] == 0.4
    assert payload["claim_span"] == "supporting span"
    assert payload["uncertainty_type"] == "evidence_selection"
    assert payload["key_evidence"] == "one clue"
    assert payload["uncertain_point"] is None


def test_validate_selective_structured_output_allows_missing_confidence() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "42",
                "claim_span": "5 times 8 plus 2",
                "uncertainty_type": "calculation",
            }
        ),
        OUTPUT_MODE_SELECTIVE_COMM,
    )
    assert payload["final_answer"] == "42"
    assert payload["confidence_raw"] is None


def test_validate_or_recover_selective_output_uses_reasoning_fallback() -> None:
    payload = validate_or_recover_structured_output(
        "[answer]",
        OUTPUT_MODE_SELECTIVE_COMM,
        dataset="gsm8k",
        provider_reasoning_text="Therefore, the final answer is 36.",
    )
    assert payload["final_answer"] == "36"
    assert payload["uncertainty_type"] == "calculation"


def test_validate_selective_structured_output_rejects_invalid_uncertainty_type() -> None:
    with pytest.raises(ValueError):
        validate_structured_output(
            json.dumps(
                {
                    "final_answer": "yes",
                    "confidence_raw": 0.4,
                    "claim_span": "supporting span",
                    "uncertainty_type": "bad_label",
                }
            ),
            OUTPUT_MODE_SELECTIVE_COMM,
        )


def test_parse_selective_output_accepts_tagged_lines() -> None:
    payload = parse_selective_output(
        "\n".join(
            [
                "FINAL_ANSWER: 36",
                "CLAIM_SPAN: 54 - 18 = 36",
                "UNCERTAINTY_TYPE: calculation",
                "CONFIDENCE: 0.82",
                "REASON: students take the remaining seats",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "36"
    assert payload["claim_span"] == "54 - 18 = 36"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] == "0.82"


def test_parse_selective_output_recovers_from_free_form_math_text() -> None:
    payload = parse_selective_output(
        "To solve it, total seats are 72, admins use 18, parents use 18, so students are 36. The final answer is 36.",
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "36"
    assert payload["claim_span"] == "36"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] is None


def test_parse_selective_output_prefers_last_real_label_after_thought_block() -> None:
    payload = parse_selective_output(
        "\n".join(
            [
                "{thought}",
                "The model is planning its answer.",
                "CONFIDENCE: <0-1 number or NA>",
                "{/thought}",
                "FINAL_ANSWER: 52",
                "CLAIM_SPAN: 47, 52, 57 -> average 52",
                "UNCERTAINTY_TYPE: none",
                "CONFIDENCE: 1",
                "REASON: Simple arithmetic.",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "52"
    assert payload["confidence_raw"] == "1"
    assert payload["claim_span"] == "47, 52, 57 -> average 52"


def test_parse_selective_output_recovers_from_truncated_thought_math() -> None:
    payload = parse_selective_output(
        "\n".join(
            [
                "{thought}",
                "Total ounces = 2 glasses * 8 ounces/glass = 16 ounces.",
                "Total calories = 16 ounces * 3 calories/ounce = 48 calories.",
                "The output must have five lines:",
                "FINAL_ANSWER: <answer>",
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "48"
    assert payload["confidence_raw"] is None


def test_parse_selective_output_accepts_json_with_tagged_keys() -> None:
    payload = parse_selective_output(
        json.dumps(
            {
                "FINAL_ANSWER": "5",
                "CLAIM_SPAN": "2 children * 5 changes/day / 2",
                "UNCERTAINTY_TYPE": "calculation",
                "CONFIDENCE": 1.0,
                "REASON": "Simple arithmetic with clear counts.",
            }
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "5"
    assert payload["claim_span"] == "2 children * 5 changes/day / 2"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] == 1.0
    assert payload["reasoning"] == "Simple arithmetic with clear counts."


def test_parse_selective_output_accepts_json_with_na_confidence() -> None:
    payload = parse_selective_output(
        json.dumps(
            {
                "FINAL_ANSWER": "14",
                "CLAIM_SPAN": "28 remaining lollipops, 2 per bag -> 14 bags",
                "UNCERTAINTY_TYPE": "calculation",
                "CONFIDENCE": "NA",
                "REASON": "Simple subtraction and division.",
            }
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "14"
    assert payload["confidence_raw"] is None


def test_parse_selective_output_recovers_from_truncated_json() -> None:
    payload = parse_selective_output(
        "\n".join(
            [
                "{",
                '  "FINAL_ANSWER": "14",',
                '  "CLAIM_SPAN": "28 remaining lollipops, 2 per bag -> 14 bags",',
                '  "UNCERTAINTY_TYPE": "calculation",',
                '  "CONFIDENCE": "NA",',
                '  "REASON": "Simple subtraction',
            ]
        ),
        dataset="gsm8k",
    )
    assert payload["final_answer"] == "14"
    assert payload["claim_span"] == "28 remaining lollipops, 2 per bag -> 14 bags"
    assert payload["uncertainty_type"] == "calculation"
    assert payload["confidence_raw"] is None


def test_validate_comm_necessary_solver_output_allows_empty_final_answer_as_abstention() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "",
                "reasoning_trace": "Visible evidence does not connect the entities.",
                "evidence_summary": "No grounded answer in this shard.",
                "supporting_facts": [],
                "confidence_raw": 0.0,
            }
        ),
        OUTPUT_MODE_COMM_NECESSARY_SOLVER,
    )
    assert payload["final_answer"] is None
    assert payload["evidence_summary"] == "No grounded answer in this shard."


def test_validate_comm_necessary_belief_output_normalizes_empty_answer_to_no_change() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": True,
                "final_answer": "",
                "reasoning_trace": "Peer packets still do not ground an answer.",
                "evidence_summary": "Insufficient cross-shard evidence.",
                "supporting_facts": [],
                "confidence_raw": 0.0,
            }
        ),
        OUTPUT_MODE_COMM_NECESSARY_BELIEF,
    )
    assert payload["changed_answer"] is False
    assert payload["final_answer"] is None


def test_validate_budget_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "reasoning_trace": "short trace",
                "claim_span": "one claim",
                "key_evidence": "one evidence",
                "keyword_clues": ["alpha", "beta"],
                "confidence_raw": 0.8,
                "uncertain_point": None,
            }
        ),
        OUTPUT_MODE_BUDGET_SOLVER,
    )
    assert payload["keyword_clues"] == ["alpha", "beta"]
    assert payload["confidence_raw"] == 0.8


def test_validate_or_recover_budget_solver_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 50, "confidence_raw": 0.8, "keyword_clues": ["50"], "reasoning_trace": "done"',
        OUTPUT_MODE_BUDGET_SOLVER,
    )
    assert payload["final_answer"] == "50"
    assert payload["confidence_raw"] == 0.8
    assert payload["keyword_clues"] == ["50"]


def test_validate_or_recover_budget_solver_defaults_missing_confidence() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer":"82","reasoning_trace":"done","claim_span":"capacity math","key_evidence":"5000-3755"}',
        OUTPUT_MODE_BUDGET_SOLVER,
    )
    assert payload["final_answer"] == "82"
    assert payload["confidence_raw"] == 0.5
    assert payload["keyword_clues"] == ["capacity math"]


def test_validate_budget_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": False,
                "new_answer": "no",
                "confidence_delta": 0.1,
                "reason_for_change": "peer confirms it",
                "remaining_disagreement": None,
            }
        ),
        OUTPUT_MODE_BUDGET_BELIEF_UPDATE,
    )
    assert payload["changed_answer"] is False
    assert payload["new_answer"] == "no"


def test_validate_comm_necessary_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "Scott Adkins",
                "reasoning_trace": "Actor match from visible evidence.",
                "evidence_summary": "Universal Soldier and Holby City overlap.",
                "supporting_facts": [{"title": "Scott Adkins", "sent_id": 0}],
                "confidence_raw": 0.6,
            }
        ),
        OUTPUT_MODE_COMM_NECESSARY_SOLVER,
    )
    assert payload["final_answer"] == "Scott Adkins"
    assert payload["supporting_facts"] == [{"title": "Scott Adkins", "sent_id": 0}]


def test_validate_or_recover_comm_necessary_belief_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"changed_answer": false, "final_answer": "Saoirse Ronan", "reasoning_trace": "peer confirms", '
        '"supporting_facts":[{"title":"Billy Howle","sent_id":2}], "confidence_raw": 0.7',
        OUTPUT_MODE_COMM_NECESSARY_BELIEF,
    )
    assert payload["changed_answer"] is False
    assert payload["final_answer"] == "Saoirse Ronan"
    assert payload["supporting_facts"] == [{"title": "Billy Howle", "sent_id": 2}]


def test_validate_or_recover_core_soft_rejection_fail_opens() -> None:
    payload = validate_or_recover_structured_output(
        "The request was rejected because it was considered high risk",
        OUTPUT_MODE_CORE,
    )
    assert payload["final_answer"] == "unknown"
    assert payload["reasoning"] == "provider_soft_rejection"


def test_validate_or_recover_comm_necessary_soft_rejection_fail_opens() -> None:
    payload = validate_or_recover_structured_output(
        "The request was rejected because it was considered high risk",
        OUTPUT_MODE_COMM_NECESSARY_SOLVER,
    )
    assert payload["final_answer"] == "unknown"
    assert payload["supporting_facts"] == []


def test_validate_sparc_solver_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "42",
                "reasoning_trace": "short trace",
                "claim_span": "2+40",
                "confidence_raw": 0.7,
                "uncertain_point": "unit conversion",
                "key_evidence": "5 groups of 8 and 2 extra",
            }
        ),
        OUTPUT_MODE_SPARC_SOLVER,
    )
    assert payload["final_answer"] == "42"
    assert payload["confidence_raw"] == 0.7


def test_validate_or_recover_cue_solver_from_partial_json() -> None:
    payload = validate_or_recover_structured_output(
        '{"final_answer": 50, "confidence": 0.6, "top_claims": ["50"], "evidence_items": ["calc"], "reasoning_sketch": "done"',
        OUTPUT_MODE_CUE_SOLVER,
    )
    assert payload["final_answer"] == "50"
    assert payload["confidence"] == 0.6
    assert payload["top_claims"] == ["50"]


def test_validate_sparc_message_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "final_answer": "yes",
                "confidence_raw": "0.8",
                "claim_span": "the key factual claim",
            }
        ),
        OUTPUT_MODE_SPARC_MESSAGE,
    )
    assert payload["claim_span"] == "the key factual claim"


def test_validate_sparc_belief_update_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "changed_answer": True,
                "new_answer": "no",
                "confidence_delta": -0.1,
                "reason_for_change": "peer evidence is stronger",
                "remaining_disagreement": None,
            }
        ),
        OUTPUT_MODE_SPARC_BELIEF_UPDATE,
    )
    assert payload["changed_answer"] is True
    assert payload["new_answer"] == "no"


def test_validate_sparc_audit_structured_output() -> None:
    payload = validate_structured_output(
        json.dumps(
            {
                "decision": "resolve_for_a",
                "verified_answer": "48",
                "rationale": "candidate A matches the evidence",
            }
        ),
        OUTPUT_MODE_SPARC_AUDIT,
    )
    assert payload["decision"] == "resolve_for_a"


def test_selective_signal_summary_and_decision() -> None:
    summary = summarize_confidence_rows(
        [
            {"agent_id": 1, "confidence_valid": True, "confidence_value": 0.9},
            {"agent_id": 2, "confidence_valid": True, "confidence_value": 0.4},
            {"agent_id": 3, "confidence_valid": False, "confidence_value": None},
        ]
    )
    assert summary.mean_confidence == 0.65
    assert summary.confidence_spread == 0.5
    assert summary.invalid_agent_ids == [3]
    decision = decide_trigger(
        trigger_type="hybrid_trigger",
        initial_disagreement=False,
        mean_confidence=summary.mean_confidence,
        confidence_spread=summary.confidence_spread,
        any_invalid_confidence=summary.any_invalid_confidence,
        fail_open_to_always=False,
    )
    assert decision.triggered is True


def test_missing_confidence_is_not_counted_as_invalid() -> None:
    summary = summarize_confidence_rows(
        [
            {"agent_id": 1, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
            {"agent_id": 2, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
            {"agent_id": 3, "confidence_valid": False, "confidence_value": None, "confidence_source": "missing"},
        ]
    )
    assert summary.invalid_agent_ids == []
    assert summary.any_invalid_confidence is False
    assert summary.mean_confidence is None


def test_selective_divergence_summary_supports_voc_signals() -> None:
    summary = summarize_divergence_rows(
        [
            {
                "normalized_answer": "A",
                "claim_span": "shirt 17.5 shorts 24.5 total 42",
                "uncertainty_type": "calculation",
            },
            {
                "normalized_answer": "A",
                "claim_span": "50 minus 42 leaves 8",
                "uncertainty_type": "evidence_selection",
            },
            {
                "normalized_answer": "A",
                "claim_span": "50 minus 42 leaves 8",
                "uncertainty_type": "evidence_selection",
            },
        ]
    )
    assert summary.answer_unique_count == 1
    assert summary.answer_divergence_score == 0.0
    assert 0.0 <= summary.claim_similarity_mean <= 1.0
    assert summary.claim_divergence_score > 0.55
    assert summary.uncertainty_type_diversity_score == 0.5


def test_voc_trigger_v2_requires_spread_for_confidence_only_case() -> None:
    decision = decide_trigger(
        trigger_type="voc_trigger_v2",
        initial_disagreement=False,
        answer_divergence_score=0.0,
        claim_divergence_score=0.2,
        uncertainty_type_diversity_score=0.0,
        mean_confidence=0.94,
        confidence_spread=0.05,
        any_invalid_confidence=False,
        fail_open_to_always=False,
    )
    assert decision.triggered is False


def test_voc_trigger_v2_ignores_missing_confidence_when_other_signals_are_weak() -> None:
    decision = decide_trigger(
        trigger_type="voc_trigger_v2",
        initial_disagreement=False,
        answer_divergence_score=0.0,
        claim_divergence_score=0.1,
        uncertainty_type_diversity_score=0.0,
        mean_confidence=None,
        confidence_spread=None,
        any_invalid_confidence=True,
        fail_open_to_always=True,
    )
    assert decision.triggered is False
    assert decision.fail_open_applied is False


@pytest.mark.parametrize(
    ("raw_text", "mode"),
    [
        ('```json\n{"final_answer":"yes","reasoning":"short"}\n```', OUTPUT_MODE_CORE),
        ("The answer is yes.", OUTPUT_MODE_CORE),
        ('{"final_answer":"yes","confidence_raw":"high","reasoning":"short","key_evidence":null,"uncertain_point":null}', OUTPUT_MODE_SELECTIVE_COMM),
        ('{"final_answer":"yes","reasoning_trace":"short","claim_span":"claim","key_evidence":"evidence","keyword_clues":[],"confidence_raw":0.8,"uncertain_point":null}', OUTPUT_MODE_BUDGET_SOLVER),
        ('{"final_answer":"yes","reasoning":"short"}{"final_answer":"no","reasoning":"alt"}', OUTPUT_MODE_CORE),
        ('{"reasoning":"short"}', OUTPUT_MODE_CORE),
        ('{"final_answer":"yes","uncertainty_type":"bad_label"}', OUTPUT_MODE_SELECTIVE_COMM),
        ('{"final_answer":"yes","unexpected":"field"}', OUTPUT_MODE_CORE),
    ],
)
def test_validate_structured_output_rejects_malformed_payloads(raw_text: str, mode: str) -> None:
    with pytest.raises(ValueError):
        validate_structured_output(raw_text, mode)  # type: ignore[arg-type]


def test_extract_message_channels_supports_reasoning_metadata() -> None:
    assistant_text, provider_reasoning_text = _extract_message_channels(
        {
            "choices": [
                {
                    "message": {
                        "content": [{"type": "text", "text": '{"final_answer":"yes","reasoning":"short"}'}],
                        "reasoning": "hidden provider reasoning",
                    }
                }
            ]
        }
    )
    assert assistant_text.startswith('{"final_answer":"yes"')
    assert provider_reasoning_text == "hidden provider reasoning"


def test_generate_and_load_split_manifests(tmp_path: Path) -> None:
    source_path = tmp_path / "gsm8k.jsonl"
    source_path.write_text(
        "\n".join(
            [
                json.dumps({"question": "1+1?", "answer": "#### 2"}, ensure_ascii=False),
                json.dumps({"question": "2+2?", "answer": "#### 4"}, ensure_ascii=False),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark_path = tmp_path / "benchmark.toml"
    benchmark_path.write_text(
        "\n".join(
            [
                'name = "Toy GSM8K"',
                'slug = "toy_gsm8k"',
                'loader = "gsm8k_jsonl"',
                f'source_path = "{source_path.as_posix()}"',
                'source_split = "test"',
                'sample_id_prefix = "toy"',
                'question_field = "question"',
                'answer_field = "answer"',
                'smoke_size = 1',
                'pilot_size = 2',
                'main_size = 2',
                'random_seed = 42',
                'notes = ""',
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    benchmark = load_benchmark_config(benchmark_path)
    created = generate_split_manifests([benchmark], tmp_path / "splits")
    assert created
    smoke_ids = load_split_ids("toy_gsm8k", "count20_seed42", tmp_path / "splits")
    samples = select_samples(benchmark, "count20_seed42", tmp_path / "splits")
    assert len(smoke_ids) == 1
    assert [sample.sample_id for sample in samples] == smoke_ids


def test_request_cache_round_trip(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    record = CachedResponse(
        cache_key="abc",
        payload_json=json_dump({"a": 1}),
        response_json=json_dump({"b": 2}),
        http_status=200,
        latency_ms=12.5,
        provider_request_id="req_1",
    )
    cache.put(record)
    loaded = cache.get("abc")
    cache.close()
    assert loaded == record


def test_build_request_cache_key_depends_only_on_payload() -> None:
    payload = {"model": "demo", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.0}
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) == build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=dict(payload),
    )
    assert build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload=payload,
    ) != build_request_cache_key(
        provider="demo_provider",
        request_model="demo_model",
        payload={**payload, "temperature": 0.7},
    )


def test_cache_successful_response_rejects_failed_request(tmp_path: Path) -> None:
    cache = RequestCache(tmp_path / "requests.sqlite")
    with pytest.raises(ValueError, match="must not be cached"):
        cache_successful_response(
            cache,
            cache_key="abc",
            payload={"model": "demo"},
            response_payload={
                "http_status": 500,
                "assistant_text": "",
                "provider_reasoning_text": "",
                "latency_ms": 0.0,
                "provider_request_id": "req_failed",
                "request_error": "boom",
            },
        )
    assert cache.get("abc") is None
    cache.close()


def test_request_cache_router_shards_by_provider_and_model(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    first = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    second = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    third = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4",
        dataset="gsm8k",
    )
    router.close()

    assert first is second
    assert first.db_path != third.db_path
    assert first.db_path.name == "requests.sqlite"
    assert "deepseek-v4-flash" in first.db_path.parts
    assert "gsm8k" in first.db_path.parts


def test_resolve_cache_shard_path_matches_router(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    cache = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    router.close()

    resolved = resolve_cache_shard_path(
        tmp_path,
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    assert cache.db_path == resolved


def test_summarize_cache_root_collects_provider_stats(tmp_path: Path) -> None:
    router = RequestCacheRouter(tmp_path)
    deepseek_cache = router.for_request_target(
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    )
    dashscope_cache = router.for_request_target(
        provider="dashscope",
        request_model="qwen-turbo",
        dataset="strategyqa",
    )
    deepseek_cache.put(
        CachedResponse(
            cache_key="a",
            payload_json=json_dump({"request": 1}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=10.0,
            provider_request_id="req_a",
        )
    )
    deepseek_cache.put(
        CachedResponse(
            cache_key="b",
            payload_json=json_dump({"request": 2}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=11.0,
            provider_request_id="req_b",
        )
    )
    dashscope_cache.put(
        CachedResponse(
            cache_key="c",
            payload_json=json_dump({"request": 3}),
            response_json=json_dump({"ok": True}),
            http_status=200,
            latency_ms=12.0,
            provider_request_id="req_c",
        )
    )
    router.close()

    summary = summarize_cache_root(tmp_path)
    assert summary.shard_count == 2
    assert summary.provider_count == 2
    assert summary.total_request_count == 3
    assert summary.total_size_bytes > 0
    assert {item.provider for item in summary.providers} == {"dashscope", "deepseek"}
    assert {item.model_count for item in summary.providers} == {1}
    assert {item.dataset_count for item in summary.providers} == {1}

    shard = inspect_cache_shard(resolve_cache_shard_path(
        tmp_path,
        provider="deepseek",
        request_model="deepseek-v4-flash",
        dataset="gsm8k",
    ), tmp_path)
    assert shard.exists is True
    assert shard.request_count == 2
    assert shard.provider == "deepseek"
    assert shard.request_model == "deepseek-v4-flash"
    assert shard.dataset == "gsm8k"


def test_rate_limiter_without_waiting() -> None:
    limiter = SlidingWindowRateLimiter(requests_per_minute=100, tokens_per_minute=1000)
    started = time.monotonic()
    limiter.acquire(10)
    limiter.acquire(10)
    assert time.monotonic() - started < 1.0


def test_rate_limiter_settle_releases_reserved_tokens() -> None:
    limiter = SlidingWindowRateLimiter(
        requests_per_minute=None,
        tokens_per_minute=200,
        window_seconds=0.05,
    )
    reservation = limiter.acquire(90)
    limiter.settle(reservation, 10)
    started = time.monotonic()
    limiter.acquire(90)
    assert time.monotonic() - started < 0.02


def test_execute_completion_request_reconciles_usage() -> None:
    class FakeProvider:
        def chat_completion(self, payload: dict[str, object]) -> ProviderResponse:
            return ProviderResponse(
                http_status=200,
                raw_payload={"ok": True},
                assistant_text='{"final_answer": "42", "reasoning": "ok"}',
                provider_reasoning_text="",
                finish_reason="stop",
                usage_reported={"prompt_tokens": 8, "completion_tokens": 12, "total_tokens": 20},
                usage_estimated={"prompt_tokens": 8, "completion_tokens": 64, "total_tokens": 72},
                usage_source="reported",
                latency_ms=10.0,
                provider_request_id="req_test",
                response_id="resp_test",
            )

    limiter = SlidingWindowRateLimiter(
        requests_per_minute=None,
        tokens_per_minute=200,
        window_seconds=0.05,
    )
    payload = {
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 512,
    }
    response_payload = execute_completion_request(FakeProvider(), payload, limiter=limiter)
    assert response_payload["request_error"] is None
    assert response_payload["usage_reported"]["total_tokens"] == 20
    assert sum(event.tokens for event in limiter.token_events) == 20


def test_provider_reuses_shared_http_client(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[object] = []
    created_http2_flags: list[bool] = []

    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            created_clients.append(self)
            created_http2_flags.append(bool(kwargs.get("http2")))

        def close(self) -> None:
            return None

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("experiment_core.foundation.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider_a = None
    provider_b = None
    try:
        provider_a = OpenAICompatibleProvider(model)
        provider_b = OpenAICompatibleProvider(model)
        assert len(created_clients) == 1
        assert created_http2_flags == [False]
    finally:
        if provider_a is not None:
            provider_a.close()
        if provider_b is not None:
            provider_b.close()


def test_provider_rotates_shared_http_client_after_protocol_error(monkeypatch: pytest.MonkeyPatch) -> None:
    created_clients: list[object] = []

    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            self.closed = False
            self.should_fail = len(created_clients) == 0
            created_clients.append(self)

        def post(self, url: str, *, headers: dict[str, str], json: dict[str, object], timeout: object) -> httpx.Response:
            if self.should_fail:
                raise httpx.RemoteProtocolError(
                    "Invalid input ConnectionInputs.RECV_WINDOW_UPDATE in state ConnectionState.CLOSED"
                )
            return httpx.Response(
                200,
                request=httpx.Request("POST", url, headers=headers, json=json),
                json={
                    "id": "resp_test",
                    "choices": [
                        {
                            "message": {"content": '{"final_answer": "42"}'},
                            "finish_reason": "stop",
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                },
            )

        def close(self) -> None:
            self.closed = True

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("experiment_core.foundation.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider = None
    try:
        provider = OpenAICompatibleProvider(model)
        response = provider.chat_completion(
            {
                "model": model.model_id,
                "messages": [{"role": "user", "content": "hello"}],
                "temperature": 0.0,
                "top_p": 1.0,
                "max_tokens": 64,
            }
        )
        assert response.http_status == 200
        assert response.finish_reason == "stop"
        assert len(created_clients) == 2
        assert getattr(created_clients[0], "closed") is True
    finally:
        if provider is not None:
            provider.close()


def test_provider_close_swallows_transport_close_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class DummyClient:
        def __init__(self, **kwargs: object) -> None:
            return None

        def close(self) -> None:
            raise RuntimeError("Invalid input ConnectionInputs.RECV_WINDOW_UPDATE in state ConnectionState.CLOSED")

    model = resolve_model_ref("xiaomimimo/mimo-v2.5")
    monkeypatch.setenv(model.api_key_env, "test-key")
    monkeypatch.setattr("experiment_core.foundation.providers.httpx.Client", DummyClient)
    OpenAICompatibleProvider._shared_clients.clear()
    provider = OpenAICompatibleProvider(model)
    provider.close()
    assert provider._closed is True


def test_buffered_jsonl_writer_writes_rows(tmp_path: Path) -> None:
    target = tmp_path / "rows.jsonl"
    with target.open("w", encoding="utf-8") as handle:
        writer = BufferedJsonlWriter(handle, flush_every=2, flush_interval_seconds=60.0)
        writer.write_row({"id": 1})
        writer.write_row({"id": 2})
        writer.write_row({"id": 3})
        writer.close()
    rows = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert rows == [{"id": 1}, {"id": 2}, {"id": 3}]


def test_finalize_run_outputs_attaches_hf_publish_result(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "manifest.json").write_text(json.dumps({"run_id": "demo-run"}, ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp_path / "report.md").write_text("# report\n", encoding="utf-8")
    (tmp_path / "metrics.json").write_text(json.dumps({"summary": []}, ensure_ascii=False, indent=2), encoding="utf-8")
    monkeypatch.setattr(
        "experiment_core.foundation.runtime.publish_run_if_configured",
        lambda root, validation: {"published": True, "remote_repo": "owner/research-runs"},
    )

    payload = finalize_run_outputs(
        tmp_path,
        validator=lambda _: {"passed": True},
    )

    assert payload["hf_publish"]["published"] is True
    validation_payload = json.loads((tmp_path / "run_validation.json").read_text(encoding="utf-8"))
    assert validation_payload["hf_publish"]["remote_repo"] == "owner/research-runs"

