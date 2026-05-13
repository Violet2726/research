"""跨 family 参考 run 接口测试。"""

from __future__ import annotations

from pathlib import Path
import json

import pytest

from research_experiments.core.config import ResolvedModelConfig
from research_experiments.families.shared.reference_runs import (
    TriggerReferenceConfig,
    resolve_trigger_reference_selection,
    write_policy_reference_summary,
)


def test_write_policy_reference_summary_writes_machine_readable_payload(tmp_path: Path) -> None:
    manifest = {
        "family_name": "selective_comm",
        "experiment_name": "trigger_early_exit_main",
        "phase_name": "count20",
        "run_id": "20260511T010203Z-demo",
        "primary_model_ref": "deepseek/deepseek-v4-flash",
        "resolved_model": {
            "name": "deepseek/deepseek-v4-flash",
            "provider": "deepseek",
            "model_id": "deepseek-v4-flash",
        },
    }
    metrics_payload = {
        "summary": [
            {
                "dataset": "overall",
                "method_name": "always_communicate",
                "method_kind": "policy",
                "question_count": 20,
                "accuracy_mean": 0.80,
                "trigger_rate": 1.0,
                "early_exit_rate": 0.0,
                "total_tokens_mean": 120.0,
                "communication_tokens_mean": 40.0,
            },
            {
                "dataset": "overall",
                "method_name": "hybrid_trigger",
                "method_kind": "policy",
                "question_count": 20,
                "accuracy_mean": 0.78,
                "trigger_rate": 0.5,
                "early_exit_rate": 0.5,
                "total_tokens_mean": 90.0,
                "communication_tokens_mean": 20.0,
            },
            {
                "dataset": "gsm8k",
                "method_name": "hybrid_trigger",
                "method_kind": "policy",
                "question_count": 10,
                "accuracy_mean": 0.80,
            },
        ]
    }

    payload = write_policy_reference_summary(tmp_path, manifest=manifest, metrics_payload=metrics_payload)
    assert payload["kind"] == "policy_reference_summary"
    assert payload["run_id"] == "20260511T010203Z-demo"
    assert set(payload["overall_policies"]) == {"always_communicate", "hybrid_trigger"}

    written = json.loads((tmp_path / "policy_reference_summary.json").read_text(encoding="utf-8"))
    assert written["model_name"] == "deepseek/deepseek-v4-flash"


def test_resolve_trigger_reference_selection_prefers_exact_model_and_keeps_default_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_ROOT", tmp_path.as_posix())
    _write_reference_summary(
        tmp_path / "selective_comm" / "trigger_early_exit_main" / "count20" / "20260511T010203Z-exact",
        model_name="deepseek/deepseek-v4-flash",
        provider="deepseek",
        model_id="deepseek-v4-flash",
        always_accuracy=0.80,
        default_accuracy=0.79,
        question_count=20,
    )
    _write_reference_summary(
        tmp_path / "selective_comm" / "trigger_early_exit_main" / "count20" / "20260511T010000Z-prefix",
        model_name="deepseek/deepseek-v4",
        provider="deepseek",
        model_id="deepseek-v4",
        always_accuracy=0.80,
        default_accuracy=0.60,
        question_count=20,
    )
    decision = resolve_trigger_reference_selection(
        backbone=_resolved_model("deepseek/deepseek-v4-flash", "deepseek", "deepseek-v4-flash"),
        reference=TriggerReferenceConfig(
            source_family="selective_comm",
            source_experiment="trigger_early_exit_main",
            source_phase="count20",
            default_policy="hybrid_trigger",
            fallback_policy="disagreement_triggered",
            drop_questions_threshold=1.0,
        ),
    )
    assert decision["selected_policy"] == "hybrid_trigger"
    assert decision["reference_model_name"] == "deepseek/deepseek-v4-flash"
    assert decision["reason"] == "default_policy_kept"


def test_resolve_trigger_reference_selection_falls_back_when_drop_exceeds_threshold(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_ROOT", tmp_path.as_posix())
    _write_reference_summary(
        tmp_path / "selective_comm" / "trigger_early_exit_main" / "count20" / "20260511T010203Z-demo",
        model_name="deepseek/deepseek-v4-flash",
        provider="deepseek",
        model_id="deepseek-v4-flash",
        always_accuracy=0.80,
        default_accuracy=0.60,
        question_count=20,
    )
    decision = resolve_trigger_reference_selection(
        backbone=_resolved_model("deepseek/deepseek-v4-flash", "deepseek", "deepseek-v4-flash"),
        reference=TriggerReferenceConfig(
            source_family="selective_comm",
            source_experiment="trigger_early_exit_main",
            source_phase="count20",
            default_policy="hybrid_trigger",
            fallback_policy="disagreement_triggered",
            drop_questions_threshold=1.0,
        ),
    )
    assert decision["selected_policy"] == "disagreement_triggered"
    assert decision["reason"] == "default_policy_drops_more_than_threshold"
    assert decision["drop_questions"] == 4.0


def test_resolve_trigger_reference_selection_returns_default_when_reference_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RESEARCH_RUNS_ROOT", tmp_path.as_posix())
    decision = resolve_trigger_reference_selection(
        backbone=_resolved_model("deepseek/deepseek-v4-flash", "deepseek", "deepseek-v4-flash"),
        reference=TriggerReferenceConfig(
            source_family="selective_comm",
            source_experiment="trigger_early_exit_main",
            source_phase="count20",
            default_policy="hybrid_trigger",
            fallback_policy="disagreement_triggered",
            drop_questions_threshold=1.0,
        ),
    )
    assert decision["selected_policy"] == "hybrid_trigger"
    assert decision["reason"] == "reference_missing"


def _write_reference_summary(
    run_dir: Path,
    *,
    model_name: str,
    provider: str,
    model_id: str,
    always_accuracy: float,
    default_accuracy: float,
    question_count: int,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "kind": "policy_reference_summary",
        "family_name": "selective_comm",
        "experiment_name": "trigger_early_exit_main",
        "phase_name": "count20",
        "run_id": run_dir.name,
        "run_dir": run_dir.as_posix(),
        "model_name": model_name,
        "provider": provider,
        "model_id": model_id,
        "overall_policies": {
            "always_communicate": {
                "method_name": "always_communicate",
                "question_count": question_count,
                "accuracy_mean": always_accuracy,
            },
            "hybrid_trigger": {
                "method_name": "hybrid_trigger",
                "question_count": question_count,
                "accuracy_mean": default_accuracy,
            },
        },
    }
    (run_dir / "policy_reference_summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _resolved_model(name: str, provider: str, model_id: str) -> ResolvedModelConfig:
    return ResolvedModelConfig(
        name=name,
        provider=provider,
        model_id=model_id,
        base_url="https://example.invalid",
        api_key_env="API_KEY",
        chat_path="/chat/completions",
        default_temperature=0.0,
        default_top_p=1.0,
        default_max_output_tokens=256,
        reasoning_effort="none",
        supports_response_format=True,
        response_format="json_object",
        timeout_seconds=30,
        max_retries=2,
        tags=["general_qa"],
    )

