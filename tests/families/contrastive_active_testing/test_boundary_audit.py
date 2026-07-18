from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from research_experiments.core.data.datasets import DatasetSample
from research_experiments.families.contrastive_active_testing.boundary import (
    verify_frozen_v3_mechanism,
)
from research_experiments.families.contrastive_active_testing.config import (
    load_experiment_config,
    load_phase_benchmarks,
    load_protocol_config,
)
from research_experiments.families.contrastive_active_testing.run.boundary_execute import _bounded
from research_experiments.families.contrastive_active_testing.run.boundary_report import (
    evaluate_boundary_human_audit,
    materialize_boundary_artifacts,
)
from research_experiments.families.contrastive_active_testing.run.sample import (
    NetworkAttemptBudget,
    run_catch_sample,
)

EXPERIMENT = "configs/families/contrastive_active_testing/experiments/catch_cross_domain_boundary_audit.toml"


def test_boundary_config_is_nonconfirmatory_isolated_and_capped() -> None:
    experiment = load_experiment_config(EXPERIMENT)
    protocol = load_protocol_config(experiment.protocol)
    assert experiment.study_type == "post_failure_cross_domain_boundary_audit"
    assert experiment.confirmatory is False
    assert protocol.protocol_version == "catch_v3"
    assert protocol.budget_scope == "boundary_audit"
    assert protocol.max_network_attempts == 3_000
    assert [item.slug for item in load_phase_benchmarks(experiment, "boundary_audit")] == [
        "bbeh",
        "musr",
        "seqbench",
        "gpqa_diamond",
    ]
    assert set(experiment.cache_namespaces.values()) == {
        "catch-provider-audit-v1",
        "catch-boundary-v3-bbeh",
        "catch-boundary-v3-musr",
        "catch-boundary-v3-seqbench",
        "catch-boundary-v3-gpqa",
    }
    assert verify_frozen_v3_mechanism()["exact_component_hash_match"] is True


def test_boundary_report_materializes_full_researcher_contract(tmp_path: Path) -> None:
    prediction_base = {
        "run_id": "run",
        "dataset": "bbeh",
        "split": "boundary",
        "sample_id": "s1",
        "task": "task",
        "prediction": "A",
        "gold": "A",
        "score": 1.0,
        "initial_vote_prediction": "A",
        "initial_vote_score": 1.0,
        "candidate_oracle_correct": True,
        "target_oracle_correct": True,
        "triggered": True,
        "override_accepted": False,
        "corrected_by_debate": False,
        "harmed_by_debate": False,
        "total_tokens_per_question": 10,
        "network_attempts_per_question": 0,
    }
    predictions = [
        {**prediction_base, "method_name": method}
        for method in ("sc_5", "adaptive_sc_8", "catch", "direct_judge_3", "pair_judge_3")
    ]
    selector_turn = {
        "dataset": "bbeh",
        "sample_id": "s1",
        "role": "icv_selector",
        "protocol_parse_status": "ok",
        "validated_output": {"contrasts": []},
        "validated_contrasts": [],
        "dropped_contrasts": [],
        "leakage_count": 0,
    }
    router = {
        "dataset": "bbeh",
        "sample_id": "s1",
        "eligible_challengers": [],
        "validated_contrasts": [],
        "dropped_contrasts": [],
        "decision": {"resolver": "insufficient_indexed_contrast"},
        "witness_panels": [],
    }
    source_manifest = {
        "dataset_sources": [
            {"dataset": "bbeh", "revision": "rev", "sha256": "a" * 64}
        ],
        "screening_manifests": {"bbeh": {"sample_ids": ["s1"]}},
        "disagreement_manifests": {"bbeh": {"sample_ids": ["s1"]}},
    }
    materialize_boundary_artifacts(
        tmp_path,
        screening_rows=[
            {
                "dataset": "bbeh",
                "sample_id": "s1",
                "sc5_score": 1,
                "candidate_oracle_correct": True,
                "target_oracle_correct": True,
                "triggered": True,
                "candidate_count": 2,
                "invalid_stage_answer_count": 0,
            }
        ],
        turns=[selector_turn],
        routers=[router],
        predictions=predictions,
        source_manifest=source_manifest,
        checkpoints={"bbeh": {"status": "completed"}},
    )
    for relative in (
        "report.md",
        "metrics.json",
        "sample_outcomes.jsonl",
        "selector_funnel.json",
        "witness_analysis.json",
        "failure_cases.md",
        "reproducibility_manifest.json",
        "index.md",
        "diagnostics/human_audit_sample.json",
    ):
        assert (tmp_path / relative).exists(), relative
    metrics = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert metrics["confirmatory"] is False
    assert metrics["mechanism_assessment"]["heldout_authorized"] is False


def test_boundary_human_audit_recomputes_complete_blinded_annotations() -> None:
    expected_items = [
        {"dataset": "bbeh", "coordinate_sha256": f"hash-{index}"}
        for index in range(10)
    ]
    completed_items = []
    for index, item in enumerate(expected_items):
        annotation = {
            "decidable_from_source": index != 0,
            "mutually_exclusive": index != 1,
            "atomic": index != 2,
            "context_sufficient": index != 3,
            "answer_leakage": False,
        }
        completed_items.append(
            {
                **item,
                "annotator_1": dict(annotation),
                "annotator_2": dict(annotation),
                "adjudication": None,
            }
        )
    evaluation = evaluate_boundary_human_audit(
        {"items": completed_items},
        expected_sample={"items": expected_items},
    )
    assert evaluation["audit_complete"] is True
    assert evaluation["sample_contract_matches"] is True
    assert evaluation["mechanism_validity_thresholds_met"] is True
    assert evaluation["cohen_kappa_pooled_non_leakage"] == 1.0


def test_boundary_scheduler_keeps_multiple_samples_in_flight_without_exceeding_cap() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    class Progress:
        def update_scheduler_state(self, **kwargs) -> None:
            del kwargs

    def worker(value: int) -> int:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.01)
        with lock:
            active -= 1
        return value

    rows = list(_bounded(range(60), max_workers=15, worker=worker, progress=Progress()))
    assert len(rows) == 60
    assert 1 < peak <= 15


def test_boundary_selected_protocol_reuses_screening_stage_rows_without_calls() -> None:
    sample = DatasetSample(
        dataset="bbeh",
        sample_id="bbeh-shared-stage",
        question="Return yes.",
        reference_answer="yes",
        prompt_context="",
        metadata={"task": "fixture", "answer_contract": {"kind": "free_text"}},
    )
    stage_rows = tuple(
        {
            "dataset": "bbeh",
            "sample_id": sample.sample_id,
            "role": "stage_a_solver",
            "agent_id": index,
            "answer_class_key": "yes",
            "prediction": "yes",
            "normalized_answer": "yes",
            "validated_output": {"reasoning": "The source explicitly requests yes.", "final_answer": "yes"},
            "actual_total_tokens": 10,
            "total_tokens": 10,
            "network_attempt_count": 0,
        }
        for index in range(1, 6)
    )
    turns, router, predictions = run_catch_sample(
        sample,
        run_id="run",
        split_name="boundary",
        experiment=SimpleNamespace(global_seed=42),
        protocol=SimpleNamespace(protocol_version="catch_v3", stage_candidates=5),
        endpoint=None,
        network_budget=NetworkAttemptBudget(100),
        phase_name="boundary_audit",
        run_direct_judge=False,
        precomputed_stage_rows=stage_rows,
    )
    assert turns == list(stage_rows)
    assert router["triggered"] is False
    assert {row["method_name"] for row in predictions} == {"sc_5", "adaptive_sc_8", "catch"}
