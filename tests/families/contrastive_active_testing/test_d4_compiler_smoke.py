from __future__ import annotations

import hashlib
from types import SimpleNamespace

import pytest

from research_experiments.families.contrastive_active_testing import d4_compiler_smoke as smoke


def test_fixed_capability_smoke_selection_is_hash_stable_and_balanced(monkeypatch) -> None:
    capabilities = ("cap-a", "cap-b")
    samples = [
        SimpleNamespace(dataset="bbeh", sample_id=f"{capability}-{index}", capability=capability)
        for capability in capabilities
        for index in range(8)
    ]
    monkeypatch.setattr(
        smoke,
        "route_for_sample",
        lambda sample: SimpleNamespace(capability_id=sample.capability),
    )
    first = smoke.fixed_capability_smoke_selection(samples, capability_ids=capabilities, seed=42)
    second = smoke.fixed_capability_smoke_selection(samples, capability_ids=capabilities, seed=42)
    assert [(item.capability, item.sample_id) for item in first] == [
        (item.capability, item.sample_id) for item in second
    ]
    assert len(first) == 10
    assert {item.capability: sum(row.capability == item.capability for row in first) for item in first} == {
        "cap-a": 5,
        "cap-b": 5,
    }


def test_compiler_smoke_assessment_requires_all_predeclared_conditions() -> None:
    capabilities = tuple(f"cap-{index}" for index in range(9))
    rows = [
        {
            "capability_id": capability,
            "dataset": "bbeh",
            "sample_id": f"{capability}-sample-{sample_index}",
            "agent_id": compiler_index,
            "source_ir_v3_status": "ok",
            "verification_passed": True,
            "canonical_answer": "A",
            "reference_checker_status": "PASSED_FIXTURE",
            "concrete_witness_status": {"status": "PASSED"},
            "metamorphic_checks_passed": True,
            "candidate_or_gold_leakage": False,
        }
        for capability in capabilities
        for sample_index in range(5)
        for compiler_index in range(1, 4)
    ]
    result = smoke.assess_d4_source_compiler_smoke(rows, expected_capabilities=capabilities)
    assert result["passed"] is True
    assert result["summary"]["source_ir_v3_pass_count"] == 135

    for row in rows:
        if row["capability_id"] == "cap-0":
            row["verification_passed"] = False
    failed = smoke.assess_d4_source_compiler_smoke(rows, expected_capabilities=capabilities)
    assert failed["passed"] is False
    assert failed["conditions"]["at_least_122_source_ir_v3_passes"] is True
    assert failed["conditions"]["every_capability_has_one_three_way_verified_agreement"] is False


def test_terminal_smoke_cannot_be_rerun_in_a_new_directory(tmp_path) -> None:
    result_path = tmp_path / "terminal" / "result.json"
    result_path.parent.mkdir()
    result_path.write_text('{"passed": false}', encoding="utf-8")
    experiment = SimpleNamespace(
        name="fixture",
        raw={
            "kernel_revision": "d4_proof_carrying_v1",
            "source_compiler_smoke_status": "failed_blocking_downstream",
            "source_compiler_smoke_result_path": result_path.as_posix(),
            "source_compiler_smoke_result_sha256": hashlib.sha256(result_path.read_bytes()).hexdigest(),
        },
    )
    with pytest.raises(RuntimeError, match="terminal"):
        smoke.run_d4_source_compiler_smoke(
            experiment=experiment,
            backbone=SimpleNamespace(),
            run_dir=tmp_path / "new-run",
        )
