"""Guardrails for experiment config comparator conventions and inventory."""

from __future__ import annotations

import tomllib
from pathlib import Path

from research_experiments.family_runtime.comparators import (
    COT_1,
    MAD_3A_R1,
    MAD_3A_R2,
    MAD_5A_R1,
    MAD_FIXED_R1,
    MV_5,
    MV_6,
    MV_7,
    is_protected_standard_bare_name,
)

ROOT = Path(__file__).resolve().parents[2]


def _load_toml(relative_path: str) -> dict:
    return tomllib.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_multi_agent_and_imad_budget_matched_controls_stay_aligned() -> None:
    multi_agent = _load_toml("configs/families/multi_agent/experiments/same_context_controlled_debate.toml")
    setup = multi_agent["setups"][0]
    assert setup["name"] == "mad_3a_r1"
    assert setup["matched_controls"] == [MV_6]

    imad = _load_toml("configs/families/imad/experiments/imad_same_context_main.toml")
    methods = {row["name"]: row for row in imad["methods"]}
    assert methods[MAD_FIXED_R1]["matched_controls"] == [MV_6]
    assert methods["mad_fixed_r2"]["matched_controls"] == []
    assert methods["mad_fixed_r3"]["matched_controls"] == []
    assert methods["imad_adaptive"]["matched_controls"] == []


def test_consensagent_and_madjudge_controls_stay_aligned() -> None:
    consensagent = _load_toml("configs/families/consensagent/experiments/consensagent_main.toml")
    setups = {row["name"]: row for row in consensagent["setups"]}
    assert setups["consensagent_3a"]["matched_controls"] == [COT_1, MV_6]
    assert setups["mad_3a_r1"]["matched_controls"] == [COT_1, MV_6]
    assert setups["mad_3a_r2"]["matched_controls"] == [COT_1, MV_6]

    madjudge = _load_toml("configs/families/madjudge/experiments/madjudge_main.toml")
    judge_setups = {row["name"]: row for row in madjudge["setups"]}
    assert judge_setups["madjudge_7a"]["matched_controls"] == [COT_1, MV_7]
    assert judge_setups["madjudge_5a"]["matched_controls"] == [COT_1, MV_5]


def test_baseline_compare_inventory_stays_aligned() -> None:
    payload = _load_toml("configs/families/baseline_compare/experiments/core_six_method_baseline.toml")
    control_catalog = _load_toml(payload["control_catalog"])
    control_methods = payload["control_methods"]
    method_order = payload["method_order"]
    setup_names = [item["name"] for item in payload["setups"]]

    assert control_methods == [COT_1, "sc_3", "sc_5"]
    assert setup_names == [MAD_3A_R1, MAD_3A_R2, MAD_5A_R1]
    assert set(method_order) == set(control_methods) | set(setup_names)
    assert set(control_methods).issubset(set(control_catalog["methods"]))


def test_dmad_family_local_methods_do_not_use_protected_bare_names() -> None:
    targets = [
        "configs/families/dmad/experiments/dmad_reasoning_main.toml",
        "configs/families/dmad/experiments/dmad_reasoning_appendix.toml",
        "configs/families/dmad/experiments/dmad_reasoning_extended.toml",
        "configs/families/dmad/experiments/dmad_joint_cross_domain_exploration.toml",
    ]
    for path in targets:
        payload = _load_toml(path)
        for method in payload.get("methods", []):
            name = str(method["name"])
            if name.startswith("mad_") or name.startswith("dmad_"):
                continue
            assert not is_protected_standard_bare_name(name), f"{path} uses protected bare method name {name}"


def test_adaptive_sparse_mad_experiments_directory_only_keeps_current_mainline_configs() -> None:
    experiments_dir = ROOT / "configs" / "families" / "adaptive_sparse_mad" / "experiments"
    assert not (experiments_dir / "_archive").exists()
    assert sorted(path.name for path in experiments_dir.glob("*.toml")) == [
        "same_context_full_counterfactual_v1.toml",
        "same_context_full_counterfactual_v1_screen.toml",
        "same_context_main_v5.toml",
    ]
