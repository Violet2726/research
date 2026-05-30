"""约束实验 phase 与 methods/setups 的引用关系，防止配置逐步失配。"""

from __future__ import annotations

import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str) -> dict:
    return tomllib.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_phase_setup_references_exist_for_multi_agent_style_experiments() -> None:
    for relative_path in (
        "configs/families/multi_agent/experiments/same_context_controlled_debate.toml",
        "configs/families/consensagent/experiments/consensagent_main.toml",
        "configs/families/madjudge/experiments/madjudge_main.toml",
    ):
        payload = _load(relative_path)
        setup_names = {item["name"] for item in payload.get("setups", [])}
        assert setup_names, relative_path
        phases = payload.get("phases", {})
        assert phases, relative_path
        for phase_name, phase_payload in phases.items():
            requested = set(phase_payload.get("setups", []))
            assert requested, f"{relative_path}:{phase_name} 缺少 setups"
            assert requested.issubset(setup_names), f"{relative_path}:{phase_name} 引用了未声明 setup"


def test_phase_method_references_exist_for_single_agent_style_experiments() -> None:
    for relative_path in (
        "configs/families/single_agent/experiments/same_context_core_benchmarks.toml",
        "configs/families/single_agent/experiments/same_context_main_table.toml",
        "configs/families/single_agent/experiments/cross_provider_robustness.toml",
        "configs/families/sid_lite/experiments/sid_lite_mechanism_validation.toml",
        "configs/families/free_mad_lite/experiments/free_mad_lite_mechanism_validation.toml",
        "configs/families/comm_necessary/experiments/hotpotqa_split_context_communication_necessity.toml",
    ):
        payload = _load(relative_path)
        declared = set(payload.get("methods", []))
        phases = payload.get("phases", {})
        assert phases, relative_path
        for phase_name, phase_payload in phases.items():
            phase_methods = set(phase_payload.get("methods", [])) if "methods" in phase_payload else set()
            if declared:
                if phase_methods:
                    assert phase_methods.issubset(declared), f"{relative_path}:{phase_name} 引用了未声明 method"
                else:
                    assert "split_suffix" in phase_payload or "split_overrides" in phase_payload, f"{relative_path}:{phase_name} 缺少执行范围"


def test_imad_declared_methods_have_unique_names_and_valid_round_limits() -> None:
    payload = _load("configs/families/imad/experiments/imad_same_context_main.toml")
    methods = payload.get("methods", [])
    names = [item["name"] for item in methods]
    assert len(names) == len(set(names))
    for item in methods:
        assert int(item["round_limit"]) >= 1
        assert item["mode"] in {"fixed", "adaptive"}
