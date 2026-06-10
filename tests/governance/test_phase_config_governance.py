"""约束实验 phase 与 methods/setups 的引用关系，防止配置逐步失配。"""

from __future__ import annotations

import re
import tomllib
from collections.abc import Iterator
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _load(relative_path: str) -> dict:
    return tomllib.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def _iter_phase_split_entries(phase_payload: dict) -> Iterator[tuple[str, str]]:
    """统一展开 phase 的 split 引用，避免治理规则各自解析一遍。"""
    if "split_suffix" in phase_payload:
        yield "*", str(phase_payload["split_suffix"])
    for dataset_name, split_name in (phase_payload.get("split_overrides") or {}).items():
        yield dataset_name, str(split_name)


def _load_benchmarks_by_slug(experiment_payload: dict) -> dict[str, dict]:
    """加载 experiment 显式声明的 benchmark，用于校验 split 例外是否有数据规模依据。"""
    benchmarks: dict[str, dict] = {}
    for relative_path in experiment_payload.get("benchmark_configs", []):
        benchmark = _load(relative_path)
        benchmarks[str(benchmark["slug"])] = benchmark
    return benchmarks


def test_phase_setup_references_exist_for_multi_agent_style_experiments() -> None:
    for relative_path in (
        "configs/families/multi_agent/experiments/same_context_controlled_debate.toml",
        "configs/families/multi_agent/experiments/standard_baseline_controls.toml",
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
        "configs/families/single_agent/experiments/canonical_simple_baselines.toml",
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


def test_canonical_single_agent_baseline_config_is_fixed_to_temp_0p7() -> None:
    experiment = _load("configs/families/single_agent/experiments/canonical_simple_baselines.toml")
    catalog = _load(experiment["method_catalog"])
    assert experiment["cot_uses_reruns"] is True
    assert experiment["phases"]["count100"]["split_overrides"]["competition_math"] == "count100_seed42"
    assert experiment["phases"]["count100"]["methods"] == ["cot_1", "mv_3", "sc_5"]
    assert experiment["phases"]["count100"]["reruns_override"] == 3

    for method_name in ("cot_1", "mv_3", "sc_5"):
        assert catalog["methods"][method_name]["temperature"] == 0.7


def test_count_phases_do_not_point_to_different_count_splits() -> None:
    for path in sorted((ROOT / "configs/families").rglob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        for phase_name, phase_payload in (payload.get("phases") or {}).items():
            phase_match = re.fullmatch(r"count(?P<count>\d+)(?:_.+)?", phase_name)
            if phase_match is None:
                continue
            expected_count = int(phase_match.group("count"))
            for dataset_name, split_name in _iter_phase_split_entries(phase_payload):
                split_match = re.fullmatch(r"count(?P<count>\d+)_seed\d+", str(split_name))
                if split_match is None:
                    continue
                actual_count = int(split_match.group("count"))
                assert actual_count == expected_count, f"{path}:{phase_name}:{dataset_name} -> {split_name}"


def test_count_phase_full_split_fallbacks_match_declared_dataset_size() -> None:
    """允许小数据集在大 count phase 使用 full split，但必须与 benchmark 主规模一致。"""
    for path in sorted((ROOT / "configs/families").rglob("*.toml")):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        benchmarks = _load_benchmarks_by_slug(payload)
        for phase_name, phase_payload in (payload.get("phases") or {}).items():
            phase_match = re.fullmatch(r"count(?P<count>\d+)(?:_.+)?", phase_name)
            if phase_match is None:
                continue
            expected_count = int(phase_match.group("count"))
            for dataset_name, split_name in _iter_phase_split_entries(phase_payload):
                split_match = re.fullmatch(r"full(?P<size>\d+)_seed\d+", split_name)
                if split_match is None:
                    continue
                full_size = int(split_match.group("size"))
                if dataset_name == "*":
                    candidates = benchmarks.values()
                else:
                    assert dataset_name in benchmarks, f"{path}:{phase_name}:{dataset_name} 未在 benchmark_configs 中声明"
                    candidates = [benchmarks[dataset_name]]
                for benchmark in candidates:
                    declared_size = int(benchmark["main_size"])
                    assert declared_size == full_size, f"{path}:{phase_name}:{dataset_name} -> {split_name}"
                    assert declared_size <= expected_count, f"{path}:{phase_name}:{dataset_name} -> {split_name}"


def test_imad_declared_methods_have_unique_names_and_valid_round_limits() -> None:
    payload = _load("configs/families/imad/experiments/imad_same_context_main.toml")
    methods = payload.get("methods", [])
    names = [item["name"] for item in methods]
    assert len(names) == len(set(names))
    for item in methods:
        assert int(item["round_limit"]) >= 1
        assert item["mode"] in {"fixed", "adaptive"}
