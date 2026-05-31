"""约束标准 comparator 的共享实现入口，防止同名方法再次分叉。"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_standard_vanilla_mad_runners_do_not_use_tail_override_aliases() -> None:
    targets = [
        ROOT / "src" / "research_experiments" / "families" / "multi_agent" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "imad" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "consensagent" / "run" / "sample.py",
    ]
    banned_markers = (
        "_run_mad_sample_shared",
        "_run_method_sample_original",
        "_run_consensagent_sample_original",
        "_original = _run_",
    )
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for marker in banned_markers:
            assert marker not in text, f"{path} 仍包含覆写式旧入口标记 {marker}"


def test_standard_vanilla_mad_runners_call_shared_core_directly() -> None:
    targets = [
        ROOT / "src" / "research_experiments" / "families" / "multi_agent" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "imad" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "consensagent" / "run" / "sample.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "run_shared_vanilla_mad_rounds(" in text, f"{path} 没有直接调用共享 vanilla MAD core"
        assert "build_shared_vanilla_mad_prediction(" in text, f"{path} 没有直接调用共享 prediction builder"
        if "imad" in path.as_posix() or "consensagent" in path.as_posix():
            assert "CONTROLLED_PROMPT_VERSION" in text, f"{path} 没有显式锁定共享 controlled prompt 版本"


def test_stage_a_mv3_reuse_families_call_shared_builder() -> None:
    targets = [
        ROOT / "src" / "research_experiments" / "families" / "budget_comm" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "selective_comm" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "sid_lite" / "run" / "sample.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "build_stage_a_mv3_prediction(" in text, f"{path} 没有复用共享 mv_3 builder"


def test_single_agent_and_selective_comm_reuse_shared_no_comm_core() -> None:
    targets = [
        ROOT / "src" / "research_experiments" / "families" / "single_agent" / "run" / "sample.py",
        ROOT / "src" / "research_experiments" / "families" / "selective_comm" / "run" / "sample.py",
    ]
    for path in targets:
        text = path.read_text(encoding="utf-8")
        assert "run_unified_control_sample(" in text, f"{path} 没有复用共享 no-comm sample helper"

    selective_comm_text = targets[1].read_text(encoding="utf-8")
    assert "def _run_control_method(" in selective_comm_text
    assert "build_cot_messages" not in selective_comm_text, "selective_comm 仍在本地重建 cot prompt"
    assert "build_mv_messages" not in selective_comm_text, "selective_comm 仍在本地重建 mv prompt"
