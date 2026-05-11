"""跨 family 参考 run 的治理测试。"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"


def test_family_runners_do_not_directly_scan_selective_comm_runs() -> None:
    forbidden_markers = (
        'default_runs_root("selective_comm")',
        "trigger_early_exit_main",
    )
    allowed_path = SRC / "experiment_core" / "orchestration" / "reference_runs.py"
    violations: list[str] = []
    for path in SRC.rglob("runner.py"):
        if path == allowed_path:
            continue
        if path.parent.name == "selective_comm":
            continue
        text = path.read_text(encoding="utf-8")
        if any(marker in text for marker in forbidden_markers):
            violations.append(path.relative_to(ROOT).as_posix())
    assert not violations, violations


def test_sid_lite_runner_uses_neutral_schema_ids() -> None:
    text = (SRC / "sid_lite" / "runner.py").read_text(encoding="utf-8")
    assert "SPARC" not in text
    assert "SCHEMA_ANSWER_WITH_PROXY_SIGNALS_DELIBERATION" in text
    assert "SCHEMA_BELIEF_UPDATE_DELTA" in text
