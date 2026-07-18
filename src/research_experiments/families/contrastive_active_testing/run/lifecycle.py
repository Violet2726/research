"""CATCH 实验的 best-effort 非阻断收尾工具。"""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_experiments.families.contrastive_active_testing.run.validate import validate_run


def write_nonblocking_validation(run_dir: str | Path) -> dict[str, Any]:
    """Always attempt to write compatibility validation without archives or gates."""

    root = Path(run_dir)
    target = root / "run_validation.json"
    try:
        payload = validate_run(root)
    except Exception as exc:  # validation can never invalidate collected results
        payload = {
            "passed": False,
            "artifact_valid": False,
            "run_status": "completed_with_errors",
            "scientific_gate_applicable": False,
            "scientific_gate_passed": None,
            "performance_gate_passed": None,
            "artifact_errors": ["validator_exception"],
            "artifact_violations": ["validator_exception"],
            "warnings": [],
            "scientific_violations": [],
            "validator_exception": {
                "error_type": type(exc).__name__,
                "message": str(exc),
            },
            "archive_integrity": {"applicable": False, "passed": None},
            "counts": {},
        }
    payload["generated_at"] = datetime.now(UTC).isoformat()
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload


def render_report_with_fallback(
    run_dir: str | Path,
    renderer: Callable[[str | Path, str | Path | None], dict[str, Any]],
) -> dict[str, Any]:
    """Keep raw results usable even if the rich Markdown renderer has a bug."""

    root = Path(run_dir)
    try:
        return renderer(root, None)
    except Exception as exc:
        target = root / "report.md"
        target.write_text(
            "\n".join(
                [
                    "# CATCH run — report generation incomplete",
                    "",
                    "The experiment records were preserved, but the detailed report renderer failed.",
                    "",
                    f"- Error: `{type(exc).__name__}: {exc}`",
                    "- Raw turns: `turns/agent_turns.jsonl`",
                    "- Predictions: `views/predictions.jsonl`",
                    "- Metrics: `views/metrics.json`",
                    "",
                ]
            ),
            encoding="utf-8",
        )
        return {
            "report_path": target.as_posix(),
            "error_type": type(exc).__name__,
            "message": str(exc),
        }
