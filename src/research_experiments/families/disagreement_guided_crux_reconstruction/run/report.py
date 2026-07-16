"""DGCR 汇总与报告入口。"""

from __future__ import annotations

import json
from pathlib import Path


def summarize_run(run_dir: str | Path) -> dict:
    root = Path(run_dir)
    return json.loads((root / "run_summary.json").read_text(encoding="utf-8"))


def render_report(run_dir: str | Path, publish_dir: str | Path | None = None) -> Path:
    del publish_dir
    root = Path(run_dir)
    return root / "report.md"
