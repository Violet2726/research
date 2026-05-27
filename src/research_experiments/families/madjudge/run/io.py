"""MADJudge 运行路径管理。"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RunPaths:
    """MADJudge 运行的文件路径集合。"""

    run_root: Path

    @property
    def turns_path(self) -> Path:
        return self.run_root / "turns.jsonl"

    @property
    def debate_messages_path(self) -> Path:
        return self.run_root / "debate_messages.jsonl"

    @property
    def predictions_path(self) -> Path:
        return self.run_root / "predictions.jsonl"

    @property
    def metrics_path(self) -> Path:
        return self.run_root / "metrics.json"

    @property
    def debate_diagnostics_path(self) -> Path:
        return self.run_root / "debate_diagnostics.json"

    @property
    def cost_breakdown_path(self) -> Path:
        return self.run_root / "cost_breakdown.json"
