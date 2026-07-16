"""DGCR 运行产物校验。"""

from __future__ import annotations

import json
from pathlib import Path


def validate_run(run_dir: str | Path) -> dict[str, object]:
    root = Path(run_dir)
    required = [root / "manifest.json", root / "views" / "metrics.json", root / "views" / "predictions.jsonl", root / "turns" / "agent_turns.jsonl", root / "turns" / "router_decisions.jsonl", root / "diagnostics" / "gate.json"]
    missing = [path.as_posix() for path in required if not path.exists()]
    if missing:
        raise ValueError("DGCR run is missing artifacts: " + ", ".join(missing))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    if not manifest.get("cache_namespace") or manifest.get("request_source") != "fresh_dgcr_confirmation_cache":
        raise ValueError("DGCR manifest lacks an isolated cache namespace or source declaration.")
    if manifest["cache_namespace"] not in {"dgcr-dev-v1", "dgcr-heldout-v1"}:
        raise ValueError("DGCR run used an unexpected cache namespace.")
    turns = [json.loads(line) for line in (root / "turns" / "agent_turns.jsonl").read_text(encoding="utf-8").splitlines() if line]
    invalid_turns = [
        row for row in turns
        if row.get("cache_namespace") != manifest["cache_namespace"]
        or row.get("request_source") != "dgcr_confirmation_cache"
        or not isinstance(row.get("payload"), dict)
        or "raw_finish_reason" not in row
        or "network_attempt_count" not in row
        or "reasoning_tokens" not in row
        or row.get("usage_source") != "reported"
        or row.get("actual_total_tokens") is None
    ]
    if invalid_turns:
        raise ValueError("DGCR turns are missing required request/audit fields or actual-token reporting.")
    return {"passed": True, "family_name": "disagreement_guided_crux_reconstruction"}
