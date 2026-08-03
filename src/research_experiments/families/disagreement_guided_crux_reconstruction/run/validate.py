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
    if (
        manifest.get("cache_policy") != "global_validated_response_v3"
        or manifest.get("request_source") != "global_validated_response_cache"
    ):
        raise ValueError("DGCR manifest lacks the global validated-response cache policy.")
    turns = [json.loads(line) for line in (root / "turns" / "agent_turns.jsonl").read_text(encoding="utf-8").splitlines() if line]
    invalid_turns = [
        row for row in turns
        if row.get("cache_policy") != manifest["cache_policy"]
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
