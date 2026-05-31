"""family 运行校验的共享低层辅助。"""

from __future__ import annotations

import json
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Any

from research_experiments.reporting.run_figures import validate_figure_contract
from research_experiments.workspace.run_archives import validate_archive_contract


def summarize_turn_statuses(turn_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize output_status counts across turn-level records."""

    request_failures = sum(1 for row in turn_rows if row.get("output_status") == "request_fail")
    schema_failures = sum(1 for row in turn_rows if row.get("output_status") == "schema_fail")
    ok_count = sum(1 for row in turn_rows if row.get("output_status") == "ok")
    total = len(turn_rows)
    return {
        "request_failures": request_failures,
        "schema_failures": schema_failures,
        "ok_count": ok_count,
        "total_turns": total,
        "output_success_rate": round(ok_count / total, 4) if total else 0.0,
    }


def validate_shared_contracts(run_dir: str | Path) -> dict[str, Any]:
    """Run shared figure and archive contract validation."""

    root = Path(run_dir)
    return {
        "figure_contract": validate_figure_contract(root),
        "archive_contract": validate_archive_contract(root),
    }


def missing_relative_paths(root: str | Path, required_paths: list[Path]) -> list[str]:
    """收集某个 run 根目录下缺失的相对路径列表。"""

    resolved_root = Path(root)
    return [path.relative_to(resolved_root).as_posix() for path in required_paths if not path.exists()]


def validate_rate_limit_check(
    progress_path: str | Path,
    turn_rows: list[dict[str, Any]],
    *,
    manifest: dict[str, Any] | None = None,
    requests_per_minute_limit: int | None = None,
    tokens_per_minute_limit: int | None = None,
) -> dict[str, Any]:
    """Validate that a run stayed within its configured rate limits."""

    progress_file = Path(progress_path)
    progress_payload = load_json(progress_file) if progress_file.exists() else {}
    progress_429_count = max(0, int(progress_payload.get("rate_limit_429_count") or 0))

    rpm_limit = _optional_int(
        requests_per_minute_limit
        if requests_per_minute_limit is not None
        else (manifest or {}).get("requests_per_minute_limit")
    )
    tpm_limit = _optional_int(
        tokens_per_minute_limit
        if tokens_per_minute_limit is not None
        else (manifest or {}).get("tokens_per_minute_limit")
    )

    events: list[tuple[datetime, int, dict[str, Any]]] = []
    for row in turn_rows:
        if row.get("cache_hit"):
            continue
        timestamp = row.get("request_started_at")
        if not timestamp:
            continue
        events.append(
            (
                _parse_timestamp(str(timestamp)),
                max(0, int(row.get("estimated_request_tokens") or 0)),
                row,
            )
        )
    events.sort(key=lambda item: item[0])

    active_window: deque[tuple[datetime, int, dict[str, Any]]] = deque()
    active_token_total = 0
    violations: list[dict[str, Any]] = []
    for timestamp, estimated_tokens, row in events:
        while active_window and (timestamp - active_window[0][0]).total_seconds() >= 60.0:
            _, expired_tokens, _ = active_window.popleft()
            active_token_total -= expired_tokens
        active_window.append((timestamp, estimated_tokens, row))
        active_token_total += estimated_tokens

        if rpm_limit is not None and len(active_window) > rpm_limit:
            violations.append(_rate_violation("rpm", len(active_window), rpm_limit, row))
        if tpm_limit is not None and active_token_total > tpm_limit:
            violations.append(_rate_violation("tpm", active_token_total, tpm_limit, row))

    return {
        "passed": progress_429_count == 0 and not violations,
        "enabled": bool(rpm_limit or tpm_limit),
        "progress_present": progress_file.exists(),
        "progress_429_count": progress_429_count,
        "network_event_count": len(events),
        "event_replay_available": bool(events),
        "requests_per_minute_limit": rpm_limit,
        "tokens_per_minute_limit": tpm_limit,
        "violation_count": len(violations),
        "violations": violations[:20],
    }


def load_json(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_json_if_present(path: str | Path) -> dict[str, Any]:
    """Read a UTF-8 JSON file, or return an empty payload when it is absent."""

    resolved = Path(path)
    if not resolved.exists():
        return {}
    return load_json(resolved)


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file."""

    with Path(path).open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_jsonl_if_present(path: str | Path) -> list[dict[str, Any]]:
    """Read a UTF-8 JSONL file, or return an empty list when it is absent."""

    resolved = Path(path)
    if not resolved.exists():
        return []
    return load_jsonl(resolved)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _rate_violation(kind: str, observed: int, limit: int, row: dict[str, Any]) -> dict[str, Any]:
    return {
        "kind": kind,
        "observed": observed,
        "limit": limit,
        "dataset": row.get("dataset"),
        "sample_id": row.get("sample_id"),
        "method_name": row.get("method_name"),
        "agent_id": row.get("agent_id"),
    }


def _optional_int(value: object) -> int | None:
    if value is None or value == "":
        return None
    return int(value)
