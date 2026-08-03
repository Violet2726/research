"""将旧 namespace 响应分片离线迁移到全局有效响应缓存。"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from research_experiments.core.execution.cache import (
    CACHE_KEY_POLICY_VERSION,
    REQUESTS_INSERT_SQL,
    REQUESTS_TABLE_SCHEMA,
    build_request_cache_key,
    cache_rejection_reason,
    json_dump,
    normalize_payload_for_cache_key,
    resolve_cache_shard_path,
)
from research_experiments.family_runtime.free_text_protocol import parse_free_text_answer_output
from research_experiments.family_runtime.json_object_protocol import parse_json_object_answer_output


def inspect_cache_rekey(source_root: str | Path) -> dict[str, Any]:
    """Inspect a cache root without writing files or making provider calls."""

    source = Path(source_root).resolve()
    shards = sorted(path for path in source.rglob("*.sqlite") if path.is_file())
    report, _ = _collect_rekey_candidates(source, shards)
    report["mode"] = "dry_run"
    return report


def apply_cache_rekey(
    source_root: str | Path,
    *,
    temporary_root: str | Path | None = None,
    activation_root: str | Path | None = None,
) -> dict[str, Any]:
    """Build, validate and atomically replace the active cache root.

    ``activation_root`` is used when an immutable pre-migration backup is the
    source.  The source remains untouched; only the validated temporary tree
    is swapped into the requested active location.
    """

    source = Path(source_root).resolve()
    if not source.is_dir():
        raise ValueError(f"Cache root does not exist: {source}")
    activation = Path(activation_root or source_root).resolve()
    if not activation.is_dir():
        raise ValueError(f"Activation cache root does not exist: {activation}")
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    target = (
        Path(temporary_root).resolve()
        if temporary_root is not None
        else activation.with_name(f"{activation.name}.global-rekeying-{timestamp}")
    )
    if target in {source, activation} or target.exists():
        raise ValueError(f"Temporary cache root must be new and distinct: {target}")
    if target.parent.resolve() != activation.parent.resolve():
        raise ValueError("Temporary cache root must share the activation parent for atomic replacement.")

    shards = sorted(path for path in source.rglob("*.sqlite") if path.is_file())
    report, candidates = _collect_rekey_candidates(source, shards)
    _write_rekeyed_cache(target, candidates)
    validation = _validate_rekey_target(target, report["output_row_count"])
    if not validation["passed"]:
        raise RuntimeError(f"Rekey target validation failed: {validation}")

    backup = activation.with_name(f"{activation.name}.before-global-{timestamp}")
    if backup.exists():
        raise RuntimeError(f"Refusing to overwrite cache backup: {backup}")
    activation.replace(backup)
    try:
        target.replace(activation)
    except BaseException:
        backup.replace(activation)
        raise

    report.update(
        {
            "mode": "applied",
            "cache_key_policy": CACHE_KEY_POLICY_VERSION,
            "cache_layout": "global_provider_model_dataset_v3",
            "backup_root": backup.as_posix(),
            "active_root": activation.as_posix(),
            "validation": validation,
            "output_shards": _shard_digests(activation),
        }
    )
    report_path = activation.parent / f"{activation.name}.global-rekey-{timestamp}.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8")
    report["report_path"] = report_path.as_posix()
    return report


def _collect_rekey_candidates(
    source_root: Path,
    shards: list[Path],
) -> tuple[dict[str, Any], dict[Path, dict[str, tuple[tuple[Any, ...], dict[str, Any]]]]]:
    dropped: Counter[str] = Counter()
    collision_count = 0
    input_rows = 0
    output_rows = 0
    candidates: dict[Path, dict[str, tuple[tuple[Any, ...], dict[str, Any]]]] = {}
    shard_reports: list[dict[str, Any]] = []

    for shard in shards:
        relative = shard.relative_to(source_root)
        connection = sqlite3.connect(f"file:{shard.as_posix()}?mode=ro", uri=True)
        try:
            integrity = str(connection.execute("PRAGMA integrity_check").fetchone()[0])
            if integrity.lower() != "ok":
                raise RuntimeError(f"Cache shard integrity check failed: {shard}: {integrity}")
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(requests)").fetchall()
            }
            if not columns:
                dropped["missing_requests_table"] += 1
                shard_reports.append(_shard_report(relative, 0, 0))
                continue
            rows = _read_source_rows(connection, columns)
            shard_input = 0
            shard_candidates = 0
            for row in rows:
                input_rows += 1
                shard_input += 1
                rowid, created_at, payload_json, response_json, completion_tokens = row
                try:
                    payload = json.loads(payload_json)
                    response = json.loads(response_json)
                except (TypeError, json.JSONDecodeError):
                    dropped["invalid_json"] += 1
                    continue
                if not isinstance(payload, dict) or not isinstance(response, dict):
                    dropped["invalid_record_shape"] += 1
                    continue
                rejection = _migration_rejection_reason(relative, payload, response)
                if rejection is not None:
                    dropped[rejection] += 1
                    continue
                validated_output = _validated_output_for_migration(relative, payload, response)
                if validated_output is None:
                    dropped["unverifiable_protocol_output"] += 1
                    continue
                shard_candidates += 1
                identity = _request_identity(relative, payload)
                if identity is None:
                    dropped["unrecoverable_provider_identity"] += 1
                    continue
                provider, request_model, dataset = identity
                cache_key = build_request_cache_key(
                    provider=provider,
                    request_model=request_model,
                    payload=payload,
                )
                target_relative = resolve_cache_shard_path(
                    Path("."),
                    provider=provider,
                    request_model=request_model,
                    dataset=dataset,
                ).relative_to(Path("."))
                candidate_sort_key = (
                    str(created_at or "9999-12-31T23:59:59Z"),
                    relative.as_posix(),
                    int(rowid),
                )
                serialized = {
                    "cache_key": cache_key,
                    "payload_json": json_dump(normalize_payload_for_cache_key(payload)),
                    "response_json": json_dump(_project_response(response, validated_output)),
                    "completion_tokens": _completion_tokens(response, completion_tokens),
                }
                previous = candidates.setdefault(target_relative, {}).get(cache_key)
                if previous is not None:
                    collision_count += 1
                    if candidate_sort_key >= previous[0]:
                        continue
                candidates[target_relative][cache_key] = (candidate_sort_key, serialized)
            shard_reports.append(_shard_report(relative, shard_input, shard_candidates))
        finally:
            connection.close()

    output_rows = sum(len(rows) for rows in candidates.values())
    report = {
        "schema": "global_cache_migration_report_v1",
        "cache_key_policy": CACHE_KEY_POLICY_VERSION,
        "cache_layout": "global_provider_model_dataset_v3",
        "source_root": source_root.as_posix(),
        "input_shard_count": len(shards),
        "input_row_count": input_rows,
        "output_row_count": output_rows,
        "dropped_by_reason": dict(sorted(dropped.items())),
        "collision_count": collision_count,
        "collision_winner": "earliest_created_at_then_shard_path_then_rowid",
        "historical_validation": "strict_current_protocol_inferred_shared_parser; specialized_unverifiable_dropped",
        "shards": shard_reports,
    }
    return report, candidates


def _read_source_rows(connection: sqlite3.Connection, columns: set[str]):
    if {"created_at", "http_status", "latency_ms", "provider_request_id"}.issubset(columns):
        return connection.execute(
            """
            SELECT rowid, created_at, payload_json, response_json, NULL
            FROM requests
            ORDER BY created_at ASC, rowid ASC
            """
        )
    if {"payload_json", "response_json", "completion_tokens"}.issubset(columns):
        return connection.execute(
            """
            SELECT rowid, NULL, payload_json, response_json, completion_tokens
            FROM requests
            ORDER BY rowid ASC
            """
        )
    return []


def _shard_report(relative: Path, input_count: int, output_count: int) -> dict[str, Any]:
    return {
        "source_shard": relative.as_posix(),
        "input_row_count": input_count,
        "output_row_count": output_count,
    }


def _migration_rejection_reason(relative_shard: Path, payload: dict[str, Any], response: dict[str, Any]) -> str | None:
    reason = cache_rejection_reason(response)
    if reason is not None:
        return reason
    if _is_retired_d4_payload(relative_shard, payload):
        return "retired_d4_protocol"
    if _is_mainline_d4_tagged_payload(relative_shard, payload):
        try:
            parse_free_text_answer_output(
                str(response.get("assistant_text") or ""),
                dataset=_dataset_from_shard(relative_shard),
            )
        except ValueError:
            return "d4_tagged_protocol_failure"
    return None


def _validated_output_for_migration(
    relative_shard: Path,
    payload: dict[str, Any],
    response: dict[str, Any],
) -> dict[str, Any] | None:
    existing = response.get("validated_output")
    if isinstance(existing, dict) and existing:
        return dict(existing)
    if _is_mainline_d4_tagged_payload(relative_shard, payload):
        try:
            return parse_free_text_answer_output(
                str(response.get("assistant_text") or ""),
                dataset=_dataset_from_shard(relative_shard),
            )
        except ValueError:
            return None
    prompt = _all_message_text(payload)
    assistant_text = str(response.get("assistant_text") or "")
    # Reconstruct the two shared family-runtime validators when the prompt
    # carries their unambiguous protocol declaration.  This is deliberately
    # conservative: specialized JSON objects (DGCR, CRED-V, D4 compiler, …)
    # are not admitted without their family-specific proof validator.
    if "return only the requested tagged lines" in prompt or "reasoning, final_answer" in prompt:
        try:
            return parse_free_text_answer_output(assistant_text, dataset=_dataset_from_shard(relative_shard))
        except ValueError:
            return None
    if (
        "return one json object only" in prompt
        or "return strict json only" in prompt
        or "json answer object" in prompt
    ):
        try:
            return parse_json_object_answer_output(assistant_text, dataset=_dataset_from_shard(relative_shard))
        except ValueError:
            return None
    # Keep the repository's explicit v2 compiler replay fixture importable;
    # the live compiler caller still runs its full SourceIR proof chain and
    # evicts this marker on a failed hit.  The dedicated smoke namespace is
    # rejected above and never reaches this branch.
    if "kernel-d4" in relative_shard.as_posix().lower() and "candidate-blind" in prompt:
        return {"compiler_v2_replay_requires_proof": True}
    return None


def _request_identity(relative_shard: Path, payload: dict[str, Any]) -> tuple[str, str, str] | None:
    parts = relative_shard.parts
    try:
        provider_index = parts.index("providers")
        provider = str(parts[provider_index + 1])
        model = str(payload.get("model") or "").strip()
        dataset = "/".join(parts[provider_index + 3 : -1])
    except (ValueError, IndexError):
        return None
    if not provider or not model or not dataset:
        return None
    return provider, model, dataset


def _dataset_from_shard(relative_shard: Path) -> str:
    parts = relative_shard.parts
    try:
        start = parts.index("providers") + 3
    except ValueError:
        return ""
    return "/".join(parts[start:-1])


def _all_message_text(payload: dict[str, Any]) -> str:
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    return "\n".join(str(item.get("content") or "") for item in messages if isinstance(item, dict)).lower()


def _is_retired_d4_payload(relative_shard: Path, payload: dict[str, Any]) -> bool:
    path_text = relative_shard.as_posix().lower()
    if "kernel-d4" not in path_text:
        # The source-only compiler smoke used a content-addressed namespace
        # without the historical ``kernel-d4`` path component.  Its prompt is
        # still unambiguously the retired JSON compiler protocol.
        text = _all_message_text(payload)
        if "d4-source-compiler-smoke" not in path_text:
            return False
    elif "d4-source-compiler-smoke" not in path_text:
        text = _all_message_text(payload)
        # A v2 compiler record in the ordinary D4 development shard is a
        # current, source-only artifact; only the explicit smoke namespace is
        # retired from the active cache.
        if "compiler protocol: catch_d4_candidate_blind_compiler_v1" not in text:
            return any(marker in text for marker in ("return strict json only", "answer_first", "final_answer, reasoning"))
        return True
    text = _all_message_text(payload)
    return any(
        marker in text
        for marker in (
            "return strict json only",
            "answer_first",
            "final_answer, reasoning",
            "compiler protocol: catch_d4_candidate_blind_compiler_v1",
            "candidate-blind source compiler",
            "return exactly one json object",
        )
    )


def _is_mainline_d4_tagged_payload(relative_shard: Path, payload: dict[str, Any]) -> bool:
    if "kernel-d4" not in relative_shard.as_posix().lower():
        return False
    text = _all_message_text(payload)
    return (
        "reasoning, final_answer" in text
        and not _is_retired_d4_payload(relative_shard, payload)
    )


def _project_response(response: dict[str, Any], validated_output: dict[str, Any]) -> dict[str, Any]:
    return {
        "assistant_text": str(response.get("assistant_text") or ""),
        "provider_reasoning_text": str(response.get("provider_reasoning_text") or ""),
        "finish_reason": "stop",
        "validated_output": validated_output,
    }


def _completion_tokens(response: dict[str, Any], fallback: Any) -> int | None:
    usage = response.get("usage_reported") or response.get("usage_estimated") or {}
    value = usage.get("completion_tokens") if isinstance(usage, dict) else None
    if value is None:
        value = fallback
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed >= 0 else None


def _write_rekeyed_cache(
    target_root: Path,
    candidates: dict[Path, dict[str, tuple[tuple[Any, ...], dict[str, Any]]]],
) -> None:
    target_root.mkdir(parents=True, exist_ok=False)
    for relative, rows_by_key in candidates.items():
        target = target_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(target)
        try:
            connection.execute("PRAGMA journal_mode=DELETE")
            connection.execute(REQUESTS_TABLE_SCHEMA)
            ordered = [item[1] for _, item in sorted(rows_by_key.items())]
            connection.executemany(
                REQUESTS_INSERT_SQL,
                [
                    (
                        row["cache_key"],
                        row["payload_json"],
                        row["response_json"],
                        row["completion_tokens"],
                    )
                    for row in ordered
                ],
            )
            connection.commit()
        finally:
            connection.close()


def _validate_rekey_target(target_root: Path, expected_row_count: int) -> dict[str, Any]:
    shards = sorted(target_root.rglob("*.sqlite"))
    row_count = 0
    checks: list[bool] = []
    key_mismatches = 0
    invalid_rows = 0
    for shard in shards:
        relative = shard.relative_to(target_root)
        connection = sqlite3.connect(shard)
        try:
            checks.append(str(connection.execute("PRAGMA integrity_check").fetchone()[0]).lower() == "ok")
            row_count += int(connection.execute("SELECT COUNT(*) FROM requests").fetchone()[0])
            identity = _request_identity(relative, {"model": relative.parts[2] if len(relative.parts) > 2 else ""})
            if identity is None:
                invalid_rows += 1
                continue
            provider, _path_model, _dataset = identity
            for stored_key, payload_json, response_json in connection.execute(
                "SELECT cache_key, payload_json, response_json FROM requests"
            ):
                try:
                    payload = json.loads(payload_json)
                    response = json.loads(response_json)
                except (TypeError, json.JSONDecodeError):
                    invalid_rows += 1
                    continue
                request_model = str(payload.get("model") or "")
                if not request_model:
                    invalid_rows += 1
                    continue
                expected_key = build_request_cache_key(
                    provider=provider,
                    request_model=request_model,
                    payload=payload,
                )
                if stored_key != expected_key:
                    key_mismatches += 1
                if cache_rejection_reason(response) is not None or not isinstance(response.get("validated_output"), dict):
                    invalid_rows += 1
        finally:
            connection.close()
    return {
        "passed": (
            bool(checks)
            and all(checks)
            and row_count == expected_row_count
            and key_mismatches == 0
            and invalid_rows == 0
            and not (target_root / "namespaces").exists()
        ),
        "shard_count": len(shards),
        "row_count": row_count,
        "expected_row_count": expected_row_count,
        "key_mismatches": key_mismatches,
        "invalid_rows": invalid_rows,
        "namespaces_directory_present": (target_root / "namespaces").exists(),
    }


def _shard_digests(root: Path) -> list[dict[str, str]]:
    return [
        {
            "shard": shard.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(shard.read_bytes()).hexdigest(),
        }
        for shard in sorted(root.rglob("*.sqlite"))
    ]
