"""Gold-blind merger for frozen CATCH comparator runs on identical Stage-A candidates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def merge_comparison_predictions(
    primary_run: str | Path,
    comparator_runs: list[str | Path],
    output_path: str | Path,
) -> dict[str, Any]:
    """Merge comparator predictions only after exact Stage-A signature checks."""

    primary = Path(primary_run)
    primary_predictions = _read_jsonl(primary / "views" / "predictions.jsonl")
    primary_turns = _read_jsonl(primary / "turns" / "agent_turns.jsonl")
    primary_signatures = _stage_signatures(primary_turns)
    primary_keys = {
        (str(item.get("dataset")), str(item.get("sample_id")))
        for item in primary_predictions
        if item.get("method_name") == "catch_kernel"
    }
    if not primary_keys:
        raise ValueError("Primary run contains no CATCH-Kernel predictions.")
    merged = list(primary_predictions)
    imported_methods: dict[str, int] = {}
    source_runs: dict[str, str] = {}
    existing = {(str(item.get("dataset")), str(item.get("sample_id")), str(item.get("method_name"))) for item in merged}
    for raw_run in comparator_runs:
        run = Path(raw_run)
        predictions = _read_jsonl(run / "views" / "predictions.jsonl")
        signatures = _stage_signatures(_read_jsonl(run / "turns" / "agent_turns.jsonl"))
        methods = {str(item.get("method_name")) for item in predictions}
        allowed = methods & {"catch", "catch_cert_v2"}
        if len(allowed) != 1:
            raise ValueError(f"Comparator run must contain exactly one frozen comparator method: {run}")
        method = next(iter(allowed))
        rows = [
            item
            for item in predictions
            if item.get("method_name") == method
            and (str(item.get("dataset")), str(item.get("sample_id"))) in primary_keys
        ]
        row_keys = {(str(item.get("dataset")), str(item.get("sample_id"))) for item in rows}
        if row_keys != primary_keys:
            missing = sorted(primary_keys - row_keys)
            raise ValueError(f"Comparator {method} does not cover the frozen primary set; missing={missing[:5]}")
        for key in sorted(primary_keys):
            if signatures.get(key) != primary_signatures.get(key):
                raise ValueError(f"Stage-A candidate signature mismatch for {method}:{key}")
        imported = 0
        for row in rows:
            key = (str(row.get("dataset")), str(row.get("sample_id")), method)
            if key in existing:
                continue
            merged.append({**row, "comparison_source_run": run.resolve().as_posix()})
            existing.add(key)
            imported += 1
        imported_methods[method] = imported
        source_runs[method] = run.resolve().as_posix()
    merged.sort(
        key=lambda item: (
            str(item.get("dataset")),
            str(item.get("sample_id")),
            str(item.get("method_name")),
        )
    )
    target = Path(output_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(json.dumps(item, ensure_ascii=False) + "\n" for item in merged), encoding="utf-8")
    return {
        "schema_version": "catch_kernel_comparison_merge_v1",
        "primary_sample_count": len(primary_keys),
        "prediction_count": len(merged),
        "imported_methods": imported_methods,
        "source_runs": source_runs,
        "output_sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "output": target.resolve().as_posix(),
    }


def _stage_signatures(rows: list[dict[str, Any]]) -> dict[tuple[str, str], str]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("role") != "stage_a_solver":
            continue
        key = (str(row.get("dataset")), str(row.get("sample_id")))
        grouped.setdefault(key, []).append(row)
    signatures: dict[tuple[str, str], str] = {}
    for key, values in grouped.items():
        payload = [
            {
                "agent_id": int(item.get("agent_id") or 0),
                "answer_class_key": str(item.get("answer_class_key") or ""),
                "prediction": str(item.get("prediction") or ""),
            }
            for item in sorted(values, key=lambda item: int(item.get("agent_id") or 0))
        ]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        signatures[key] = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return signatures


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
