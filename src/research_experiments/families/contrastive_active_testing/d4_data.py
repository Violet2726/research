"""MuSR-X 与 SuperGPQA 科学迁移的盲金标准 manifest 工具。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from typing import Any


def latent_graph_hash(graph: Any) -> str:
    """Hash the latent reasoning graph before any natural-language rendering."""

    return _sha256(graph)


def partition_latent_records(
    records: Iterable[dict[str, Any]],
    *,
    counts_by_split: dict[str, int],
    seed: int,
    strata: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    """Partition by latent hash, balanced by task, without reading text/gold."""

    if not counts_by_split or any(int(value) < 0 for value in counts_by_split.values()):
        raise ValueError("Latent split counts must be non-negative and non-empty.")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("Latent strata must be a non-empty unique tuple.")
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_hashes: set[str] = set()
    seen_record_ids: set[str] = set()
    for record in records:
        task = str(record.get("task") or "")
        if task not in strata:
            continue
        record_id = str(record.get("record_id") or "")
        if not record_id or record_id in seen_record_ids:
            raise ValueError("Latent records require unique non-empty record_id values.")
        seen_record_ids.add(record_id)
        if "latent_graph" not in record or record.get("latent_graph") is None:
            raise ValueError("Latent records require a non-empty latent_graph before rendering.")
        graph_hash = latent_graph_hash(record.get("latent_graph"))
        if graph_hash in seen_hashes:
            raise ValueError("Duplicate latent graph hash encountered before rendering.")
        seen_hashes.add(graph_hash)
        by_stratum[task].append(
            {
                "record_id": record_id,
                "task": task,
                "latent_graph_sha256": graph_hash,
            }
        )
    for task, rows in by_stratum.items():
        rng = random.Random(int(_sha256(f"{seed}\0{task}\0latent-split")[:16], 16))
        rng.shuffle(rows)

    output: dict[str, list[dict[str, str]]] = {name: [] for name in counts_by_split}
    offsets = {task: 0 for task in strata}
    for split, total in counts_by_split.items():
        base, remainder = divmod(int(total), len(strata))
        for index, task in enumerate(strata):
            take = base + int(index < remainder)
            start = offsets[task]
            end = start + take
            if end > len(by_stratum.get(task, [])):
                raise ValueError(f"Insufficient latent records for {task} in split {split}.")
            output[split].extend(by_stratum[task][start:end])
            offsets[task] = end
        output[split].sort(key=lambda row: (row["task"], row["latent_graph_sha256"]))
    return output


def sealed_manifest(
    partition: dict[str, list[dict[str, str]]],
    *,
    generator_repository: str,
    generator_commit: str,
    generation_lock_sha256: str,
    narrative_generator_id: str,
    quality_validation_protocol_sha256: str,
    custodian_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "schema": "catch_d4_latent_first_sealed_manifest_v2",
        "generator_repository": str(generator_repository),
        "generator_commit": str(generator_commit),
        "generation_lock_sha256": str(generation_lock_sha256),
        "narrative_generator_id": str(narrative_generator_id),
        "quality_validation_protocol_sha256": str(quality_validation_protocol_sha256),
        "custodian_id": str(custodian_id),
        "independent_custodian": True,
        "seed": int(seed),
        "split_order": list(partition),
        "splits": partition,
        "contains_text": False,
        "contains_gold": False,
    }
    return {**payload, "sha256": _sha256(payload)}


def validate_sealed_manifest(
    payload: dict[str, Any],
    *,
    expected_counts: dict[str, int],
    expected_strata: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "generator_repository",
        "generator_commit",
        "generation_lock_sha256",
        "narrative_generator_id",
        "quality_validation_protocol_sha256",
        "custodian_id",
        "independent_custodian",
        "seed",
        "split_order",
        "splits",
        "contains_text",
        "contains_gold",
        "sha256",
    }
    unsigned = {key: value for key, value in payload.items() if key != "sha256"}
    raw_splits = payload.get("splits")
    splits = dict(raw_splits) if isinstance(raw_splits, dict) else {}
    expected_names = list(expected_counts)
    hashes_by_split: dict[str, set[str]] = {}
    ids_by_split: dict[str, set[str]] = {}
    row_schema_valid = True
    values_valid = True
    unique_within_split = True
    balance_valid = True
    tasks_by_split: dict[str, set[str]] = {}
    for split, rows in splits.items():
        if not isinstance(rows, list):
            row_schema_valid = False
            rows = []
        task_counts: dict[str, int] = defaultdict(int)
        raw_hashes: list[str] = []
        raw_ids: list[str] = []
        for row in rows:
            if not isinstance(row, dict) or set(row) != {"record_id", "task", "latent_graph_sha256"}:
                row_schema_valid = False
                continue
            record_id = str(row.get("record_id") or "")
            task = str(row.get("task") or "")
            graph_hash = str(row.get("latent_graph_sha256") or "")
            values_valid &= bool(record_id and task and re.fullmatch(r"[0-9a-f]{64}", graph_hash))
            raw_ids.append(record_id)
            raw_hashes.append(graph_hash)
            task_counts[task] += 1
        hashes_by_split[str(split)] = {
            value for value in raw_hashes if value
        }
        ids_by_split[str(split)] = {value for value in raw_ids if value}
        tasks_by_split[str(split)] = set(task_counts)
        unique_within_split &= len(raw_hashes) == len(set(raw_hashes)) and len(raw_ids) == len(set(raw_ids))
        if task_counts:
            balance_valid &= max(task_counts.values()) - min(task_counts.values()) <= 1
    overlaps = {
        f"{left}:{right}": len(hashes_by_split[left] & hashes_by_split[right])
        for index, left in enumerate(sorted(hashes_by_split))
        for right in sorted(hashes_by_split)[index + 1 :]
    }
    id_overlaps = {
        f"{left}:{right}": len(ids_by_split[left] & ids_by_split[right])
        for index, left in enumerate(sorted(ids_by_split))
        for right in sorted(ids_by_split)[index + 1 :]
    }
    task_sets = list(tasks_by_split.values())
    consistent_task_sets = bool(task_sets) and all(value == task_sets[0] for value in task_sets)
    conditions = {
        "top_level_schema": set(payload) == required_keys,
        "schema": payload.get("schema") == "catch_d4_latent_first_sealed_manifest_v2",
        "hash": payload.get("sha256") == _sha256(unsigned),
        "no_text": payload.get("contains_text") is False,
        "no_gold": payload.get("contains_gold") is False,
        "split_names": set(splits) == set(expected_names) and payload.get("split_order") == expected_names,
        "row_schema": row_schema_valid,
        "row_values": values_valid,
        "unique_within_split": unique_within_split,
        "balanced_by_task": balance_valid
        and consistent_task_sets
        and (
            expected_strata is None
            or (bool(task_sets) and task_sets[0] == set(expected_strata))
        ),
        "counts": all(len(list(splits.get(split) or [])) == count for split, count in expected_counts.items()),
        "disjoint": not any(overlaps.values()) and not any(id_overlaps.values()),
        "official_generator": payload.get("generator_repository")
        == "https://github.com/Zayne-Sprague/MuSR",
        "generator_commit": bool(re.fullmatch(r"[0-9a-f]{40}", str(payload.get("generator_commit") or ""))),
        "generation_lock": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("generation_lock_sha256") or ""))
        ),
        "narrative_generator": bool(str(payload.get("narrative_generator_id") or "").strip()),
        "quality_validation_protocol": bool(
            re.fullmatch(
                r"[0-9a-f]{64}",
                str(payload.get("quality_validation_protocol_sha256") or ""),
            )
        ),
        "independent_custodian": payload.get("independent_custodian") is True
        and bool(str(payload.get("custodian_id") or "").strip()),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "latent_hash_overlaps": overlaps,
        "record_id_overlaps": id_overlaps,
    }


def text_sha256(text: str) -> str:
    return hashlib.sha256(_normalize_text(text).encode("utf-8")).hexdigest()


def minhash_signature(text: str, *, permutations: int = 64, shingle_size: int = 5) -> tuple[int, ...]:
    tokens = re.findall(r"\w+", _normalize_text(text))
    shingles = {
        " ".join(tokens[index : index + shingle_size])
        for index in range(max(1, len(tokens) - shingle_size + 1))
    } or {""}
    signature = []
    for index in range(permutations):
        signature.append(min(int(_sha256(f"{index}\0{value}")[:16], 16) for value in shingles))
    return tuple(signature)


def estimated_minhash_similarity(left: tuple[int, ...], right: tuple[int, ...]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    return sum(a == b for a, b in zip(left, right, strict=True)) / len(left)


def science_record_eligible(record: dict[str, Any]) -> bool:
    discipline = str(record.get("discipline") or "").casefold()
    domain = str(record.get("domain") or record.get("field") or "").casefold()
    modality = str(record.get("modality") or "text").casefold()
    options = record.get("options")
    inferred_choice = isinstance(options, (list, dict)) and len(options) >= 2
    answer_type = str(
        record.get("answer_type") or ("single_choice" if inferred_choice else "unknown")
    ).casefold()
    return (
        discipline == "science"
        and domain in {"physics", "chemistry", "biology"}
        and modality == "text"
        and answer_type == "single_choice"
    )


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()
