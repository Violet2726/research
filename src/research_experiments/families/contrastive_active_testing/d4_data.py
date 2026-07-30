"""MuSR-X 与 SuperGPQA 科学迁移的盲金标准 manifest 工具。"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


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
    rendered_asset_relative_path: str,
    rendered_asset_sha256: str,
    render_audit_relative_path: str,
    render_audit_sha256: str,
    custodian_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "schema": "catch_d4_latent_first_sealed_manifest_v3",
        "generator_repository": str(generator_repository),
        "generator_commit": str(generator_commit),
        "generation_lock_sha256": str(generation_lock_sha256),
        "narrative_generator_id": str(narrative_generator_id),
        "quality_validation_protocol_sha256": str(quality_validation_protocol_sha256),
        "rendered_asset_relative_path": str(rendered_asset_relative_path),
        "rendered_asset_sha256": str(rendered_asset_sha256),
        "render_audit_relative_path": str(render_audit_relative_path),
        "render_audit_sha256": str(render_audit_sha256),
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
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "generator_repository",
        "generator_commit",
        "generation_lock_sha256",
        "narrative_generator_id",
        "quality_validation_protocol_sha256",
        "rendered_asset_relative_path",
        "rendered_asset_sha256",
        "render_audit_relative_path",
        "render_audit_sha256",
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
    rendered_asset_relative = Path(str(payload.get("rendered_asset_relative_path") or ""))
    render_audit_relative = Path(str(payload.get("render_audit_relative_path") or ""))
    rendered_asset_path_safe = bool(
        rendered_asset_relative.as_posix()
        and not rendered_asset_relative.is_absolute()
        and ".." not in rendered_asset_relative.parts
    )
    render_audit_path_safe = bool(
        render_audit_relative.as_posix()
        and not render_audit_relative.is_absolute()
        and ".." not in render_audit_relative.parts
    )
    linked_asset_verified = manifest_path is None
    linked_audit_verified = manifest_path is None
    if manifest_path is not None:
        manifest_root = Path(manifest_path).resolve().parent
        if rendered_asset_path_safe:
            rendered_asset_path = manifest_root / rendered_asset_relative
            linked_asset_verified = rendered_asset_path.is_file() and hashlib.sha256(
                rendered_asset_path.read_bytes()
            ).hexdigest() == str(payload.get("rendered_asset_sha256") or "")
        if render_audit_path_safe:
            render_audit_path = manifest_root / render_audit_relative
            linked_audit_verified = render_audit_path.is_file() and hashlib.sha256(
                render_audit_path.read_bytes()
            ).hexdigest() == str(payload.get("render_audit_sha256") or "")
    conditions = {
        "top_level_schema": set(payload) == required_keys,
        "schema": payload.get("schema") == "catch_d4_latent_first_sealed_manifest_v3",
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
        "rendered_asset_path": rendered_asset_path_safe,
        "rendered_asset_hash": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("rendered_asset_sha256") or ""))
        ),
        "rendered_asset_file": linked_asset_verified,
        "render_audit_path": render_audit_path_safe,
        "render_audit_hash": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("render_audit_sha256") or ""))
        ),
        "render_audit_file": linked_audit_verified,
        "independent_custodian": payload.get("independent_custodian") is True
        and bool(str(payload.get("custodian_id") or "").strip()),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "latent_hash_overlaps": overlaps,
        "record_id_overlaps": id_overlaps,
        "linked_rendered_asset": "verified" if linked_asset_verified else "missing_or_hash_mismatch",
        "linked_render_audit": "verified" if linked_audit_verified else "missing_or_hash_mismatch",
    }


def partition_text_records(
    records: Iterable[dict[str, Any]],
    *,
    counts_by_split: dict[str, int],
    seed: int,
    strata: tuple[str, ...],
) -> dict[str, list[dict[str, str]]]:
    """Split text records by committed hashes and emit no text or gold.

    This function is intended for an independent data custodian.  The returned
    rows reveal only stable IDs, strata, and hashes; question text, options and
    answers stay in the separately sealed source asset.
    """

    if not counts_by_split or any(int(value) < 0 for value in counts_by_split.values()):
        raise ValueError("Text split counts must be non-negative and non-empty.")
    if not strata or len(set(strata)) != len(strata):
        raise ValueError("Text strata must be a non-empty unique tuple.")
    by_stratum: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen_ids: set[str] = set()
    seen_question_hashes: set[str] = set()
    seen_record_hashes: set[str] = set()
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            raise ValueError(f"Text record {index} must be an object.")
        record_id = str(record.get("record_id") or record.get("uuid") or "").strip()
        stratum = str(record.get("stratum") or record.get("task") or record.get("field") or "").strip()
        question = str(record.get("question") or record.get("input") or "").strip()
        if not record_id or record_id in seen_ids:
            raise ValueError("Text records require unique non-empty record_id/uuid values.")
        if stratum not in strata:
            raise ValueError(f"Text record {record_id} has an unregistered stratum {stratum!r}.")
        if not question:
            raise ValueError(f"Text record {record_id} requires a non-empty question/input.")
        question_hash = text_sha256(question)
        record_hash = _sha256(record)
        if question_hash in seen_question_hashes or record_hash in seen_record_hashes:
            raise ValueError("Duplicate question or source-record hash encountered before sealing.")
        seen_ids.add(record_id)
        seen_question_hashes.add(question_hash)
        seen_record_hashes.add(record_hash)
        by_stratum[stratum].append(
            {
                "record_id": record_id,
                "stratum": stratum,
                "question_sha256": question_hash,
                "source_record_sha256": record_hash,
            }
        )
    for stratum, rows in by_stratum.items():
        rng = random.Random(int(_sha256(f"{seed}\0{stratum}\0text-split")[:16], 16))
        rng.shuffle(rows)

    output: dict[str, list[dict[str, str]]] = {name: [] for name in counts_by_split}
    offsets = {stratum: 0 for stratum in strata}
    for split, total in counts_by_split.items():
        base, remainder = divmod(int(total), len(strata))
        for index, stratum in enumerate(strata):
            take = base + int(index < remainder)
            start = offsets[stratum]
            end = start + take
            if end > len(by_stratum.get(stratum, [])):
                raise ValueError(f"Insufficient text records for {stratum} in split {split}.")
            output[split].extend(by_stratum[stratum][start:end])
            offsets[stratum] = end
        output[split].sort(key=lambda row: (row["stratum"], row["question_sha256"]))
    return output


def text_sealed_manifest(
    partition: dict[str, list[dict[str, str]]],
    *,
    dataset_id: str,
    source_repository: str,
    source_revision: str,
    source_asset_sha256: str,
    license_id: str,
    partition_protocol_sha256: str,
    dedup_audit_relative_path: str,
    dedup_audit_sha256: str,
    custodian_id: str,
    seed: int,
) -> dict[str, Any]:
    payload = {
        "schema": "catch_d4_text_sealed_manifest_v1",
        "dataset_id": str(dataset_id),
        "source_repository": str(source_repository),
        "source_revision": str(source_revision),
        "source_asset_sha256": str(source_asset_sha256),
        "license_id": str(license_id),
        "partition_protocol_sha256": str(partition_protocol_sha256),
        "dedup_audit_relative_path": str(dedup_audit_relative_path),
        "dedup_audit_sha256": str(dedup_audit_sha256),
        "custodian_id": str(custodian_id),
        "independent_custodian": True,
        "seed": int(seed),
        "split_order": list(partition),
        "splits": partition,
        "contains_text": False,
        "contains_gold": False,
    }
    return {**payload, "sha256": _sha256(payload)}


def validate_text_sealed_manifest(
    payload: dict[str, Any],
    *,
    expected_dataset_id: str,
    expected_counts: dict[str, int],
    expected_strata: tuple[str, ...],
    manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    required_keys = {
        "schema",
        "dataset_id",
        "source_repository",
        "source_revision",
        "source_asset_sha256",
        "license_id",
        "partition_protocol_sha256",
        "dedup_audit_relative_path",
        "dedup_audit_sha256",
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
    row_schema_valid = True
    row_values_valid = True
    balance_valid = True
    task_sets: list[set[str]] = []
    ids_by_split: dict[str, set[str]] = {}
    questions_by_split: dict[str, set[str]] = {}
    records_by_split: dict[str, set[str]] = {}
    unique_within_split = True
    for split, rows in splits.items():
        if not isinstance(rows, list):
            row_schema_valid = False
            rows = []
        ids: list[str] = []
        questions: list[str] = []
        records: list[str] = []
        counts: dict[str, int] = defaultdict(int)
        for row in rows:
            if not isinstance(row, dict) or set(row) != {
                "record_id",
                "stratum",
                "question_sha256",
                "source_record_sha256",
            }:
                row_schema_valid = False
                continue
            record_id = str(row.get("record_id") or "")
            stratum = str(row.get("stratum") or "")
            question_hash = str(row.get("question_sha256") or "")
            record_hash = str(row.get("source_record_sha256") or "")
            row_values_valid &= bool(
                record_id
                and stratum in expected_strata
                and re.fullmatch(r"[0-9a-f]{64}", question_hash)
                and re.fullmatch(r"[0-9a-f]{64}", record_hash)
            )
            ids.append(record_id)
            questions.append(question_hash)
            records.append(record_hash)
            counts[stratum] += 1
        ids_by_split[str(split)] = set(ids)
        questions_by_split[str(split)] = set(questions)
        records_by_split[str(split)] = set(records)
        unique_within_split &= (
            len(ids) == len(set(ids))
            and len(questions) == len(set(questions))
            and len(records) == len(set(records))
        )
        task_sets.append(set(counts))
        if counts:
            balance_valid &= max(counts.values()) - min(counts.values()) <= 1
    overlaps = {
        f"{left}:{right}": {
            "record_ids": len(ids_by_split[left] & ids_by_split[right]),
            "question_hashes": len(questions_by_split[left] & questions_by_split[right]),
            "source_record_hashes": len(records_by_split[left] & records_by_split[right]),
        }
        for index, left in enumerate(sorted(splits))
        for right in sorted(splits)[index + 1 :]
    }
    relative_path = Path(str(payload.get("dedup_audit_relative_path") or ""))
    safe_relative_path = bool(
        relative_path.as_posix()
        and not relative_path.is_absolute()
        and ".." not in relative_path.parts
    )
    linked_audit_verified = manifest_path is None
    linked_audit_reason = "not_requested"
    if manifest_path is not None and safe_relative_path:
        audit_path = Path(manifest_path).resolve().parent / relative_path
        linked_audit_verified = audit_path.is_file() and hashlib.sha256(audit_path.read_bytes()).hexdigest() == str(
            payload.get("dedup_audit_sha256") or ""
        )
        linked_audit_reason = "verified" if linked_audit_verified else "missing_or_hash_mismatch"
    expected_names = list(expected_counts)
    source_repository = str(payload.get("source_repository") or "")
    repository_valid = bool(re.fullmatch(r"https://[^\s]+", source_repository))
    if expected_dataset_id == "supergpqa_science":
        repository_valid &= source_repository == "https://huggingface.co/datasets/m-a-p/SuperGPQA"
    conditions = {
        "top_level_schema": set(payload) == required_keys,
        "schema": payload.get("schema") == "catch_d4_text_sealed_manifest_v1",
        "dataset_id": payload.get("dataset_id") == expected_dataset_id,
        "hash": payload.get("sha256") == _sha256(unsigned),
        "no_text": payload.get("contains_text") is False,
        "no_gold": payload.get("contains_gold") is False,
        "split_names": set(splits) == set(expected_names) and payload.get("split_order") == expected_names,
        "counts": all(len(list(splits.get(split) or [])) == count for split, count in expected_counts.items()),
        "row_schema": row_schema_valid,
        "row_values": row_values_valid,
        "unique_within_split": unique_within_split,
        "balanced_by_stratum": balance_valid
        and bool(task_sets)
        and all(task_set == set(expected_strata) for task_set in task_sets),
        "disjoint": not any(any(values.values()) for values in overlaps.values()),
        "source_repository": repository_valid,
        "source_revision": bool(re.fullmatch(r"[0-9a-f]{40}", str(payload.get("source_revision") or ""))),
        "source_asset": bool(re.fullmatch(r"[0-9a-f]{64}", str(payload.get("source_asset_sha256") or ""))),
        "license": bool(str(payload.get("license_id") or "").strip()),
        "partition_protocol": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("partition_protocol_sha256") or ""))
        ),
        "dedup_audit_path": safe_relative_path,
        "dedup_audit_hash": bool(
            re.fullmatch(r"[0-9a-f]{64}", str(payload.get("dedup_audit_sha256") or ""))
        ),
        "dedup_audit_file": linked_audit_verified,
        "independent_custodian": payload.get("independent_custodian") is True
        and bool(str(payload.get("custodian_id") or "").strip()),
    }
    return {
        "passed": all(conditions.values()),
        "conditions": conditions,
        "cross_split_overlaps": overlaps,
        "linked_dedup_audit": linked_audit_reason,
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


def write_text_sealed_manifest_from_files(
    *,
    records_path: str | Path,
    output_path: str | Path,
    dataset_id: str,
    counts_by_split: dict[str, int],
    strata: tuple[str, ...],
    source_repository: str,
    source_revision: str,
    license_id: str,
    partition_protocol_path: str | Path,
    dedup_audit_path: str | Path,
    custodian_id: str,
    seed: int,
) -> dict[str, Any]:
    source = Path(records_path)
    output = Path(output_path)
    protocol = Path(partition_protocol_path)
    audit = Path(dedup_audit_path)
    for label, path in {"records": source, "partition_protocol": protocol, "dedup_audit": audit}.items():
        if not path.is_file():
            raise ValueError(f"D4 text sealing requires an existing {label} file: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partition = partition_text_records(
        _load_sealing_records(source),
        counts_by_split=counts_by_split,
        seed=seed,
        strata=strata,
    )
    manifest = text_sealed_manifest(
        partition,
        dataset_id=dataset_id,
        source_repository=source_repository,
        source_revision=source_revision,
        source_asset_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        license_id=license_id,
        partition_protocol_sha256=hashlib.sha256(protocol.read_bytes()).hexdigest(),
        dedup_audit_relative_path=_manifest_relative_path(audit, output),
        dedup_audit_sha256=hashlib.sha256(audit.read_bytes()).hexdigest(),
        custodian_id=custodian_id,
        seed=seed,
    )
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_text_sealed_manifest(
        manifest,
        expected_dataset_id=dataset_id,
        expected_counts=counts_by_split,
        expected_strata=strata,
        manifest_path=output,
    )
    if not validation["passed"]:
        raise ValueError(f"Generated D4 text manifest failed validation: {validation}")
    return {"manifest": manifest, "validation": validation, "output_path": output.resolve().as_posix()}


def write_latent_sealed_manifest_from_files(
    *,
    latent_records_path: str | Path,
    rendered_asset_path: str | Path,
    render_audit_path: str | Path,
    output_path: str | Path,
    counts_by_split: dict[str, int],
    strata: tuple[str, ...],
    generator_repository: str,
    generator_commit: str,
    generation_lock_path: str | Path,
    narrative_generator_id: str,
    quality_validation_protocol_path: str | Path,
    custodian_id: str,
    seed: int,
) -> dict[str, Any]:
    latent_records = Path(latent_records_path)
    rendered_asset = Path(rendered_asset_path)
    render_audit = Path(render_audit_path)
    generation_lock = Path(generation_lock_path)
    quality_protocol = Path(quality_validation_protocol_path)
    output = Path(output_path)
    required = {
        "latent_records": latent_records,
        "rendered_asset": rendered_asset,
        "render_audit": render_audit,
        "generation_lock": generation_lock,
        "quality_validation_protocol": quality_protocol,
    }
    for label, path in required.items():
        if not path.is_file():
            raise ValueError(f"D4 latent sealing requires an existing {label} file: {path}")
    output.parent.mkdir(parents=True, exist_ok=True)
    partition = partition_latent_records(
        _load_sealing_records(latent_records),
        counts_by_split=counts_by_split,
        seed=seed,
        strata=strata,
    )
    manifest = sealed_manifest(
        partition,
        generator_repository=generator_repository,
        generator_commit=generator_commit,
        generation_lock_sha256=hashlib.sha256(generation_lock.read_bytes()).hexdigest(),
        narrative_generator_id=narrative_generator_id,
        quality_validation_protocol_sha256=hashlib.sha256(quality_protocol.read_bytes()).hexdigest(),
        rendered_asset_relative_path=_manifest_relative_path(rendered_asset, output),
        rendered_asset_sha256=hashlib.sha256(rendered_asset.read_bytes()).hexdigest(),
        render_audit_relative_path=_manifest_relative_path(render_audit, output),
        render_audit_sha256=hashlib.sha256(render_audit.read_bytes()).hexdigest(),
        custodian_id=custodian_id,
        seed=seed,
    )
    output.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    validation = validate_sealed_manifest(
        manifest,
        expected_counts=counts_by_split,
        expected_strata=strata,
        manifest_path=output,
    )
    if not validation["passed"]:
        raise ValueError(f"Generated D4 latent manifest failed validation: {validation}")
    return {"manifest": manifest, "validation": validation, "output_path": output.resolve().as_posix()}


def _normalize_text(value: str) -> str:
    return " ".join(str(value or "").casefold().split())


def _sha256(value: Any) -> str:
    raw = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _load_sealing_records(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        records = pq.read_table(path).to_pylist()
    elif suffix == ".jsonl":
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload if isinstance(payload, list) else payload.get("records") if isinstance(payload, dict) else None
    else:
        raise ValueError("D4 sealing supports only .jsonl, .json, or .parquet record assets.")
    if not isinstance(records, list) or not records or not all(isinstance(item, dict) for item in records):
        raise ValueError("D4 sealing source must contain a non-empty list of objects.")
    return [dict(item) for item in records]


def _manifest_relative_path(target: Path, manifest_path: Path) -> str:
    try:
        relative = target.resolve().relative_to(manifest_path.resolve().parent)
    except ValueError as exc:
        raise ValueError("Linked D4 sealing artifacts must live under the manifest directory.") from exc
    if not relative.parts or ".." in relative.parts:
        raise ValueError("Linked D4 sealing artifact path is invalid.")
    return relative.as_posix()
