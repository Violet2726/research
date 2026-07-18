"""Deterministic data views and sampling for the post-failure boundary audit.

固定四数据集的数据视图、分层抽样与源文件完整性契约。
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import replace
from pathlib import Path
from typing import Any

from research_experiments.core.data.datasets import DatasetSample, resolve_dataset_source_path

BOUNDARY_STUDY_TYPE = "post_failure_cross_domain_boundary_audit"
BOUNDARY_DATASETS = ("bbeh", "musr", "seqbench", "gpqa_diamond")
FROZEN_BBEH_V3_MECHANISM_SHA256 = {
    "algorithms.py": "93e99dd51da8c88df8e83343d7bd537a2269f3dc64a56732c5a90fdc35027090",
    "icv.py": "d36b2a9f4e435e952f65ebc7b4daedd197005e303f1c6d05822a2c68ee11c73d",
    "prompts.py": "10adbfc68c74f0a377aca15724fed41693d8f5d5411861b098e696ca1058442d",
}


def verify_frozen_v3_mechanism() -> dict[str, Any]:
    """Prove prompt, selector validator, and decoder match failed BBEH v3."""

    family_root = Path(__file__).resolve().parent
    observed = {
        name: _sha256_file(family_root / name)
        for name in FROZEN_BBEH_V3_MECHANISM_SHA256
    }
    mismatches = {
        name: {"expected": expected, "observed": observed[name]}
        for name, expected in FROZEN_BBEH_V3_MECHANISM_SHA256.items()
        if observed[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"Frozen BBEH-v3 mechanism files changed: {mismatches}")
    return {
        "source_run_id": "20260718T090517Z-xiaomimimo-mimo-v2.5",
        "source_status": "failed_structural_preflight",
        "exact_component_hash_match": True,
        "component_sha256": observed,
    }


def verify_source_asset(benchmark) -> dict[str, Any]:
    """Verify a pinned source before any provider call is admitted."""

    path = resolve_dataset_source_path(benchmark.source_path)
    if not path.exists():
        raise FileNotFoundError(f"Boundary-audit dataset source is missing: {path}")
    actual_sha = _sha256_file(path)
    expected_sha = str(benchmark.source_sha256 or "").lower()
    if not benchmark.source_revision or not expected_sha:
        raise ValueError(f"{benchmark.slug} must freeze source_revision and source_sha256.")
    if actual_sha != expected_sha:
        raise RuntimeError(
            f"{benchmark.slug} source SHA-256 changed: expected={expected_sha}, actual={actual_sha}"
        )
    return {
        "dataset": benchmark.slug,
        "path": path.as_posix(),
        "revision": benchmark.source_revision,
        "sha256": actual_sha,
        "size_bytes": path.stat().st_size,
        "source_url": benchmark.source_url,
    }


def boundary_sample_view(sample: DatasetSample) -> DatasetSample:
    """Inline detached options so frozen judges and solvers share one visible task.

    BBEH already stores its answer region in ``question``.  GPQA and MuSR use
    ``prompt_context`` in their ordinary loaders, so the audit creates a local
    immutable view instead of changing those loaders' historical prompt shape.
    """

    metadata = dict(sample.metadata)
    if sample.dataset == "gpqa_diamond":
        metadata.setdefault("task", str(metadata.get("high_level_domain") or "unknown"))
    contract = metadata.get("answer_contract")
    if not sample.prompt_context or not isinstance(contract, dict):
        return replace(sample, metadata=metadata)
    separator = "\n\n"
    combined = f"{sample.question.rstrip()}{separator}{sample.prompt_context.strip()}"
    start = len(sample.question.rstrip()) + len(separator)
    updated_contract = dict(contract)
    updated_contract.update(
        {
            "block_start": start,
            "block_end": len(combined),
            "source_style": "titled_terminal",
        }
    )
    metadata["answer_contract"] = updated_contract
    metadata["options_block_start"] = start
    metadata["options_block_end"] = len(combined)
    return replace(sample, question=combined, prompt_context="", metadata=metadata)


def select_screening_samples(
    dataset: str,
    samples: list[DatasetSample],
    *,
    count: int = 100,
    seed: int = 42,
) -> list[DatasetSample]:
    """Apply the frozen dataset-native screening allocation."""

    if len(samples) < count:
        raise ValueError(f"{dataset} contains only {len(samples)} samples; {count} required.")
    if dataset == "musr":
        quotas = {"murder_mysteries": 34, "object_placements": 33, "team_allocation": 33}
        return _select_by_fixed_quotas(samples, quotas, field="domain", seed=seed)
    if dataset == "gpqa_diamond":
        return _select_largest_remainder(
            samples,
            count=count,
            field="high_level_domain",
            seed=seed,
            minimum_one=True,
        )
    if dataset == "seqbench":
        return _select_seqbench(samples, count=count, seed=seed)
    return sorted(samples, key=lambda sample: _stable_hash(seed, dataset, sample.sample_id))[:count]


def select_disagreement_states(states: Iterable[Any], *, count: int = 20, seed: int = 42) -> list[Any]:
    """Gold-free, stratum-round-robin selection used after Stage-A screening."""

    grouped: dict[str, list[Any]] = defaultdict(list)
    for state in states:
        if not state.stage.triggered:
            continue
        stratum = boundary_stratum(state.sample)
        grouped[stratum].append(state)
    for stratum, rows in grouped.items():
        rows.sort(key=lambda state: _stable_hash(seed, stratum, state.sample.sample_id))
    strata = sorted(grouped, key=lambda value: _stable_hash(seed, "stratum", value))
    selected: list[Any] = []
    while strata and len(selected) < count:
        remaining: list[str] = []
        for stratum in strata:
            if grouped[stratum] and len(selected) < count:
                selected.append(grouped[stratum].pop(0))
            if grouped[stratum]:
                remaining.append(stratum)
        strata = remaining
    return selected


def boundary_stratum(sample: DatasetSample) -> str:
    if sample.dataset == "seqbench":
        return f"B{int(sample.metadata.get('backtracking_count_B') or 0)}_N{float(sample.metadata.get('noise_ratio_N') or 0):g}"
    if sample.dataset == "gpqa_diamond":
        return str(sample.metadata.get("high_level_domain") or "unknown")
    return str(sample.metadata.get("task") or sample.metadata.get("domain") or "unknown")


def stable_payload_hash(payload: Any) -> str:
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _select_by_fixed_quotas(
    samples: list[DatasetSample], quotas: dict[str, int], *, field: str, seed: int
) -> list[DatasetSample]:
    grouped = _group_samples(samples, field)
    selected: list[DatasetSample] = []
    for key, quota in quotas.items():
        rows = sorted(grouped.get(key, []), key=lambda sample: _stable_hash(seed, key, sample.sample_id))
        if len(rows) < quota:
            raise ValueError(f"Stratum {key!r} has {len(rows)} rows but quota is {quota}.")
        selected.extend(rows[:quota])
    return selected


def _select_largest_remainder(
    samples: list[DatasetSample], *, count: int, field: str, seed: int, minimum_one: bool
) -> list[DatasetSample]:
    grouped = _group_samples(samples, field)
    quotas = _largest_remainder_quotas(
        {key: len(rows) for key, rows in grouped.items()},
        count=count,
        seed=seed,
        minimum_one=minimum_one,
    )
    return _select_by_fixed_quotas(samples, quotas, field=field, seed=seed)


def _select_seqbench(samples: list[DatasetSample], *, count: int, seed: int) -> list[DatasetSample]:
    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        grouped[boundary_stratum(sample)].append(sample)
    quotas = _largest_remainder_quotas(
        {key: len(rows) for key, rows in grouped.items()}, count=count, seed=seed, minimum_one=True
    )
    selected: list[DatasetSample] = []
    for cell in sorted(grouped, key=lambda key: _stable_hash(seed, "cell", key)):
        rows = sorted(
            grouped[cell],
            key=lambda sample: (int(sample.metadata.get("logical_depth_L") or 0), sample.sample_id),
        )
        deciles: list[list[DatasetSample]] = [[] for _ in range(10)]
        for rank, sample in enumerate(rows):
            decile = min(9, rank * 10 // len(rows))
            deciles[decile].append(sample)
        for index, bucket in enumerate(deciles):
            bucket.sort(key=lambda sample: _stable_hash(seed, f"{cell}:L{index}", sample.sample_id))
        interleaved: list[DatasetSample] = []
        while any(deciles):
            for bucket in deciles:
                if bucket:
                    interleaved.append(bucket.pop(0))
        selected.extend(interleaved[: quotas[cell]])
    return selected


def _largest_remainder_quotas(
    sizes: dict[str, int], *, count: int, seed: int, minimum_one: bool
) -> dict[str, int]:
    nonempty = {key: size for key, size in sizes.items() if size > 0}
    base = {key: (1 if minimum_one else 0) for key in nonempty}
    if sum(base.values()) > count:
        raise ValueError("Requested count is below the number of mandatory nonempty strata.")
    remaining = count - sum(base.values())
    residual_sizes = {key: max(0, size - base[key]) for key, size in nonempty.items()}
    total = sum(residual_sizes.values())
    raw = {key: (remaining * residual_sizes[key] / total if total else 0.0) for key in nonempty}
    quotas = {key: base[key] + int(raw[key]) for key in nonempty}
    leftover = count - sum(quotas.values())
    order = sorted(
        nonempty,
        key=lambda key: (-(raw[key] - int(raw[key])), _stable_hash(seed, "quota", key)),
    )
    for key in order[:leftover]:
        quotas[key] += 1
    if any(quotas[key] > nonempty[key] for key in quotas):
        raise ValueError("Largest-remainder quota exceeds a source stratum.")
    return quotas


def _group_samples(samples: list[DatasetSample], field: str) -> dict[str, list[DatasetSample]]:
    grouped: dict[str, list[DatasetSample]] = defaultdict(list)
    for sample in samples:
        grouped[str(sample.metadata.get(field) or "unknown")].append(sample)
    return grouped


def _stable_hash(seed: int, namespace: str, value: str) -> str:
    return hashlib.sha256(f"{seed}\0{namespace}\0{value}".encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
