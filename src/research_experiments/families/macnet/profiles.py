"""MacNet 角色 profile 读取与分配。"""

from __future__ import annotations

from pathlib import Path
import json
import random
import zipfile

from research_experiments.core.data.datasets import resolve_dataset_source_path


BUILTIN_PROFILES = {
    "actor": [
        "You are a careful solver. Preserve valid work, refine weak steps, and produce one concise final artifact.",
        "You are a synthesis-oriented solver. Merge upstream hints, keep only evidence-backed reasoning, and avoid unnecessary verbosity.",
    ],
    "critic": [
        "You are a strict reviewer. Point out the single highest-impact weakness and give one actionable revision instruction.",
        "You are a topology critic. Preserve strong parts, surface one risk, and write a short directive for the downstream actor.",
    ],
}

PROFILE_CATEGORY_HINTS = {
    "mmlu": ["Science", "Reference_Books", "Language", "Business"],
    "humaneval": ["Development", "Data", "Tools_Utilities", "Science"],
    "commongen_hard": ["Language", "Culture", "Entertainment", "Reference_Books"],
}


def load_profile_bank(profile_asset_path: str | Path) -> dict[str, list[str]]:
    """从官方 zip 里读取 SRDD_Profile 角色库。"""

    resolved = resolve_dataset_source_path(profile_asset_path)
    if not resolved.exists():
        return {}
    bank: dict[str, list[str]] = {}
    with zipfile.ZipFile(resolved) as archive:
        for member in archive.namelist():
            marker = "/SRDD_Profile/"
            if marker not in member or not member.endswith(".txt"):
                continue
            suffix = member.split(marker, 1)[1]
            parts = Path(suffix).parts
            if len(parts) < 2:
                continue
            category = parts[0]
            text = archive.read(member).decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            bank.setdefault(category, []).append(text)
    return bank


def pick_profile_text(
    *,
    profile_bank: dict[str, list[str]],
    dataset_slug: str,
    role_kind: str,
    seed: int,
) -> str:
    """按数据集与角色类型稳定挑选一条 profile。"""

    fallback_pool = BUILTIN_PROFILES[role_kind]
    if not profile_bank:
        return random.Random(seed).choice(fallback_pool)
    categories = PROFILE_CATEGORY_HINTS.get(dataset_slug) or sorted(profile_bank)
    available = [category for category in categories if profile_bank.get(category)]
    if not available:
        return random.Random(seed).choice(fallback_pool)
    rng = random.Random(seed)
    category = rng.choice(sorted(available))
    return rng.choice(profile_bank[category])


def summarize_profile_bank(profile_bank: dict[str, list[str]]) -> dict[str, int]:
    """返回 profile bank 的轻量摘要。"""

    return {category: len(rows) for category, rows in sorted(profile_bank.items())}
