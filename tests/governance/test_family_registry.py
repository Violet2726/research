"""覆盖 family 注册表与 UTF-8 编码治理规则。"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

from research_experiments.core.execution.rate_limits import (
    STANDARD_MAX_CONCURRENT_REQUESTS,
    STANDARD_REQUESTS_PER_MINUTE_LIMIT,
    STANDARD_TOKENS_PER_MINUTE_LIMIT,
)
from research_experiments.families.registry import (
    registered_family_names,
    registered_family_registrations,
    validator_map,
)
from research_experiments.workspace.artifact_cleanup import RUN_VALIDATORS

ROOT = Path(__file__).resolve().parents[2]
FAMILIES_SRC = ROOT / "src" / "research_experiments" / "families"

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".toml",
    ".json",
    ".yml",
    ".yaml",
    ".sh",
    ".ps1",
    ".txt",
    ".lock",
}
TEXT_FILENAMES = {
    ".editorconfig",
    ".gitattributes",
    ".gitignore",
}


def test_family_registry_matches_source_tree_and_cli_scripts() -> None:
    src_families = sorted(
        path.name
        for path in FAMILIES_SRC.iterdir()
        if path.is_dir()
        and (path / "__init__.py").exists()
        and path.name != "shared"
    )
    assert list(registered_family_names()) == src_families

    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    assert list(scripts) == ["research_cli"]


def test_every_family_exports_registration_and_family_test_directory() -> None:
    registrations = {item.family_name: item for item in registered_family_registrations()}
    for family_name in registered_family_names():
        assert family_name in registrations
        assert (FAMILIES_SRC / family_name / "registration.py").exists()
        assert not (FAMILIES_SRC / family_name / "spec.py").exists()
        assert not (FAMILIES_SRC / family_name / "family_manifest.py").exists()
        assert not (FAMILIES_SRC / family_name / "run" / "io.py").exists()
        assert (ROOT / "tests" / "families" / family_name).is_dir()


def test_every_family_registration_declares_artifact_aliases() -> None:
    for registration in registered_family_registrations():
        assert isinstance(registration.artifact_aliases, dict)
        assert registration.artifact_aliases


def test_artifact_cleanup_validator_registry_stays_in_sync() -> None:
    assert validator_map() == RUN_VALIDATORS


def test_family_experiment_configs_use_standard_runtime_limits() -> None:
    expected = {
        "max_concurrent_requests": STANDARD_MAX_CONCURRENT_REQUESTS,
        "requests_per_minute_limit": STANDARD_REQUESTS_PER_MINUTE_LIMIT,
        "tokens_per_minute_limit": STANDARD_TOKENS_PER_MINUTE_LIMIT,
    }
    mismatches: list[str] = []
    for path in sorted((ROOT / "configs" / "families").rglob("*.toml")):
        if "experiments" not in path.parts:
            continue
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        actual = {key: payload.get(key) for key in expected}
        if actual != expected:
            rel_path = path.relative_to(ROOT).as_posix()
            mismatches.append(f"{rel_path}: {actual}")

    assert not mismatches, mismatches


def test_tracked_text_files_are_utf8_and_only_powershell_keeps_bom() -> None:
    output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        check=True,
        capture_output=True,
    ).stdout
    tracked_paths = [Path(item.decode("utf-8")) for item in output.split(b"\0") if item]

    non_utf8: list[str] = []
    unexpected_bom: list[str] = []
    for rel_path in tracked_paths:
        if rel_path.name in TEXT_FILENAMES or rel_path.suffix.lower() in TEXT_SUFFIXES:
            abs_path = ROOT / rel_path
            if not abs_path.exists():
                continue
            data = abs_path.read_bytes()
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                non_utf8.append(rel_path.as_posix())
                continue
            if data.startswith(b"\xef\xbb\xbf") and rel_path.suffix.lower() != ".ps1":
                unexpected_bom.append(rel_path.as_posix())

    assert not non_utf8, non_utf8
    assert not unexpected_bom, unexpected_bom


def test_active_shared_packages_default_to_chinese_module_docstrings() -> None:
    targets = [
        ROOT / "src" / "research_experiments" / "cli",
        ROOT / "src" / "research_experiments" / "family_runtime",
        ROOT / "src" / "research_experiments" / "matrix",
        ROOT / "src" / "research_experiments" / "workspace" / "datasets",
    ]
    missing: list[str] = []
    for root in targets:
        for path in root.rglob("*.py"):
            text = path.read_text(encoding="utf-8")
            if not text.lstrip().startswith('"""'):
                missing.append(path.relative_to(ROOT).as_posix())
                continue
            docstring = text.split('"""', 2)[1]
            if not any("\u4e00" <= char <= "\u9fff" for char in docstring):
                missing.append(path.relative_to(ROOT).as_posix())
    assert not missing, missing
