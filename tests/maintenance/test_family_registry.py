"""Registry and UTF-8 governance tests."""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

from experiment_core.orchestration.registry import registered_family_names, validator_map
from experiment_core.tools.artifact_cleanup import RUN_VALIDATORS


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"

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
        for path in SRC.iterdir()
        if path.is_dir()
        and (path / "__init__.py").exists()
        and path.name not in {"experiment_core", "research_experiments.egg-info"}
    )
    assert list(registered_family_names()) == src_families

    payload = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    scripts = payload["project"]["scripts"]
    for family in src_families:
        assert f"{family}_cli" in scripts


def test_artifact_cleanup_validator_registry_stays_in_sync() -> None:
    assert RUN_VALIDATORS == validator_map()


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
            data = (ROOT / rel_path).read_bytes()
            try:
                data.decode("utf-8")
            except UnicodeDecodeError:
                non_utf8.append(rel_path.as_posix())
                continue
            if data.startswith(b"\xef\xbb\xbf") and rel_path.suffix.lower() != ".ps1":
                unexpected_bom.append(rel_path.as_posix())

    assert not non_utf8, non_utf8
    assert not unexpected_bom, unexpected_bom
