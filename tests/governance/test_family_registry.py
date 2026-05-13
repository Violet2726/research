"""覆盖 family 注册表与 UTF-8 编码治理规则。"""

from __future__ import annotations

from pathlib import Path
import subprocess
import tomllib

from research_experiments.families.registry import registered_family_names, validator_map
from research_experiments.tools.artifact_cleanup import RUN_VALIDATORS


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
