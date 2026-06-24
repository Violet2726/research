"""约束仓库文本编码与共享配置说明，防止 UTF-8 和中文注解规范回退。"""

from __future__ import annotations

import ast
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    "",
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
}
TEXT_FILENAMES = {
    ".editorconfig",
    ".env.example",
    ".gitattributes",
}
ACTIVE_TEXT_ROOTS = {
    "configs",
    "docs",
    "src",
    "tests",
}
SHARED_BENCHMARKS_ROOT = ROOT / "configs" / "core" / "shared" / "benchmarks"
SRC_ROOT = ROOT / "src" / "research_experiments"
REPLACEMENT_CHARACTER = chr(0xFFFD)
REPLACEMENT_CODEPOINT = 0xFFFD
MOJIBAKE_MARKERS = tuple(
    "".join(chr(codepoint) for codepoint in marker)
    for marker in (
        (0x7EFE, 0xFE36, 0x6F7C),  # 约束
        (0x93C2, 0x56E6, 0x6E73),  # 文本
        (0x74A7, 0x5B58, 0x69D1),  # 说明
        (0x93B6, 0x5CA9, 0x619F),  # 报告
        (0x699B, 0x6A3F, 0x8A2B),  # 默认
        (0x94DB, REPLACEMENT_CODEPOINT),
        (0x9286, REPLACEMENT_CODEPOINT),
        (0x9435, REPLACEMENT_CODEPOINT),
    )
)


def _tracked_text_paths() -> list[Path]:
    """通过 git 的 NUL 分隔输出枚举文件，避免中文路径被 shell 转义破坏。"""
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = []
    for raw_name in output.decode("utf-8").split("\0"):
        if not raw_name:
            continue
        path = ROOT / raw_name
        if path.exists() and (path.suffix.lower() in TEXT_SUFFIXES or path.name in TEXT_FILENAMES):
            paths.append(path)
    return paths


def _active_tracked_text_paths() -> list[Path]:
    paths = []
    for path in _tracked_text_paths():
        relative_parts = path.relative_to(ROOT).parts
        if not relative_parts:
            continue
        if relative_parts[0] in ACTIVE_TEXT_ROOTS or path.name in TEXT_FILENAMES:
            paths.append(path)
    return paths


def test_tracked_text_files_are_utf8_decodable() -> None:
    for path in _tracked_text_paths():
        path.read_text(encoding="utf-8")


def test_active_text_files_do_not_contain_replacement_or_mojibake_markers() -> None:
    for path in _active_tracked_text_paths():
        text = path.read_text(encoding="utf-8")
        assert REPLACEMENT_CHARACTER not in text, f"{path} 包含 Unicode 替换字符，疑似编码损坏"
        for marker in MOJIBAKE_MARKERS:
            assert marker not in text, f"{path} 包含疑似 mojibake 片段 {marker!r}"


def test_existing_src_module_docstrings_are_chinese_annotated() -> None:
    for path in SRC_ROOT.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstring = ast.get_docstring(tree)
        if docstring is None:
            continue
        assert any("\u4e00" <= char <= "\u9fff" for char in docstring), f"{path} 的模块 docstring 仍是英文旧壳"


def test_shared_benchmark_notes_are_not_ascii_only_old_shells() -> None:
    for path in SHARED_BENCHMARKS_ROOT.rglob("*.toml"):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        notes = str(payload.get("notes") or "")
        assert notes, f"{path} 缺少 notes 说明"
        assert not notes.isascii(), f"{path} 的 notes 仍是纯英文旧壳"
