"""约束仓库文本编码与共享配置说明，防止 UTF-8 和中文注解规范回退。"""

from __future__ import annotations

import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".toml",
}
SHARED_BENCHMARKS_ROOT = ROOT / "configs" / "core" / "shared" / "benchmarks"


def _tracked_text_paths() -> list[Path]:
    """通过 git 的 NUL 分隔输出枚举文件，避免中文路径被 shell 转义破坏。"""
    output = subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT)
    paths = []
    for raw_name in output.decode("utf-8").split("\0"):
        if not raw_name:
            continue
        path = ROOT / raw_name
        if path.suffix.lower() in TEXT_SUFFIXES:
            paths.append(path)
    return paths


def test_tracked_text_files_are_utf8_decodable() -> None:
    for path in _tracked_text_paths():
        path.read_text(encoding="utf-8")


def test_shared_benchmark_notes_are_not_ascii_only_old_shells() -> None:
    for path in SHARED_BENCHMARKS_ROOT.rglob("*.toml"):
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        notes = str(payload.get("notes") or "")
        assert notes, f"{path} 缺少 notes 说明"
        assert not notes.isascii(), f"{path} 的 notes 仍是纯英文旧壳"
