"""统一 UTF-8 文本与结构化文件 I/O。"""

from __future__ import annotations

import json
import tomllib
from pathlib import Path
from typing import Any


def read_text(path: str | Path) -> str:
    """读取 UTF-8 文本。"""

    return Path(path).read_text(encoding="utf-8")


def write_text(path: str | Path, content: str) -> Path:
    """写入 UTF-8 文本，并确保父目录存在。"""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")
    return target


def write_markdown(path: str | Path, content: str) -> Path:
    """写入 Markdown 文本。"""

    return write_text(path, content)


def read_json(path: str | Path) -> dict[str, Any]:
    """读取 UTF-8 JSON。"""

    return json.loads(read_text(path))


def write_json(path: str | Path, payload: Any) -> Path:
    """写入 UTF-8 JSON。"""

    return write_text(path, json.dumps(payload, ensure_ascii=False, indent=2))


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """读取 UTF-8 JSONL。"""

    target = Path(path)
    with target.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> Path:
    """写入 UTF-8 JSONL。"""

    content = "\n".join(json.dumps(row, ensure_ascii=False) for row in rows)
    if content:
        content += "\n"
    return write_text(path, content)


def read_toml(path: str | Path) -> dict[str, Any]:
    """读取 TOML。"""

    with Path(path).open("rb") as handle:
        return tomllib.load(handle)
