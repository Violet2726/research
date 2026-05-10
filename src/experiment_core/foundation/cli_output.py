"""命令行 UTF-8 输出辅助。"""

from __future__ import annotations

import io
import json
import sys
from typing import Any


def configure_utf8_stdio() -> None:
    """尽量把标准输出与错误输出收口到 UTF-8。

    Windows CI 或某些本地终端会把 `sys.stdout.encoding` 设成诸如
    `cp1252`、`gbk` 等本地代码页。项目当前大量 CLI 会直接输出中文
    JSON；如果不在入口处统一收口，就会在 `print()` 阶段抛出
    `UnicodeEncodeError`。
    """

    _configure_stream("stdout")
    _configure_stream("stderr")


def emit_json(payload: Any) -> None:
    """以 UTF-8 友好的方式输出 JSON。"""

    configure_utf8_stdio()
    sys.stdout.write(json.dumps(payload, ensure_ascii=False, indent=2))
    sys.stdout.write("\n")


def _configure_stream(name: str) -> None:
    stream = getattr(sys, name, None)
    if stream is None:
        return

    try:
        stream.reconfigure(encoding="utf-8", errors="strict")
        return
    except AttributeError:
        pass
    except (ValueError, OSError):
        return

    buffer = getattr(stream, "buffer", None)
    if buffer is None:
        return

    try:
        wrapped = io.TextIOWrapper(buffer, encoding="utf-8", errors="strict", write_through=True)
    except (ValueError, OSError):
        return
    setattr(sys, name, wrapped)
