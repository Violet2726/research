"""统一 CLI 的共享辅助定义。"""

from __future__ import annotations

from collections.abc import Callable

ToolMain = Callable[[list[str] | None], None]

