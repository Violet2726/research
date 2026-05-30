"""统一 CLI 分发辅助函数。"""

from __future__ import annotations

from collections.abc import Callable


def dispatch_experiment_cli(
    *,
    family_name: str,
    cli_main_getter: Callable[[str], Callable[[list[str] | None], object]],
    argv: list[str] | None,
) -> None:
    """把实验级 CLI 子命令分发到外部提供的 family CLI 入口。"""

    cli_main_getter(family_name)(argv)
